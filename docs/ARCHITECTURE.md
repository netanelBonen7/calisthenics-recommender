# Architecture

## Overview

`calisthenics-recommender` is a local embedding-based calisthenics exercise recommender backend. It is designed around a simple boundary: offline cache building is separate from online recommendation serving.

During offline cache building, raw exercise records are converted into searchable exercise text, embedded once, and written to an embedded cache. At runtime, the API embeds only the user query and searches the precomputed exercise embeddings. The embedded cache can be stored as JSONL or SQLite, and FastAPI serves recommendations from that cache.

This keeps request-time work small and makes the runtime reproducible: a running API instance needs a config file, an embedded cache, and an embedding provider that matches how the cache was built.

The project also supports a SQLite-only operator workflow for incremental cache maintenance. Raw SQLite changes are detected with lightweight triggers that record pending work, and Python application code later regenerates or deletes derived cache entries.

## High-Level Data Flow

Offline cache build:

```text
CSV or SQLite raw exercises
-> ExerciseRepository
-> build_exercise_text(...)
-> EmbeddingProvider.embed(exercise_text)
-> EmbeddedExercise
-> JSONL or SQLite embedded cache
```

Runtime recommendation:

```text
POST /recommend
-> RecommendRequest
-> UserRequest
-> build_query_text(...)
-> EmbeddingProvider.embed(query_text)
-> EmbeddedExerciseSearchRepository.search(...)
-> Recommendation objects
-> RecommendResponse
```

Runtime embeds only the user query. It does not re-embed every exercise per request.

SQLite pending update processing:

```text
raw SQLite exercise INSERT/UPDATE/DELETE
-> SQLite trigger records pending row by exercise_id
-> process-pending-embedding-updates
-> ProcessPendingEmbeddingUpdatesWorkflow
-> build_exercise_text(...)
-> EmbeddingProvider.embed(exercise_text)
-> SQLite embedded cache upsert/delete
```

## Layers And Responsibilities

The project follows a clean / hexagonal architecture style.

- `domain/`: core validated data objects such as exercises, user requests, embedded exercises, pending embedding updates, search results, and recommendations.
- `application/`: use cases, query and exercise text building, deterministic filtering, deterministic explanations, recommendation orchestration, and pending embedding update processing.
- `ports/`: protocols for repositories, embedded search, embedded cache updates, pending update repositories, exercise lookup, and embedding providers.
- `adapters/`: concrete implementations for CSV, SQLite, JSONL, local deterministic embeddings, fake test embeddings, and Sentence Transformers.
- `api/`: FastAPI request/response models, route handling, and runtime app creation.
- `cli/`: operator and developer commands for importing exercises, building caches, processing pending embedding updates, running demos, and debugging recommendation behavior.
- `wiring/config`: selects implementations from TOML config while keeping the API and CLI layers thin.

The application layer owns recommendation and cache-maintenance behavior. Infrastructure details such as CSV versus SQLite, JSONL versus SQLite cache, and local deterministic versus Sentence Transformers embeddings stay behind ports and adapters.

## Ports And Adapter Boundaries

The main ports exist to keep recommendation logic independent of storage and model infrastructure:

- `ExerciseRepository`: reads raw exercise records from a source such as CSV or SQLite.
- `ExerciseLookupRepository`: looks up a raw exercise by stable `exercise_id` for incremental SQLite update processing.
- `EmbeddedExerciseRepository`: writes or reads embedded exercise records as cache artifacts.
- `EmbeddedExerciseSearchRepository`: searches embedded exercises for the best matches to an embedded query.
- `EmbeddedExerciseCacheUpdater`: incrementally upserts or deletes SQLite embedded cache entries by `exercise_id`.
- `PendingEmbeddingUpdateRepository`: reads, marks, and records failures for pending SQLite embedding updates.
- `EmbeddingProvider`: embeds exercise text during cache building and embeds query text during runtime recommendation.

These abstractions let the recommendation use case work without knowing whether the embedded cache is JSONL or SQLite. API users do not choose infrastructure backends in request bodies; operator config chooses the backends before the API starts.

## Storage Model

The project separates raw exercise data from embedded exercise caches.

Raw exercises are source data. They describe exercises, equipment, categories, families, and related metadata before embedding. Current raw storage options are:

- Raw CSV.
- Raw SQLite DB.

Embedded exercise caches are derived artifacts. They contain exercise records plus precomputed embedding vectors and metadata needed for runtime search. Current embedded cache options are:

- Embedded JSONL cache.
- Embedded SQLite cache.

Generated data under `data/` is not committed. A typical local setup keeps raw input under `data/raw/`, imported raw SQLite databases under `data/db/`, and embedded caches under `data/cache/`.

## Stable Exercise Identity

Every raw exercise has a required stable `exercise_id`. The ID is the logical identity of the exercise and is used to link raw source data to derived embedded cache entries.

The important distinction is:

```text
exercise_id = stable identity
name / description / equipment / categories / families = editable exercise content
embedding = derived cache/index data
```

A display name can change while the `exercise_id` remains the same. In that case, the derived embedding should be refreshed for the same cache identity. This avoids treating renames as ambiguous delete-plus-create operations.

Embeddings are intentionally not stored inside raw exercise rows. They are opaque, bulky, model-dependent, and not meant to be edited by humans. The embedded cache owns them as generated search/index data.

## SQLite Pending Embedding Updates

For SQLite raw exercises and a SQLite embedded cache, the project supports incremental cache maintenance through `pending_embedding_updates`.

The pending update table stores durable, retryable work:

```text
exercise_id
operation       # upsert or delete
version
attempt_count
last_attempted_at
last_error
created_at
updated_at
```

The row represents the final desired cache state for an `exercise_id`:

- `upsert`: the SQLite embedded cache should contain the current raw exercise.
- `delete`: the SQLite embedded cache should not contain that exercise.

SQLite triggers on the raw `exercises` table record pending rows after insert, update, and delete. Triggers do not call embedding providers and do not modify embedded cache tables.

This separation matters because embedding generation is application logic. It depends on Python text-building code, configured embedding providers, metadata compatibility, failure handling, and retries. SQLite is only responsible for detecting that raw data changed.

## Pending Update Processing Workflow

The operator command:

```powershell
uv run process-pending-embedding-updates --config .\runtime.toml
```

runs the application workflow that drains pending SQLite updates.

For an `upsert` row, the workflow:

```text
lookup raw exercise by exercise_id
-> build exercise text
-> generate embedding
-> upsert SQLite embedded cache row
-> clear pending row after success
```

For a `delete` row, the workflow:

```text
delete SQLite embedded cache row by exercise_id
-> clear pending row after success
```

If an `upsert` row points to a raw exercise that no longer exists, the workflow treats it like a delete. This makes stale pending upserts self-healing.

Failures are recorded on the pending row and the row remains retryable. The workflow can continue processing other rows and returns a summary of seen, processed, failed, and remaining work.

## Version-Safe Pending Row Clearing

Pending rows include a `version` field. Every time a trigger or rebuild helper coalesces a new pending state for the same `exercise_id`, the version increments.

The processor clears a pending row only if both `exercise_id` and `version` still match the row it originally read. This prevents losing newer work.

Example:

```text
processor reads pull-up version 3
raw exercise changes while processor is embedding
trigger updates pull-up to version 4
processor finishes version 3
processor must not delete version 4
```

The repository therefore clears pending work with a version-checked operation rather than deleting by `exercise_id` alone.

## Metadata Compatibility

Before processing pending rows, the workflow validates that the existing SQLite embedded cache metadata matches the current configuration.

The important compatibility values are:

- embedding model
- embedding dimension
- text builder version

If metadata is missing or incompatible, processing fails before mutating the cache and the operator should run a full cache rebuild. This prevents mixing embeddings generated with incompatible models or text-building logic.

Changes to text builder logic, embedding model, embedding dimension, or embedding configuration are handled by metadata validation and full rebuilds, not by SQLite triggers. Triggers detect raw SQLite row changes only.

## Search And Filtering

The v1 application-layer exact top-K search was refactored behind an embedded search port. The current JSONL and SQLite search adapters still perform exact search over cached embeddings; they are not approximate nearest neighbor or vector-index search implementations.

Equipment is a hard filter. If the user lists available equipment, recommendations must be compatible with that equipment.

`target_family` is currently semantic input to query construction and explanation text. It is not a strict hard family filter. Future vector DBs or search backends could push filtering and search further down into the backend when that becomes useful.

## Config-Driven Runtime

TOML config selects runtime and cache-building infrastructure:

- Raw exercise source for cache building.
- Embedded cache backend and path.
- Embedding provider, model, dimension, and prefixes.
- Text builder version.

FastAPI runtime reads `CALISTHENICS_RECOMMENDER_CONFIG_PATH`. CLI commands can use optional `--config`. Explicit CLI flags can override config values where that command supports overrides.

The `/recommend` request body does not include backend choice. Backend selection is an operator concern, not an API user concern.

Incremental pending update processing is intentionally SQLite-only. CSV raw source and JSONL embedded cache remain supported for full rebuilds, but not for trigger-backed pending update processing.

## Docker Runtime Boundary

Docker packages the FastAPI runtime only. The container starts uvicorn, reads `CALISTHENICS_RECOMMENDER_CONFIG_PATH`, and expects config and cache files to be mounted as runtime artifacts.

The container does not rebuild exercise embeddings on startup and does not process pending embedding updates. Embedded caches are built offline before the container runs. Pending update processing is an explicit operator workflow, separate from API serving.

The default Docker smoke path uses local deterministic embeddings and a SQLite embedded cache so it does not depend on model downloads. The same image can run with Sentence Transformers/Qwen if the runtime config selects that provider and the model is available or downloadable in the container environment.

## Why This Design Is Useful

The design keeps expensive offline work separate from latency-sensitive online serving. That makes the runtime easier to reproduce and easier to test.

Storage and search backends can change behind adapters without rewriting recommendation orchestration. Tests can use fake or local deterministic embeddings without downloading a model. The same application flow can support local development now and more production-oriented infrastructure later.

Stable exercise identity makes raw data updates and cache maintenance safer. Trigger-backed pending updates provide durable, retryable incremental work without adding RabbitMQ, background workers, or request-time cache mutation.

## Current Limitations

- No vector DB or approximate nearest neighbor index yet.
- No cloud deployment yet.
- No frontend in the main backend branch.
- No auth, users, or persisted recommendation history.
- No structured difficulty or progression model.
- Recommendation quality has not been deeply evaluated.
- `target_family` is semantic input, not a strict hard filter.
- Pending update processing is SQLite-only and operator-driven; there is no background worker or hosted queue.

## Future Evolution

Natural next steps include pgvector or another vector DB, a cache-builder worker or job, a raw exercise admin/update flow, optional Docker ops profiles for pending update processing, and cloud deployment.

Separate services should be introduced only if scale or team boundaries justify the extra operational complexity. An optional UI branch exists separately, but it is not part of the main backend branch.
