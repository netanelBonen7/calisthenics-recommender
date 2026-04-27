# Architecture

## Overview

`calisthenics-recommender` is a local embedding-based calisthenics exercise recommender backend. It is designed around a simple boundary: offline cache building is separate from online recommendation serving.

During offline cache building, raw exercise records are converted into searchable exercise text, embedded once, and written to an embedded cache. At runtime, the API embeds only the user query and searches the precomputed exercise embeddings. The embedded cache can be stored as JSONL or SQLite, and FastAPI serves recommendations from that cache.

This keeps request-time work small and makes the runtime reproducible: a running API instance needs a config file, an embedded cache, and an embedding provider that matches how the cache was built.

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

## Layers And Responsibilities

The project follows a clean / hexagonal architecture style.

- `domain/`: core validated data objects such as exercises, user requests, embedded exercises, search results, and recommendations.
- `application/`: use cases, query and exercise text building, deterministic filtering, deterministic explanations, and recommendation orchestration.
- `ports/`: protocols for repositories, embedded search, and embedding providers.
- `adapters/`: concrete implementations for CSV, SQLite, JSONL, local deterministic embeddings, fake test embeddings, and Sentence Transformers.
- `api/`: FastAPI request/response models, route handling, and runtime app creation.
- `cli/`: operator and developer commands for importing exercises, building caches, running demos, and debugging recommendation behavior.
- `wiring/config`: selects implementations from TOML config while keeping the API and CLI layers thin.

The application layer owns recommendation behavior. Infrastructure details such as CSV versus SQLite, JSONL versus SQLite cache, and local deterministic versus Sentence Transformers embeddings stay behind ports and adapters.

## Ports And Adapter Boundaries

The main ports exist to keep recommendation logic independent of storage and model infrastructure:

- `ExerciseRepository`: reads raw exercise records from a source such as CSV or SQLite.
- `EmbeddedExerciseRepository`: writes or reads embedded exercise records as cache artifacts.
- `EmbeddedExerciseSearchRepository`: searches embedded exercises for the best matches to an embedded query.
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

## Docker Runtime Boundary

Docker packages the FastAPI runtime only. The container starts uvicorn, reads `CALISTHENICS_RECOMMENDER_CONFIG_PATH`, and expects config and cache files to be mounted as runtime artifacts.

The container does not rebuild exercise embeddings on startup. Embedded caches are built offline before the container runs.

The default Docker smoke path uses local deterministic embeddings and a SQLite embedded cache so it does not depend on model downloads. The same image can run with Sentence Transformers/Qwen if the runtime config selects that provider and the model is available or downloadable in the container environment.

## Why This Design Is Useful

The design keeps expensive offline work separate from latency-sensitive online serving. That makes the runtime easier to reproduce and easier to test.

Storage and search backends can change behind adapters without rewriting recommendation orchestration. Tests can use fake or local deterministic embeddings without downloading a model. The same application flow can support local development now and more production-oriented infrastructure later.

## Current Limitations

- No vector DB or approximate nearest neighbor index yet.
- No cloud deployment yet.
- No frontend in the main backend branch.
- No auth, users, or persisted recommendation history.
- No structured difficulty or progression model.
- Recommendation quality has not been deeply evaluated.
- `target_family` is semantic input, not a strict hard filter.

## Future Evolution

Natural next steps include pgvector or another vector DB, a cache-builder worker or job, a raw exercise admin/update flow, and cloud deployment.

Separate services should be introduced only if scale or team boundaries justify the extra operational complexity. An optional UI branch exists separately, but it is not part of the main backend branch.
