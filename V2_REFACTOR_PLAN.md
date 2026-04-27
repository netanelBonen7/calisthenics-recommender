# V2 Refactor Plan

## Purpose

Refactor v1 so embedded cache storage can move from JSONL to SQLite and vector-search mechanics move behind a search port.

The stable v1 baseline is tagged as `v1-local-api-mvp`. The current backend v2 branch is `v2-sqlite-embedded-search-refactor`.

v2 preserves the current user-visible API behavior while changing where embedded records are stored and where search mechanics live.

---

## v1 Baseline

`v1-local-api-mvp` had:

- CSV raw exercise parsing.
- SQLite raw exercise import/repository.
- JSONL embedded exercise cache.
- App-layer exact top-K search.
- CLI commands.
- FastAPI local runtime.

---

## Current Backend v2 State

The current branch includes:

- SQLite embedded cache writer/reader.
- `EmbeddedExerciseSearchRepository` search port.
- `EmbeddedExerciseSearchResult`.
- JSONL exact search adapter.
- SQLite exact search adapter.
- `recommend_exercises(...)` uses the search port.
- API runtime is config-driven through TOML loaded from `CALISTHENICS_RECOMMENDER_CONFIG_PATH`.
- CLI commands support optional `--config`:
  - `build-exercise-cache`
  - `demo-recommend`
  - `debug-recommendations`
- Config can select:
  - raw exercise source: CSV or SQLite
  - embedded cache backend: JSONL or SQLite
  - embedding provider/model/dimension/prefixes
  - text builder version for cache building

Existing explicit CLI workflows are preserved where practical. The API request/response shape is unchanged.

Not included:

- No vector DB, sqlite-vec, pgvector, or FAISS.
- No Docker yet.
- No cloud deployment yet.
- No frontend merged into this backend branch.

During v2 work, an optional React/Vite demo UI prototype was explored on branch `v2-6c-demo-ui-prototype`. It is intentionally separate and should not be treated as part of the main backend v2 line.

---

## Why This Refactor Is Useful

v1 proved the local recommendation flow works, but the recommender owned too much of the search process:

```text
recommend_exercises
-> stream embedded records
-> apply equipment filter
-> compute similarity
-> keep exact top-K
```

That was fine for a JSONL MVP, but it made future storage and search backends harder to add.

v2 moves search responsibility behind a search adapter/search repository:

```text
recommend_exercises
-> build query text
-> embed query
-> ask search port for top-K candidates
-> build recommendations and explanations
```

This keeps recommendation policy in the application layer while allowing different backends to own their own search mechanics.

---

## Completed V2 Milestones

### V2.1 - SQLite Embedded Cache Storage

Status: completed.

Added SQLite-backed embedded cache writer/reader equivalent to the JSONL embedded cache.

Completed scope:

- Added SQLite storage for embedded exercise records and embedding metadata.
- Kept raw exercises and embedded records logically separate.
- Proved JSONL and SQLite embedded caches round-trip equivalent records.
- Did not add vector database or SQLite vector extension.

### V2.2 - Embedded Search Port

Status: completed.

Introduced the search abstraction for top-K embedded candidate retrieval.

Completed scope:

- Added `EmbeddedExerciseSearchRepository`.
- Added `EmbeddedExerciseSearchResult`.
- Preserved user-visible recommendation behavior.
- Did not add sqlite-vec, pgvector, FAISS, or a vector DB.

### V2.3 - JSONL Exact Search Adapter

Status: completed.

Kept backward compatibility by moving JSONL scan/filter/similarity/top-K behavior behind the search port.

Completed scope:

- JSONL adapter streams embedded records.
- Applies hard filters such as equipment in Python.
- Computes exact similarity in Python.
- Keeps top-K internally.
- Recommender no longer owns those backend-specific mechanics directly.

### V2.4 - Recommender Uses Search Port

Status: completed.

Refactored `recommend_exercises(...)` so it embeds the query and asks the search port for top-K candidates.

Completed scope:

- Query building and query embedding remain in the recommendation flow.
- Search mechanics moved behind adapters.
- Recommendation/explanation building stayed in application logic.
- Response shape and current behavior were preserved.

### V2.5 - SQLite Exact Search Adapter

Status: completed.

Implemented a SQLite-backed exact search adapter.

Completed scope:

- Reads embedded records from SQLite.
- Applies hard filters as early as practical.
- Computes exact similarity/top-K inside the adapter for now.
- Keeps future sqlite-vec/pgvector/FAISS adapters possible.
- Did not add sqlite-vec.

### V2.6A - TOML Config For API Runtime

Status: completed.

Added config-driven FastAPI runtime wiring.

Completed scope:

- API runtime reads TOML config from `CALISTHENICS_RECOMMENDER_CONFIG_PATH`.
- Config selects embedded cache backend and path.
- Config selects embedding provider/model/dimension/prefixes.
- Runtime uses wiring to build the embedding provider and search repository.
- API request/response shape stayed unchanged.

### V2.6B - CLI `--config` Support

Status: completed.

Added optional operator config support for core CLI workflows.

Completed scope:

- `build-exercise-cache` supports optional `--config`.
- `demo-recommend` supports optional `--config`.
- `debug-recommendations` supports optional `--config`.
- Config can select raw exercise source: CSV or SQLite.
- Config can select embedded cache backend: JSONL or SQLite.
- Config can select embedding provider/model/dimension/prefixes.
- Config can select text builder version for cache building.
- Existing explicit CLI workflows are preserved where practical.

---

## V2.7 - Dockerize FastAPI Runtime

Status: next.

Goal:

Dockerize only the FastAPI runtime service.

Scope:

- Add a Docker image for the backend FastAPI runtime.
- Container starts uvicorn.
- Container reads `CALISTHENICS_RECOMMENDER_CONFIG_PATH`.
- Config/cache files are mounted or copied as runtime artifacts.
- Treat the embedding cache as an input artifact.
- Do not rebuild exercise embeddings on container startup.
- Do not add cloud deployment yet.
- Do not add vector DB yet.
- Do not add frontend serving yet unless explicitly chosen later.
- Keep Docker focused on the backend runtime.

Out of scope:

- Rebuilding embeddings in the API container at startup.
- Cache-builder worker/container unless explicitly selected later.
- Cloud deployment.
- sqlite-vec, pgvector, FAISS, or vector DB integration.
- Frontend serving.

---

## How JSONL Backward Compatibility Works

JSONL compatibility works through the search port with a JSONL exact-scan adapter.

```text
JSONL embedded cache
-> stream embedded records
-> apply hard filters
-> compute exact similarity in Python
-> keep top-K
-> return search results
```

This preserves the v1 cache format while allowing `recommend_exercises(...)` to depend on search behavior rather than JSONL repository mechanics.

---

## Where Hard Filters Should Live

After search moves behind adapters, hard retrieval filters such as equipment should be applied as early as practical inside the search adapter.

Examples:

- JSONL exact search adapter filters records while streaming.
- SQLite exact search adapter filters rows as early as practical, then computes similarity/top-K internally.
- Future vector-backed adapters can push filters into the backend when supported.

Product-level recommendation policy and deterministic explanation building should stay in application logic.

---

## What Not To Do Yet

Do not add these until explicitly selected in a future milestone:

- sqlite-vec.
- pgvector.
- FAISS.
- Dedicated vector database.
- Cloud deployment.
- Frontend in the main backend branch.
- LLM-generated explanations.
- Large recommendation evaluation framework.
- Rebuilding embeddings on API startup.
- Qwen downloads in automated tests.

Do not collapse raw exercises and embedded records into one logical concept. They may eventually share a physical database, but they should remain separate concepts.

---

## Later - Interview Prep

After Docker, create a separate `INTERVIEW_PREP.md` in a separate milestone/chat.

It should cover:

- demo script
- architecture explanation
- what changed from v1 to v2
- design decisions
- tradeoffs
- likely interview questions
- future cloud path

Do not put detailed interview prep inside `V2_REFACTOR_PLAN.md`.
