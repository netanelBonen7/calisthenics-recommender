# Project Roadmap — Next Milestones

This document tracks the planned next milestones for `calisthenics-recommender`.

Use this file together with:

- `PROJECT_BRIEF_FOR_CODEX_CURRENT.md` — current architecture/source-of-truth brief
- `README.md` — public project overview and usage guide

The purpose of this roadmap is to keep future work focused and avoid mixing unrelated milestones.

---

## Current Completed State

Completed milestones:

- 10A — Local fake-cache integration test
- 10B — Cache build CLI
- 10C — Runtime recommendation CLI
- 11 — Local Sentence Transformers/Qwen embedding provider
- 12 — Real dataset CSV compatibility and smoke tests
- 13 — Recommendation debug/inspection tooling
- 14A — README overview
- 15A — Installable CLI entry points and script execution cleanup
- 15B — SQLite raw exercise database
- 15C — Build cache from SQLite

Current capabilities:

- Clean architecture with domain/application/ports/adapters separation
- Packaged CLI commands:
  - `uv run build-exercise-cache`
  - `uv run demo-recommend`
  - `uv run debug-recommendations`
  - `uv run import-exercises-to-sqlite`
- Real dataset parsing from `data/raw/calisthenics_exercises.csv`
- SQLite raw exercise import and repository under `data/db/`
- Cache building from CSV or SQLite into local JSONL under `data/cache/`
- Local JSONL embedding cache under `data/cache/`
- Fake deterministic embeddings for tests and development
- Local Qwen/Sentence Transformers embeddings for real semantic retrieval
- Recommendation debugging tools for inspecting query text, exercise text, and retrieval candidates
- Full test suite passing

---

## Roadmap Philosophy

The goal is to build a complete, interview-presentable system without spreading too thin.

Priorities:

1. Finish a coherent local/backend system.
2. Understand the architecture deeply.
3. Add API/UI/cloud only if time allows.
4. Be able to explain future cloud use cases clearly even if they are not implemented.
5. Keep each milestone narrow and test-driven.

This project should show a clear evolution:

```text
CSV dataset
→ clean architecture core
→ precomputed JSONL embedding cache
→ local Qwen embeddings
→ SQLite adapter
→ FastAPI backend
→ optional frontend
→ optional Docker/cloud
```

---

## Milestone 15B — SQLite Raw Exercise Database

Status: completed.

### Goal

Introduce SQLite as a local database adapter for raw exercises.

Expected flow:

```text
CSV
→ SQLite import
→ SQLiteExerciseRepository
→ existing cache/recommendation pipeline
```

### Scope

- Add SQLite schema for raw exercises.
- Add an import command from CSV to SQLite.
- Add `SQLiteExerciseRepository`.
- Prove it implements the existing `ExerciseRepository` port.
- Add tests showing SQLite repository returns the same `Exercise` objects as CSV for equivalent data.
- Keep embeddings in JSONL for now.
- Do not move the embedding cache into SQLite yet.
- Do not change core recommendation logic.
- Do not change ranking/filtering behavior.

### Interview Value

This milestone demonstrates that the core application is not tied to CSV files.

The same `ExerciseRepository` port can be backed by:

```text
CSV
SQLite
PostgreSQL later
external database later
```

This supports the explanation that infrastructure can evolve without rewriting the recommender core.

---

## Milestone 15C — Build Cache From SQLite

Status: completed.

### Goal

Allow cache building from SQLite, not only CSV.

Expected flow:

```text
SQLiteExerciseRepository
→ build_embedded_exercise_cache
→ JSONL cache
```

### Scope

- Extend the cache-build CLI cleanly to choose CSV or SQLite input.
- Keep CLI code thin.
- Keep JSONL cache as the embedding storage.
- Use local-deterministic embeddings in automated tests.
- Do not run Qwen in tests.
- Do not move embedded vectors into SQLite yet.

### Interview Value

This proves that the offline cache-generation workflow can operate on a database-backed repository without changing the application workflow.

---

## Milestone 16 — FastAPI Backend Adapter

### Goal

Expose the recommender through an HTTP API.

Expected flow:

```text
POST /recommend
→ UserRequest
→ existing recommender core
→ JSON response
```

### Scope

- Add FastAPI as an adapter layer only.
- Add request/response models.
- Add `/recommend`.
- Use existing application logic.
- Do not put recommendation logic in the endpoint.
- Tests should use fake/local deterministic providers.
- Real Qwen usage remains manual/local.

### Interview Value

This turns the recommender from a CLI-only system into a backend service while preserving clean architecture boundaries.

---

## Milestone 17 — Local API Demo With Real Cache

### Goal

Run the FastAPI backend locally against a real Qwen cache.

Expected flow:

```text
real Qwen JSONL cache
→ FastAPI backend
→ POST /recommend
→ JSON recommendations
```

### Scope

- Provide local run commands.
- Provide sample request commands using curl/Postman-style examples.
- Verify response shape.
- Keep this local.
- Do not add Docker/cloud yet.

### Interview Value

This gives a realistic local backend demo that can be shown or explained in interviews.

---

## Milestone 18 — Optional Frontend UI

### Goal

Add a simple user-facing UI if time allows.

Preferred direction:

```text
React + Vite frontend
→ FastAPI backend
→ existing recommender core
```

### Scope

- Form inputs:
  - target family
  - goal
  - current level
  - available equipment
  - limit
- Recommendation cards.
- Loading state.
- Error state.
- Empty-results state.
- Keep UI thin.
- Backend remains the source of recommendation logic.

### Interview Value

This turns the backend service into a more visual demo and shows full-stack awareness.

---

## Milestone 19 — Docker Runtime Service

### Goal

Containerize the FastAPI runtime service.

### Scope

- Dockerize the runtime API service.
- Treat the embedding cache as an input artifact.
- Do not rebuild exercise embeddings on container startup.
- Optional later: separate cache-builder container/job.

### Preferred Architecture

```text
offline cache build
→ JSONL cache artifact

runtime container
→ loads existing cache
→ embeds only user query
→ returns recommendations
```

### Interview Value

This shows deployment readiness while preserving the offline-vs-runtime separation.

---

## Milestone 20 — Optional Cloud Deployment

### Goal

Explain or implement cloud deployment if time allows.

### Preferred Cloud Story

- Containerized FastAPI service on a managed container platform.
- Precomputed embedding cache stored as an artifact, for example object storage.
- Offline cache-builder job can rebuild embeddings when the dataset changes.
- Future database path:
  - SQLite
  - PostgreSQL
  - pgvector/vector search

### Example Architecture

```text
FastAPI container
→ managed container service
→ loads embedding cache artifact from object storage
→ serves POST /recommend
```

### Interview Value

This demonstrates that the local architecture has a believable path to cloud deployment without rewriting the recommendation core.

---

## Milestone 21 — Recommendation Quality Tuning

### Goal

Return to semantic/ranking quality after the complete local/backend system exists.

### Possible Work

- Target-family hard filtering or boosting.
- Improved query text builder.
- Improved exercise text builder.
- Query/document prefixes for Qwen.
- Difficulty/progression-aware filtering.
- Quality/sanity evaluation scenarios.
- Embedding model comparison.
- Better scoring calibration.

### Interview Value

This shows awareness that recommendation systems need evaluation and iteration, but avoids blocking backend completeness on semantic perfection.

---

## Do Not Do Yet

Avoid these until the core local/backend system is complete:

- Do not add a vector database too early.
- Do not add Kubernetes.
- Do not split into microservices.
- Do not perfect semantic ranking before API/database milestones.
- Do not mix SQLite, FastAPI, frontend, Docker, and cloud in one milestone.
- Do not run Qwen in automated tests.
- Do not put recommendation logic inside CLI/API/UI layers.

---

## Interview Narrative

A concise explanation of the system evolution:

> I started with a clean architecture recommender core and a CSV adapter because it was the simplest transparent data source. Then I added an offline embedding-cache workflow so exercise embeddings are precomputed and runtime recommendation only embeds the user query. I added local fake embeddings for deterministic tests and Qwen/Sentence Transformers for real local semantic retrieval. The next step is SQLite, which proves the repository boundary by replacing CSV with a local database adapter. From there, the same core can be exposed through FastAPI, then optionally a frontend, Docker, and cloud deployment.

Future cloud explanation:

> The cloud version would separate offline and online work. A cache-builder job would generate the embedding cache and store it as an artifact, while a containerized FastAPI service would load that cache and serve recommendations. If the system needed production persistence, the SQLite adapter could evolve into PostgreSQL, and later pgvector or another vector-search backend could replace the local exact scan.
