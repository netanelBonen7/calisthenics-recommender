# PROJECT_BRIEF_FOR_CODEX_CURRENT.md

## 1. Purpose

This file is the current source of truth for continuing `calisthenics-recommender` with Codex.

The stable v1 baseline is tagged as `v1-local-api-mvp`. The current backend v2 branch is `v2-sqlite-embedded-search-refactor`.

Important rule:

> Implement only the explicitly requested task. Do not mix unrelated roadmap items into one change.

---

## 2. Current Project Summary

The project is a local calisthenics exercise recommender.

The current backend v2 branch includes:

- CSV raw exercise parsing.
- SQLite raw exercise import/repository.
- JSONL embedded exercise cache.
- SQLite embedded exercise cache writer/reader.
- Offline exercise embedding generation.
- Runtime query embedding only.
- `EmbeddedExerciseSearchRepository` search port.
- `EmbeddedExerciseSearchResult`.
- JSONL exact search adapter.
- SQLite exact search adapter.
- `recommend_exercises(...)` uses the search port.
- Deterministic equipment filtering as part of search/retrieval behavior.
- FastAPI local runtime via `uvicorn`.
- TOML config-driven API runtime loaded from `CALISTHENICS_RECOMMENDER_CONFIG_PATH`.
- CLI commands for import, cache build, demo recommendations, and debugging.
- Optional `--config` support for `build-exercise-cache`, `demo-recommend`, and `debug-recommendations`.

The API request/response shape is unchanged from v1.

---

## 3. v1 Baseline

`v1-local-api-mvp` is the stable v1 tag.

v1 had:

- CSV raw exercise parsing.
- SQLite raw exercise import/repository.
- JSONL embedded exercise cache.
- App-layer exact top-K search.
- CLI commands.
- FastAPI local runtime.

v1 did not include:

- SQLite embedded cache.
- Search port or search repository.
- Vector database or vector extension.
- Docker.
- Frontend.
- Cloud deployment.

---

## 4. User Input

The recommender receives:

```text
target_family
goal
current_level
available_equipment
limit
```

Example:

```json
{
  "target_family": "Pull-up",
  "goal": "I want to build pulling strength and unlock harder pull-up variations.",
  "current_level": "I can do 5 strict pull-ups, but the last reps are slow.",
  "available_equipment": ["Bar"],
  "limit": 5
}
```

Design notes:

- `target_family` anchors semantic retrieval.
- `goal` captures user intent.
- `current_level` gives semantic progression context.
- `available_equipment` is a deterministic hard filter.
- `limit` controls response size and must be positive.

---

## 5. Output

Each recommendation returns:

```text
exercise_name
match_score
reason
required_equipment
category_family
```

Example:

```json
{
  "exercise_name": "Pull Up Negative",
  "match_score": 87,
  "reason": "Recommended because it matched your Pull-up target family through retrieval, belongs to the Pull-up families, falls under the Upper Body Pull categories, and requires Bar.",
  "required_equipment": ["Bar"],
  "category_family": {
    "categories": ["Upper Body Pull"],
    "families": ["Pull-up"]
  }
}
```

Rules:

- Explanations are deterministic and grounded in dataset fields.
- Do not use an LLM to generate explanations.
- Do not invent fields that are not in the dataset.

---

## 6. Dataset Fields

The raw exercise dataset uses:

```text
name
description
muscle_groups
families
materials
categories
```

These become validated `Exercise` objects.

The CSV adapter supports real dataset JSON-list fields and simpler semicolon-style list fields used in tests.

---

## 7. Current Pipelines

### 7.1 Raw Exercise Input

```text
CSV file
or SQLite raw exercise DB
-> ExerciseRepository.iter_exercises()
-> Iterable[Exercise]
```

### 7.2 Offline Embedded Cache Build

```text
ExerciseRepository.iter_exercises()
-> build_exercise_text(exercise)
-> EmbeddingProvider.embed(exercise_text)
-> EmbeddedExercise(exercise, embedding)
-> JSONL or SQLite embedded cache writer
```

Exercise embeddings are built offline. The cache is derived data.

### 7.3 Runtime Recommendation

```text
UserRequest
-> build_query_text(user_request)
-> EmbeddingProvider.embed(query_text)
-> EmbeddedExerciseSearchRepository.search(...)
-> JSONL or SQLite exact search adapter
-> build deterministic recommendations
```

Runtime embeds only the user query. It must not embed all exercises per request.

---

## 8. Architecture Boundaries

### 8.1 Layer Separation

The project keeps a domain/application/ports/adapters/api/cli separation:

```text
domain/
application/
ports/
adapters/
api/
cli/
```

Keep API and CLI thin. They should parse inputs, load config, call wiring, invoke application use cases, and present output.

Recommendation logic stays in the application layer.

Backend, search, and storage choices live behind ports/adapters/wiring/config.

### 8.2 Raw exercises and embedded records are separate

Raw exercises are source data. Embedded exercise records are derived data.

Current physical storage:

```text
raw CSV file
raw SQLite DB
embedded JSONL cache
embedded SQLite cache
```

Do not collapse raw exercises and embedded records into one logical concept.

### 8.3 Repository and search ports

Current ports include:

```python
class ExerciseRepository(Protocol):
    def iter_exercises(self) -> Iterable[Exercise]:
        ...

class EmbeddedExerciseRepository(Protocol):
    def iter_embedded_exercises(self) -> Iterable[EmbeddedExercise]:
        ...

class EmbeddedExerciseSearchRepository(Protocol):
    ...
```

Do not revert streaming repository ports to list-returning methods.

### 8.4 Search mechanics

v1 performed exact top-K in the application layer.

Current v2 moves backend-specific search mechanics behind `EmbeddedExerciseSearchRepository` while keeping recommendation policy and explanation building in application logic.

Existing search adapters:

- JSONL exact search adapter.
- SQLite exact search adapter.

Do not add sqlite-vec, pgvector, FAISS, or a vector database unless explicitly requested.

### 8.5 Tests must stay local and deterministic

Automated tests must not:

- download Qwen
- call external embedding APIs
- require external services
- depend on real generated cache artifacts under `data/`

Use fake, local deterministic, or injected embedding providers in tests.

---

## 9. Layers And Responsibilities

### 9.1 Domain

Key files:

```text
domain/exercise.py
domain/user_request.py
domain/recommendation.py
domain/embedded_exercise.py
domain/embedded_exercise_search_result.py
domain/types.py
```

Responsibilities:

- define and validate core data objects
- avoid infrastructure details

### 9.2 Application

Key files:

```text
application/query_builder.py
application/exercise_text_builder.py
application/filters.py
application/similarity.py
application/retriever.py
application/explanation_builder.py
application/recommend_exercises.py
application/embedded_exercise_builder.py
application/embedded_exercise_cache_workflow.py
```

Responsibilities:

- build query text
- build exercise text
- build deterministic explanations
- orchestrate recommendation runtime through the search port
- build embedded exercises from raw exercises
- orchestrate streaming cache build workflow

### 9.3 Ports

Key files:

```text
ports/exercise_repository.py
ports/embedded_exercise_repository.py
ports/embedded_exercise_search_repository.py
ports/embedding_provider.py
```

### 9.4 Adapters

Key files:

```text
adapters/csv_exercise_repository.py
adapters/sqlite_exercise_repository.py
adapters/local_embedded_exercise_cache.py
adapters/sqlite_embedded_exercise_cache.py
adapters/jsonl_embedded_exercise_search_repository.py
adapters/sqlite_embedded_exercise_search_repository.py
adapters/local_deterministic_embedding_provider.py
adapters/fake_embedding_provider.py
adapters/sentence_transformer_embedding_provider.py
```

Responsibilities:

- stream raw exercises from CSV or SQLite
- write/read local embedded exercise JSONL cache
- write/read local embedded exercise SQLite cache
- provide JSONL and SQLite exact search implementations
- provide fake deterministic embeddings for tests
- provide local deterministic embeddings for development
- provide Sentence Transformers embeddings for real local retrieval

### 9.5 API

Key files:

```text
api/app.py
api/models.py
api/runtime.py
api/main.py
```

Endpoints:

```text
GET /health
POST /recommend
```

Local runtime:

```powershell
$env:CALISTHENICS_RECOMMENDER_CONFIG_PATH = ".\runtime.toml"
uv run uvicorn calisthenics_recommender.api.main:app --reload --host 127.0.0.1 --port 8000
```

Runtime config is loaded from:

```text
CALISTHENICS_RECOMMENDER_CONFIG_PATH
```

The runtime TOML config selects:

- embedded cache backend: JSONL or SQLite
- cache path
- embedding provider/model/dimension/prefixes

Supported runtime embedding providers:

```text
local-deterministic
sentence-transformer
```

### 9.6 CLI

Packaged CLI commands:

```text
import-exercises-to-sqlite
build-exercise-cache
demo-recommend
debug-recommendations
```

Commands with optional `--config`:

```text
build-exercise-cache
demo-recommend
debug-recommendations
```

CLI config can select:

- raw exercise source: CSV or SQLite
- embedded cache backend: JSONL or SQLite
- embedding provider/model/dimension/prefixes
- text builder version for cache building

CLI code should wire adapters and application functions together. It should not contain core recommendation logic.

---

## 10. Data And Cache Convention

Raw CSV input:

```text
data/raw/
```

Raw SQLite databases:

```text
data/db/
```

Embedded JSONL or SQLite caches:

```text
data/cache/
```

These are local artifacts. Generated data should not be committed.

---

## 11. Completed Milestones

Completed v1 milestones:

```text
0 - Repository and project setup
1 - Domain models and validation
2 - Text builders
3 - Ports and fake embedding provider
4 - Filtering and similarity primitives
5 - Brute-force retriever
6 - Explanation builder and response construction
7 - End-to-end fake recommender
7.5 - Runtime uses precomputed EmbeddedExercise objects
8 - CSV exercise repository
8.5 - Streaming EmbeddedExerciseRepository and bounded top-K retrieval
8.6 - Streaming raw ExerciseRepository and CSV streaming
9A - Embedded exercise builder
9B - Local embedded exercise JSONL cache
9C - Streaming embedded exercise cache build workflow
10A - Local fake-cache integration test
10B - Cache build CLI
10C - Runtime recommendation CLI
11 - Local Sentence Transformers/Qwen embedding provider
12 - Real dataset CSV compatibility and smoke tests
13 - Recommendation debug/inspection tooling
14A - README overview
15A - Installable CLI entry points and script execution cleanup
15B - SQLite raw exercise database
15C - Build cache from SQLite
16 - FastAPI backend adapter
17 - Local API demo with real cache
```

Completed v2 milestones:

```text
V2.1 - SQLite embedded cache storage
V2.2 - Embedded search port
V2.3 - JSONL exact search adapter
V2.4 - Recommender uses search port
V2.5 - SQLite exact search adapter
V2.6A - TOML config for API runtime
V2.6B - CLI --config support
```

---

## 12. Not Implemented

The current backend v2 branch does not implement:

- Docker.
- Cloud deployment.
- Vector DB, vector extension, sqlite-vec, pgvector, or FAISS.
- Frontend in the main backend branch.
- Auth, users, or persisted recommendation history.
- LLM-generated explanations.

During v2 work, an optional React/Vite demo UI prototype was explored on branch `v2-6c-demo-ui-prototype`. It is optional side work and is not merged into the main backend v2 line.

---

## 13. Next Direction

The next milestone is `V2.7 - Dockerize FastAPI Runtime`.

Docker should focus only on the backend FastAPI runtime:

- container starts uvicorn
- container reads `CALISTHENICS_RECOMMENDER_CONFIG_PATH`
- config/cache files are mounted or copied as runtime artifacts
- no embedding rebuild on container startup
- no cloud deployment yet
- no vector DB yet
- no frontend serving yet unless explicitly chosen later

After Docker, create a separate `INTERVIEW_PREP.md` in a separate milestone/chat. Do not put detailed interview prep in `V2_REFACTOR_PLAN.md`.

---

## 14. Technology Stack

Current stack:

```text
Python >=3.11,<3.13
Pydantic v2
pytest
ruff
FastAPI
uvicorn
numpy
sentence-transformers
standard library csv/json/pathlib/logging/typing/heapq/sqlite3/tomllib
```

Do not add dependencies unless explicitly approved.

---

## 15. Workflow Rules

When implementing future work:

1. Inspect the current code before editing.
2. Add focused tests for behavior changes.
3. Keep source changes scoped to the requested task.
4. Run relevant tests.
5. Run full `uv run pytest` for meaningful code changes.
6. Run `uv run ruff check .` when Python code changes.
7. Do not commit unless explicitly told.
8. Report files changed, verification commands, and tradeoffs.

For docs-only work, do not change source code, tests, or dependencies.

---

## 16. Anti-Patterns To Avoid

Avoid:

```text
embedding all exercises per user request
returning list[...] from repository ports
direct CSV/cache/SQLite access inside API or CLI recommendation flows
recommendation logic inside scripts, CLI, API, or UI
real model downloads in automated tests
hardcoded absolute paths
hardcoded API keys
mixing Docker, frontend, vector DB, and cloud deployment into the backend refactor
implying Docker exists before V2.7 implements it
implying frontend is part of the main backend branch
recreating PROJECT_ROADMAP_NEXT.md
```
