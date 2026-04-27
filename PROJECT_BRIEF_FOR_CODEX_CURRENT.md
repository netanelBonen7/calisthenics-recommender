# PROJECT_BRIEF_FOR_CODEX_CURRENT.md

## 1. Purpose

This file is the current source of truth for continuing `calisthenics-recommender` with Codex.

v1 is closed. The current codebase is a working local API MVP. Future work should treat v1 as the stable baseline and follow `V2_REFACTOR_PLAN.md` for the next engineering phase.

Important rule:

> Implement only the explicitly requested task. Do not mix unrelated roadmap items into one change.

---

## 2. Current v1 Summary

The project is a local calisthenics exercise recommender.

v1 uses:

- CSV or SQLite for raw exercise input.
- JSONL for the embedded exercise cache.
- Offline exercise embedding generation.
- Runtime query embedding only.
- Application-layer exact top-K over streamed embedded records.
- Deterministic equipment filtering before ranking.
- FastAPI local runtime via `uvicorn`.
- CLI commands for import, cache build, demo recommendations, and debugging.

v1 does not include:

- SQLite embedded cache.
- Search port or search repository.
- Vector database or vector extension.
- Docker.
- Frontend.
- Cloud deployment.

---

## 3. v1 User Input

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

## 4. v1 Output

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
- Do not use an LLM to generate explanations in v1.
- Do not invent fields that are not in the dataset.

---

## 5. Dataset Fields

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

## 6. Current Pipelines

### 6.1 Raw Exercise Input

```text
CSV file
or SQLite raw exercise DB
-> ExerciseRepository.iter_exercises()
-> Iterable[Exercise]
```

SQLite currently stores raw exercises only. It does not store embedded exercise records in v1.

### 6.2 Offline Embedded Cache Build

```text
ExerciseRepository.iter_exercises()
-> build_exercise_text(exercise)
-> EmbeddingProvider.embed(exercise_text)
-> EmbeddedExercise(exercise, embedding)
-> LocalEmbeddedExerciseCache.write_embedded_exercises(...)
-> JSONL embedded cache
```

Exercise embeddings are built offline. The cache is derived data.

### 6.3 Runtime Recommendation

```text
UserRequest
-> build_query_text(user_request)
-> EmbeddingProvider.embed(query_text)
-> LocalEmbeddedExerciseRepository.iter_embedded_exercises()
-> application-layer equipment filtering
-> application-layer exact top-K retrieval
-> build deterministic recommendations
```

Runtime embeds only the user query. It must not embed all exercises per request.

---

## 7. Core Architecture Decisions

### 7.1 Raw exercises and embedded records are separate

Raw exercises are source data. Embedded exercise records are derived data.

Current physical storage:

```text
raw CSV file
raw SQLite DB
embedded JSONL cache
```

Future v2 may add SQLite storage for embedded records, but the logical separation should remain.

### 7.2 Repository ports are streaming-friendly

Current ports:

```python
class ExerciseRepository(Protocol):
    def iter_exercises(self) -> Iterable[Exercise]:
        ...

class EmbeddedExerciseRepository(Protocol):
    def iter_embedded_exercises(self) -> Iterable[EmbeddedExercise]:
        ...
```

Do not revert these to list-returning methods.

### 7.3 v1 retrieval is application-layer exact search

In v1, `recommend_exercises(...)` builds the query, embeds the query, streams embedded records, filters by equipment, and calls exact top-K retrieval.

This is accurate for v1. v2 should move search mechanics behind a search port.

### 7.4 API and CLI layers stay thin

API and CLI layers wire adapters to application use cases.

Do not move recommendation logic into:

```text
api/
cli/
scripts/
future UI layers
```

### 7.5 Tests must stay local and deterministic

Automated tests must not:

- download Qwen
- call external embedding APIs
- require external services
- depend on real generated cache artifacts under `data/`

Use fake, local deterministic, or injected embedding providers in tests.

---

## 8. Layers And Responsibilities

### 8.1 Domain

Current key files:

```text
domain/exercise.py
domain/user_request.py
domain/recommendation.py
domain/embedded_exercise.py
domain/types.py
```

Responsibilities:

- define and validate core data objects
- avoid infrastructure details

### 8.2 Application

Current key files:

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
- filter by equipment in v1
- compute cosine similarity in v1
- retrieve/rank candidates in v1
- build deterministic explanations
- orchestrate recommendation runtime
- build embedded exercises from raw exercises
- orchestrate streaming cache build workflow

v2 should move search mechanics behind a port while keeping recommendation policy and explanation building in application logic.

### 8.3 Ports

Current key files:

```text
ports/exercise_repository.py
ports/embedded_exercise_repository.py
ports/embedding_provider.py
```

There is no search port in v1.

v2 should add a search port such as `EmbeddedExerciseSearchRepository`.

### 8.4 Adapters

Current key files:

```text
adapters/csv_exercise_repository.py
adapters/sqlite_exercise_repository.py
adapters/local_embedded_exercise_cache.py
adapters/local_deterministic_embedding_provider.py
adapters/fake_embedding_provider.py
adapters/sentence_transformer_embedding_provider.py
```

Responsibilities:

- stream raw exercises from CSV or SQLite
- write/read local embedded exercise JSONL cache
- provide fake deterministic embeddings for tests
- provide local deterministic embeddings for development
- provide Sentence Transformers embeddings for real local retrieval

There is no SQLite embedded cache adapter in v1.

### 8.5 API

Current key files:

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
uv run uvicorn calisthenics_recommender.api.main:app --reload --host 127.0.0.1 --port 8000
```

Runtime environment variables:

```text
CALISTHENICS_RECOMMENDER_CACHE_PATH
CALISTHENICS_RECOMMENDER_EMBEDDING_PROVIDER
CALISTHENICS_RECOMMENDER_EMBEDDING_MODEL
CALISTHENICS_RECOMMENDER_QUERY_PREFIX
```

Supported v1 runtime embedding providers:

```text
local-deterministic
sentence-transformer
```

### 8.6 CLI

Packaged CLI commands:

```text
import-exercises-to-sqlite
build-exercise-cache
demo-recommend
debug-recommendations
```

CLI code should wire adapters and application functions together. It should not contain core recommendation logic.

---

## 9. Data And Cache Convention

Raw CSV input:

```text
data/raw/
```

Raw SQLite databases:

```text
data/db/
```

Embedded JSONL caches:

```text
data/cache/
```

These are local artifacts. Generated data should not be committed.

---

## 10. Completed v1 Milestones

Completed milestones:

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

v1 is closed after Milestone 17.

---

## 11. v2 Direction

The v2 plan is:

- SQLite embedded cache.
- Search port / search repository.
- JSONL search adapter for backward compatibility.
- SQLite search adapter.
- CLI/API wiring to select the new embedded cache/search backend.
- Docker runtime service after the search/cache refactor.

Use `V2_REFACTOR_PLAN.md` as the engineering plan.

---

## 12. Technology Stack

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
standard library csv/json/pathlib/logging/typing/heapq/sqlite3
```

Do not add dependencies unless explicitly approved.

---

## 13. Workflow Rules

When implementing future work:

1. Inspect the current code before editing.
2. Add focused tests for behavior changes.
3. Keep source changes scoped to the requested task.
4. Run relevant tests.
5. Run full `uv run pytest` for meaningful code changes.
6. Run `uv run ruff check .` when Python code changes.
7. Do not commit unless explicitly told.
8. Report files changed, verification commands, and tradeoffs.

For this docs-only v1 close-out, do not change source code, tests, or dependencies.

---

## 14. Anti-Patterns To Avoid

Avoid:

```text
embedding all exercises per user request
returning list[...] from repository ports
direct CSV/cache/SQLite access inside API or CLI recommendation flows
recommendation logic inside scripts, CLI, API, or UI
real model downloads in automated tests
hardcoded absolute paths
hardcoded API keys
mixing SQLite embedded cache, search port, Docker, frontend, and cloud in one milestone
implying SQLite embedded cache exists before v2 implements it
implying a search port exists before v2 implements it
```
