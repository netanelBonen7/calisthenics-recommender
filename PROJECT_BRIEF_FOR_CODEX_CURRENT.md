# PROJECT_BRIEF_FOR_CODEX_CURRENT.md

## 1. Purpose of this brief

This file is the current source of truth for continuing the `calisthenics-recommender` project with Codex or a new ChatGPT chat.

The project has evolved beyond the original milestone plan. The original brief is still useful background, but this file reflects the current architecture after the refactors through Milestone 9C.

Important rule:

> Implement only the milestone explicitly requested. Do not jump ahead.

---

## 2. Project overview

This project is a **calisthenics exercise recommender**.

The system recommends exercises from a calisthenics dataset using:

- structured user fields for reliability
- free-text user fields for semantic nuance
- embeddings for semantic matching
- deterministic hard filters for practical constraints
- streaming repositories for future scalability
- a clean architecture / hexagonal architecture style

The project is intended to be interview-presentable, with a clear separation between:

- domain models
- application/use-case logic
- ports/interfaces
- infrastructure adapters
- scripts / user-facing entry points added later

---

## 3. Core MVP scope

### 3.1 What the MVP does

The MVP recommends calisthenics exercises based on:

1. the movement/exercise family the user wants to improve
2. the user's goal in natural language
3. the user's current level/progression in natural language
4. the equipment available to the user

The MVP returns a ranked list of recommendations with a score and deterministic explanation.

### 3.2 What the MVP does not do yet

Do not implement these until explicitly requested:

- full workout plans
- sets/reps prescription
- injury advice
- progression-level guarantees
- difficulty filtering
- weak-point-specific filtering
- LLM-generated explanations
- weighted profile vectors
- diversity reranking
- vector database/index
- frontend
- Docker
- Postgres
- real embedding API integration
- real dataset quality evaluation

---

## 4. MVP user input

The MVP receives 4 user input fields:

```text
target_family
goal
current_level
available_equipment
```

Example:

```json
{
  "target_family": "Pull-up",
  "goal": "I want to build pulling strength and unlock harder pull-up variations.",
  "current_level": "I can do 5 strict pull-ups, but the last reps are slow.",
  "available_equipment": ["Bar"]
}
```

Design reasoning:

- `target_family` is structured because it anchors the recommendation.
- `goal` is free text because embeddings are good for semantic intent.
- `current_level` is free text because progression descriptions are hard to fit into one rigid field.
- `available_equipment` is structured because equipment is a hard deterministic filter.

---

## 5. MVP system output

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

- `reason` must be deterministic and grounded in actual fields.
- Do not use an LLM to generate recommendation explanations in the MVP.
- Do not invent fields that are not in the dataset.

---

## 6. Current dataset fields

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

---

## 7. Current architecture: high-level pipelines

### 7.1 Raw exercise source pipeline

```text
CsvExerciseRepository
or SQLiteExerciseRepository
→ ExerciseRepository.iter_exercises()
→ Iterable[Exercise]
```

The raw exercise source is streaming. It must not require loading all raw exercises into memory.
CSV remains the original raw input format. SQLite is a local raw-exercise database
adapter populated from CSV; embeddings remain separate in the JSONL cache.

### 7.2 Embedded exercise build pipeline

```text
Iterable[Exercise]
→ build_exercise_text(exercise)
→ EmbeddingProvider.embed(exercise_text)
→ EmbeddedExercise(exercise, embedding)
```

This is implemented by:

```text
build_embedded_exercises(...)
```

This is for offline/setup/cache-building flows, not request-time recommendation.

### 7.3 Embedded cache build workflow

```text
ExerciseRepository.iter_exercises()
→ build_embedded_exercises(...)
→ cache_writer.write_embedded_exercises(...)
```

This is implemented by:

```text
build_embedded_exercise_cache(...)
```

The workflow must stay streaming. It should not materialize all raw exercises or embedded exercises into memory.
The cache-build CLI may choose CSV or SQLite as the raw exercise input source,
but the output remains the local JSONL embedded cache.

### 7.4 Local embedded cache pipeline

```text
LocalEmbeddedExerciseCache
→ writes JSONL cache with metadata

LocalEmbeddedExerciseRepository
→ streams EmbeddedExercise records from JSONL
→ EmbeddedExerciseRepository.iter_embedded_exercises()
```

The JSONL cache is a derived artifact, not the source of truth.

### 7.5 Runtime recommendation pipeline

```text
UserRequest
→ build_query_text(user_request)
→ EmbeddingProvider.embed(query_text)

EmbeddedExerciseRepository.iter_embedded_exercises()
→ lazy equipment filtering
→ retrieve_top_matches(...)
→ build_recommendations(...)
→ list[Recommendation]
```

Runtime must embed **only the user query**.

Runtime must **not** embed all exercises per request.

---

## 8. Core architecture decisions already made

### 8.1 Embeddings are precomputed

Exercise embeddings are built before runtime and stored as derived records.

Do not re-embed exercises during a recommendation request.

### 8.2 Raw exercises and embeddings are separate logical records

Raw exercise data is source-of-truth data.

Embedding records are derived/versioned data.

Conceptually:

```text
exercises
exercise_embeddings
embedding_metadata
```

Even if a future physical database stores them together, the core architecture should continue treating them as separate logical concepts.

### 8.3 Repositories are streaming-friendly

The current ports are iterator-based:

```python
class ExerciseRepository(Protocol):
    def iter_exercises(self) -> Iterable[Exercise]:
        ...

class EmbeddedExerciseRepository(Protocol):
    def iter_embedded_exercises(self) -> Iterable[EmbeddedExercise]:
        ...
```

Do not revert these to list-returning methods.

### 8.4 Retrieval uses bounded-memory exact top-K

`retrieve_top_matches(...)` consumes `Iterable[EmbeddedExercise]` and uses a heap-based exact top-K approach.

The retriever should keep memory roughly `O(limit)`, not `O(number_of_candidates)`.

### 8.5 Future batching/windowing belongs in adapters

If a future database adapter needs batch/window size, it should be implemented inside that concrete adapter, not in `recommend_exercises`.

Examples:

- CSV adapter can stream row by row.
- JSONL adapter can stream line by line.
- SQLite/Postgres adapter can fetch rows in configurable batches.
- Vector DB adapter may eventually use a different search port.

### 8.6 Cache metadata matters

The embedded exercise cache includes metadata such as:

```text
embedding_model
embedding_dimension
text_builder_version
```

The metadata helps detect stale or incompatible cached vectors.

---

## 9. Layers and responsibilities

### 9.1 Domain layer

Contains pure data/value objects.

Current key files include:

```text
domain/exercise.py
domain/user_request.py
domain/recommendation.py
domain/embedded_exercise.py
```

Responsibilities:

- define `Exercise`
- define `UserRequest`
- define `Recommendation`
- define `EmbeddedExercise`
- validate data using Pydantic where appropriate

Domain must not know about CSV, JSONL, SQLite, APIs, embedding SDKs, scripts, or UI.

### 9.2 Application layer

Contains deterministic business/use-case logic.

Current key files include:

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
- filter by equipment
- compute cosine similarity
- retrieve/rank candidates
- build deterministic explanations
- orchestrate recommendation runtime
- build embedded exercises from raw exercises
- orchestrate streaming cache build workflow

Application layer must not directly read CSV/JSONL files or call real embedding APIs.

### 9.3 Ports layer

Defines interfaces/contracts.

Current key files include:

```text
ports/exercise_repository.py
ports/embedded_exercise_repository.py
ports/embedding_provider.py
```

Responsibilities:

- define raw exercise source contract
- define embedded exercise source contract
- define embedding provider contract

Use `typing.Protocol` and keep ports small.

### 9.4 Adapters layer

Connects the core to concrete infrastructure.

Current key files include:

```text
adapters/csv_exercise_repository.py
adapters/fake_embedding_provider.py
adapters/local_embedded_exercise_cache.py
adapters/sqlite_exercise_repository.py
api/app.py
```

Responsibilities:

- stream raw exercises from local CSV or SQLite
- provide fake deterministic embeddings for tests
- write/read local embedded exercise JSONL cache

Adapters are replaceable.

### 9.5 Scripts layer

Contains runnable utilities, not core logic.

Scripts can be added later for:

```text
scripts/build_exercise_cache.py
scripts/demo_recommend.py
```

Scripts should wire adapters and application functions together. They should not contain recommendation logic.

---

## 10. Current file hierarchy

Current source package shape:

The current package now also includes an `api/` package for the FastAPI adapter.

```text
src/
└── calisthenics_recommender/
    ├── domain/
    │   ├── exercise.py
    │   ├── user_request.py
    │   ├── recommendation.py
    │   └── embedded_exercise.py
    │
    ├── application/
    │   ├── query_builder.py
    │   ├── exercise_text_builder.py
    │   ├── filters.py
    │   ├── similarity.py
    │   ├── retriever.py
    │   ├── explanation_builder.py
    │   ├── recommend_exercises.py
    │   ├── embedded_exercise_builder.py
    │   └── embedded_exercise_cache_workflow.py
    │
    ├── ports/
    │   ├── exercise_repository.py
    │   ├── embedded_exercise_repository.py
    │   └── embedding_provider.py
    │
    └── adapters/
        ├── csv_exercise_repository.py
        ├── fake_embedding_provider.py
        └── local_embedded_exercise_cache.py
```

Tests mirror the source structure:

```text
tests/domain/
tests/application/
tests/ports/
tests/adapters/
```

---

## 11. Current completed milestones

Completed milestones include:

```text
0 — Repository and project setup
1 — Domain models and validation
2 — Text builders
3 — Ports and fake embedding provider
4 — Filtering and similarity primitives
5 — Brute-force retriever
6 — Explanation builder and response construction
7 — End-to-end fake recommender
7.5 — Runtime uses precomputed EmbeddedExercise objects
8 — CSV exercise repository
8.5 — Streaming EmbeddedExerciseRepository and bounded top-K retrieval
8.6 — Streaming raw ExerciseRepository and CSV streaming
9A — Embedded exercise builder
9B — Local embedded exercise JSONL cache
9C — Streaming embedded exercise cache build workflow
10A — Local fake-cache integration test
10B — Cache build CLI
10C — Runtime recommendation CLI
11 — Local Sentence Transformers/Qwen embedding provider
12 — Real dataset CSV compatibility and smoke tests
13 — Recommendation debug/inspection tooling
14A — README overview
15A — Installable CLI entry points and script execution cleanup
15B — SQLite raw exercise database
15C — Build cache from SQLite
16 — FastAPI backend adapter
```

---

## 12. Next recommended milestone

Milestone 16 — FastAPI backend adapter is complete.

Milestone 16 is now complete, and the next recommended milestone is the local API demo with a real cache.

### Milestone 17 — Local API Demo With Real Cache

Goal:

Run the FastAPI backend locally against a real Qwen cache.

Rules:

- Provide local run commands.
- Provide sample request commands for manual testing.
- Verify the FastAPI adapter against a real local cache.
- Keep this local.
- Do not add Docker/cloud yet.

Historical reference for the completed milestone:

### Milestone 16 — FastAPI Backend Adapter

Goal:

Expose the existing recommender core through an HTTP API:

```text
POST /recommend
→ UserRequest
→ existing recommender core
→ JSON response
```

Rules:

- Add FastAPI as an adapter layer only.
- Use the existing application logic.
- Keep recommendation logic out of the endpoint.
- Use fake/deterministic embeddings in automated tests.
- Do not call real embedding APIs.
- Do not run Qwen in automated tests.
- Do not add frontend/Docker/cloud/vector database work.

---

## 13. Future milestone ideas

After 10A, likely sequence:

```text
10B — Cache build script/command
10C — Runtime demo script/CLI
11 — Real embedding provider adapter
12 — Real dataset integration and smoke tests
13 — Recommendation quality/sanity evaluation
14 — README, architecture docs, polish
```

These are not fixed. Re-evaluate before each milestone.

---

## 14. Technology stack

Current stack:

```text
Python 3.12
Pydantic v2
pytest
ruff
FastAPI
standard library csv/json/pathlib/logging/typing/heapq
standard library sqlite3
```

Do not add dependencies unless explicitly approved.

Postponed dependencies:

```text
OpenAI/Gemini SDK
python-dotenv
pandas
SQLAlchemy
FAISS / usearch / pgvector
uvicorn
Docker
React / frontend
```

---

## 15. TDD and workflow rules

The workflow is milestone-based.

When implementing a milestone:

1. Create/use a feature branch.
2. Add focused tests first.
3. Run tests and confirm red phase.
4. Implement minimum production code.
5. Run full pytest.
6. Ruff is enforced by pre-commit hook, but Codex can run `uv run ruff check .` for verification.
7. Do not commit unless explicitly told.
8. Report:
   - files changed
   - red phase result
   - full pytest result
   - Ruff result
   - git status
   - tradeoffs/concerns

User prefers to review files before committing.

Usually:

```text
feature branch
→ commit branch
→ final pytest
→ squash merge into master
```

---

## 16. Logging rules

Use logging in real adapters/workflows when useful.

Logging is appropriate for:

- reading/writing CSV/cache files
- validation failures
- cache scan start/finish
- cache write start/finish
- safe operational counts

Avoid logging:

- full user goal/current_level
- full query text
- full exercise text
- full exercise descriptions
- raw embedding vectors
- API keys/secrets

Pure domain models and pure computation functions generally should not log.

---

## 17. Testing principles

Use fake embeddings in tests.

Do not call real embedding APIs in unit tests.

Important behavior to test:

- validation
- deterministic text building
- equipment filtering
- cosine similarity
- exact top-K retrieval
- streaming behavior
- no unnecessary materialization
- cache metadata validation
- invalid row/cache errors
- deterministic recommendation output
- no file/network side effects where a component should be pure

For streaming components, tests should prove:

- one-pass iterables work
- `len(...)` is not required
- first valid output can be yielded before later invalid data fails where appropriate
- data is not materialized into lists unless explicitly needed by the test

---

## 18. Important anti-patterns to avoid

Avoid:

```text
embedding all exercises per user request
returning list[...] from repository ports
direct CSV/cache access inside recommend_exercises
recommendation logic inside scripts or CLI
real embedding API calls in unit tests
hardcoded absolute paths
hardcoded API keys
mixing data loading, embedding, retrieval, explanation, and UI in one file
adding FastAPI/Docker before core local pipeline is verified
letting Codex implement multiple milestones at once
```

---

## 19. Current interview-oriented explanation

This project can currently be explained as:

> I built an embedding-based calisthenics exercise recommender using clean architecture. Raw exercises and embedded exercise records are separated. Raw exercises stream from an `ExerciseRepository`; precomputed embeddings stream from an `EmbeddedExerciseRepository`. The runtime recommender embeds only the user query, filters candidate exercises by deterministic equipment constraints, and retrieves exact top-K matches with bounded memory. Exercise embeddings are built offline through a streaming pipeline and stored in a local JSONL cache with metadata, so the storage backend and embedding provider can be replaced later without changing the core recommendation logic.

Testing explanation:

> I used fake deterministic embeddings to test the architecture without external API calls. The tests verify text building, filtering, similarity, top-K retrieval, streaming behavior, cache validation, and full pipeline orchestration.

Storage explanation:

> I treat embeddings as derived/versioned data, not source-of-truth data. The raw exercise dataset remains separate from the embedded exercise cache. The cache has metadata such as embedding model, vector dimension, and text-builder version to prevent silently using incompatible vectors.
