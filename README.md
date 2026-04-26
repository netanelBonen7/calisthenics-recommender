# Calisthenics Recommender

A local, embedding-based calisthenics exercise recommender built with clean architecture.

The system recommends exercises from a calisthenics dataset using structured user input, semantic embeddings, deterministic equipment filtering, and a precomputed local embedding cache.

At runtime, the recommender embeds only the user request, filters candidate exercises by available equipment, and ranks pre-embedded exercise records from a local JSONL cache.

This project is designed as an interview-presentable backend / ML-adjacent software project. It is not claiming production-ready recommendation quality yet.

---

## What The Project Does

The recommender receives a request such as:

```json
{
  "target_family": "Pull-up",
  "goal": "I want to build pulling strength and improve my strict pull-ups.",
  "current_level": "I can do a few strict pull-ups but my last reps are slow.",
  "available_equipment": ["Bar"]
}
```

It returns ranked exercise recommendations with:

```json
{
  "exercise_name": "Pull Up",
  "match_score": 62,
  "reason": "Recommended because it matched your Pull-up target family through retrieval, belongs to the Pull-up families, falls under the Upper Body Pull categories, and requires Bar.",
  "required_equipment": ["Bar"],
  "category_family": {
    "categories": ["Upper Body Pull"],
    "families": ["Pull-up"]
  }
}
```

Recommendation explanations are deterministic and grounded in dataset fields. The MVP does not use an LLM to generate explanations.

---

## Why This Project Exists

This project demonstrates how to build a small but realistic recommendation system with strong software engineering practices.

It shows practical engineering choices around:

- clean / hexagonal architecture
- domain, application, ports, and adapters separation
- streaming repositories
- offline embedding cache generation
- runtime retrieval from precomputed embeddings
- deterministic filters
- local fake embeddings for tests and development
- local open-source embedding models through Sentence Transformers
- real dataset parsing
- CLI and debugging tools
- TDD-driven development

The goal is to build a maintainable local recommendation pipeline that can later grow into an API, frontend UI, Dockerized service, or deployed application without rewriting the core logic.

---

## Main Features

- CSV exercise dataset parsing with validation
- Support for both semicolon-style list fields and real dataset JSON-list fields
- Exercise text building from structured exercise fields
- Query text building from structured and free-text user request fields
- Fake deterministic embeddings for tests and local development
- Local Sentence Transformers embeddings, using Qwen by default
- Offline embedded exercise cache building
- Runtime recommendation from a precomputed JSONL cache
- Deterministic equipment filtering before retrieval
- Exact top-K retrieval over cached embeddings
- Deterministic recommendation explanations
- Debug tooling for inspecting query text, exercise text, and top retrieval candidates

---

## Architecture Overview

The project follows a clean / hexagonal architecture style.

```text
src/calisthenics_recommender/
├── domain/
├── application/
├── ports/
├── adapters/
└── cli/

scripts/
├── build_exercise_cache.py
├── demo_recommend.py
└── debug_recommendations.py
```

### Domain Layer

Pure data models and value objects.

Examples:

```text
domain/exercise.py
domain/user_request.py
domain/embedded_exercise.py
domain/recommendation.py
```

The domain layer does not know about CSV files, JSONL caches, embedding libraries, scripts, APIs, or UI.

### Application Layer

Pure use-case and business logic.

Examples:

```text
application/query_builder.py
application/exercise_text_builder.py
application/filters.py
application/similarity.py
application/retriever.py
application/explanation_builder.py
application/recommend_exercises.py
application/embedded_exercise_cache_workflow.py
```

This layer builds query text, builds exercise text, filters candidates, computes similarity, retrieves top matches, builds deterministic explanations, and orchestrates cache building.

### Ports Layer

Small protocol interfaces.

Examples:

```text
ports/exercise_repository.py
ports/embedded_exercise_repository.py
ports/embedding_provider.py
```

The core logic depends on interfaces, not concrete infrastructure.

### Adapters Layer

Concrete infrastructure implementations.

Examples:

```text
adapters/csv_exercise_repository.py
adapters/local_embedded_exercise_cache.py
adapters/local_deterministic_embedding_provider.py
adapters/sentence_transformer_embedding_provider.py
```

Adapters connect the core to CSV files, local JSONL caches, fake deterministic embeddings, and Sentence Transformers models.

### CLI Layer

Packaged developer-facing entry points.

```text
cli/build_exercise_cache.py
cli/demo_recommend.py
cli/debug_recommendations.py
```

These CLI modules wire adapters and application functions together. They should not contain recommendation logic.

The top-level `scripts/` files are thin compatibility wrappers around the packaged CLI entry points.

---

## Pipeline Overview

### Cache Build Flow

Exercise embeddings are built offline and stored as a derived cache.

```text
CSV
→ CsvExerciseRepository
→ build_exercise_text(...)
→ EmbeddingProvider.embed(...)
→ EmbeddedExercise
→ LocalEmbeddedExerciseCache
→ JSONL cache
```

This avoids embedding all exercises on every request.

### Runtime Recommendation Flow

At runtime, only the user query is embedded.

```text
UserRequest
→ build_query_text(...)
→ EmbeddingProvider.embed(query)
→ LocalEmbeddedExerciseRepository
→ equipment filtering
→ exact top-K retrieval
→ deterministic recommendations
```

The runtime recommender does not re-embed all exercises.

---

## Data And Cache Convention

The real CSV dataset should live locally at:

```text
data/raw/calisthenics_exercises.csv
```

Generated embedding caches should live under:

```text
data/cache/
```

Example:

```text
data/cache/calisthenics_qwen_cache.jsonl
```

The original CSV is treated as read-only input. The system reads from it but does not modify it.

The real dataset file and generated cache files are intentionally gitignored local artifacts.

The JSONL cache is derived data. It includes metadata such as:

- `embedding_model`
- `embedding_dimension`
- `text_builder_version`

This helps avoid silently using vectors from an incompatible model or text format.

---

## Setup

The project currently supports Python:

```text
>=3.11,<3.13
```

The project uses `uv` for dependency management.

Install dependencies:

```powershell
uv sync
```

`uv sync` installs the project itself, which enables the packaged CLI commands used below.

Run the test suite:

```powershell
uv run pytest
```

Sentence Transformers / Qwen usage may download model files on first run. The default local Sentence Transformers model is:

```text
Qwen/Qwen3-Embedding-0.6B
```

No OpenAI API key is required.

---

## Usage Examples

### Build A Cache With Local Deterministic Embeddings

This mode is fast and useful for development and testing.

```powershell
uv run build-exercise-cache `
  --input-csv .\data\raw\calisthenics_exercises.csv `
  --output-cache .\data\cache\calisthenics_fake_cache.jsonl `
  --embedding-provider local-deterministic `
  --embedding-model fake-hash-v1 `
  --embedding-dimension 4 `
  --text-builder-version v1
```

### Build A Cache With Qwen / Sentence Transformers

This mode uses a real local open-source embedding model. The first run may download model files from Hugging Face.

```powershell
uv run build-exercise-cache `
  --input-csv .\data\raw\calisthenics_exercises.csv `
  --output-cache .\data\cache\calisthenics_qwen_cache.jsonl `
  --embedding-provider sentence-transformer `
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" `
  --text-builder-version v1
```

### Run Demo Recommendations From A Cache

Example Pull-up query using a Qwen-backed cache:

```powershell
uv run demo-recommend `
  --cache-path .\data\cache\calisthenics_qwen_cache.jsonl `
  --embedding-provider sentence-transformer `
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" `
  --target-family "Pull-up" `
  --goal "I want to build pulling strength and improve my strict pull-ups." `
  --current-level "I can do a few strict pull-ups but my last reps are slow." `
  --available-equipment "Bar" `
  --limit 5
```

Multiple equipment options can be passed by repeating the flag:

```powershell
--available-equipment "Bar" `
--available-equipment "Rings"
```

Use the same embedding provider family at runtime that was used to build the cache.

---

## Debug Recommendation Behavior

The debug script helps inspect what the system is actually embedding and retrieving.

### Inspect Query Text And Selected Exercise Texts

```powershell
uv run debug-recommendations `
  --input-csv .\data\raw\calisthenics_exercises.csv `
  --exercise-name "Str OA Row" `
  --exercise-name "Pull Up" `
  --exercise-name "Row" `
  --exercise-name "C2B Pull Up" `
  --exercise-name "Strict Bar MU" `
  --target-family "Pull-up" `
  --goal "I want to build pulling strength and improve my strict pull-ups." `
  --current-level "I can do a few strict pull-ups but my last reps are slow." `
  --available-equipment "Bar"
```

### Inspect Top Retrieval Candidates From A Cache

```powershell
uv run debug-recommendations `
  --cache-path .\data\cache\calisthenics_qwen_cache.jsonl `
  --embedding-provider sentence-transformer `
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" `
  --target-family "Pull-up" `
  --goal "I want to build pulling strength and improve my strict pull-ups." `
  --current-level "I can do a few strict pull-ups but my last reps are slow." `
  --available-equipment "Bar" `
  --limit 10
```

This is useful for understanding why certain exercises rank above others.

---

## Testing

Run all tests:

```powershell
uv run pytest
```

The test suite is designed around TDD and avoids external model/API calls.

Tests use fake or injected embedding providers so they do not download Qwen or call external embedding APIs.

Testing principles:

- fake deterministic embeddings in unit and integration tests
- injected fake models for Sentence Transformers adapter tests
- no Qwen downloads during automated tests
- no OpenAI calls
- streaming behavior is tested
- cache validation is tested
- CSV parsing is tested against both toy and real-format list fields
- scripts are tested through `main(argv)` functions
- integration tests verify local cache/recommendation wiring

Current test areas include:

```text
domain models
ports
CSV repository
local JSONL cache
fake embedding provider
Sentence Transformers provider
query/exercise text builders
equipment filters
cosine similarity
top-K retrieval
recommendation building
cache build workflow
terminal scripts
debug tooling
integration pipeline
```

Ruff is run automatically by the Git pre-commit hook. It can also be run manually with:

```powershell
uv run ruff check .
```

---

## Current Limitations

This project is not production-ready yet.

Known limitations:

- Recommendation quality has not been fully tuned.
- `target_family` currently influences semantic retrieval and explanations, but it is not yet a deterministic hard filter or boost.
- Difficulty/progression filtering is not implemented yet.
- `current_level` is embedded semantically but not interpreted as structured progression logic.
- No frontend UI yet.
- No HTTP API yet.
- No Docker image yet.
- No vector database yet.
- Local Qwen / Sentence Transformers setup can be heavy because it depends on transformer model files.

These limitations are intentional and tracked as future work.

---

## Roadmap

Likely next milestones:

```text
14 - documentation and project polish
15 - script execution and developer-experience cleanup
16 - FastAPI backend adapter
17 - local backend demo with real cache
18 - frontend UI
19 - Docker runtime service
20 - optional cloud deployment
21 - recommendation quality tuning
```

Possible recommendation-quality improvements later:

- inspect and tune query/exercise text builders
- experiment with query/document prefixes
- add target-family filtering or boosting
- add difficulty/progression metadata
- add quality/sanity evaluation scenarios
- compare embedding models

---

## Interview Explanation

A concise way to explain the project:

> I built an embedding-based calisthenics exercise recommender using clean architecture. Raw exercises stream from a CSV repository, exercise embeddings are built offline and stored in a local JSONL cache, and runtime recommendation embeds only the user query. The recommender filters candidates deterministically by equipment and retrieves exact top-K matches from precomputed embeddings. The system supports fake deterministic embeddings for tests and local Sentence Transformers/Qwen embeddings for real semantic retrieval, while keeping domain, application logic, ports, adapters, and scripts separated.

Testing explanation:

> I used fake deterministic embeddings and injected fake model objects so the test suite can validate parsing, filtering, retrieval, cache behavior, script wiring, and provider behavior without calling external APIs or downloading real models.

Storage explanation:

> I treat embeddings as derived data, not source-of-truth data. The original CSV stays read-only, while embedded records are stored separately in a generated JSONL cache with metadata such as model name, embedding dimension, and text-builder version.
