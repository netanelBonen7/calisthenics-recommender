# Calisthenics Recommender

Status: v1 closed.

`calisthenics-recommender` is a local, embedding-based calisthenics exercise recommender built as a clean architecture backend project.

v1 is a stable local API MVP. It can parse a raw exercise dataset, import raw exercises into SQLite, build a local embedded exercise cache, serve recommendations from that cache through CLI commands, and expose the same recommender through FastAPI.

v1 is not a production recommender yet. It is a working local backend foundation with clear next steps for v2.

---

## What The Project Does

The recommender receives a structured user request:

```json
{
  "target_family": "Pull-up",
  "goal": "I want to build pulling strength and improve my strict pull-ups.",
  "current_level": "I can do a few strict pull-ups but my last reps are slow.",
  "available_equipment": ["Bar"],
  "limit": 5
}
```

It returns ranked exercise recommendations:

```json
{
  "recommendations": [
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
  ]
}
```

Recommendation explanations are deterministic and grounded in dataset fields. v1 does not use an LLM to generate explanations.

---

## v1 Capabilities

- CSV raw exercise parsing.
- SQLite raw exercise import and repository.
- Cache building from CSV or SQLite raw exercise input.
- JSONL embedded exercise cache.
- Runtime recommendation from a JSONL embedded cache.
- Runtime query embedding with local deterministic embeddings or Sentence Transformers.
- Deterministic equipment filtering.
- Application-layer exact top-K search over streamed embedded records.
- CLI commands:
  - `uv run import-exercises-to-sqlite`
  - `uv run build-exercise-cache`
  - `uv run demo-recommend`
  - `uv run debug-recommendations`
- FastAPI adapter:
  - `GET /health`
  - `POST /recommend`
- Local FastAPI runtime through `calisthenics_recommender.api.main:app`.
- Automated tests using fake, local deterministic, or injected embedding providers.

---

## Architecture Overview

The project follows a clean / hexagonal architecture style.

```text
src/calisthenics_recommender/
- domain/
- application/
- ports/
- adapters/
- api/
- cli/
```

### Domain

Pure data models and validation:

```text
domain/exercise.py
domain/user_request.py
domain/embedded_exercise.py
domain/recommendation.py
domain/types.py
```

### Application

Use-case logic:

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

In v1, equipment filtering and exact top-K retrieval happen in this layer.

### Ports

Small protocol interfaces:

```text
ports/exercise_repository.py
ports/embedded_exercise_repository.py
ports/embedding_provider.py
```

There is no search port in v1. Moving search behind a port is planned for v2.

### Adapters

Concrete infrastructure:

```text
adapters/csv_exercise_repository.py
adapters/sqlite_exercise_repository.py
adapters/local_embedded_exercise_cache.py
adapters/local_deterministic_embedding_provider.py
adapters/fake_embedding_provider.py
adapters/sentence_transformer_embedding_provider.py
```

The embedded cache adapter in v1 is JSONL-backed through `local_embedded_exercise_cache.py`. There is no SQLite embedded cache yet.

### API And CLI

The API and CLI layers wire adapters to application logic. They should stay thin and should not contain recommendation logic.

---

## Current Pipeline

### Offline Cache Build

```text
CSV or SQLite raw exercises
-> ExerciseRepository
-> build_exercise_text(...)
-> EmbeddingProvider.embed(...)
-> EmbeddedExercise
-> LocalEmbeddedExerciseCache
-> JSONL embedded cache
```

### Runtime Recommendation

```text
UserRequest
-> build_query_text(...)
-> EmbeddingProvider.embed(query)
-> LocalEmbeddedExerciseRepository
-> stream embedded exercises from JSONL
-> application-layer equipment filtering
-> application-layer exact top-K retrieval
-> deterministic recommendation response
```

Runtime embeds only the user query. It does not re-embed every exercise per request.

---

## Data And Cache Convention

Raw CSV input should live under:

```text
data/raw/
```

Example:

```text
data/raw/calisthenics_exercises.csv
```

Imported raw SQLite databases should live under:

```text
data/db/
```

Example:

```text
data/db/calisthenics_exercises.sqlite
```

Embedded JSONL caches should live under:

```text
data/cache/
```

Example:

```text
data/cache/calisthenics_qwen_cache.jsonl
```

The raw dataset, generated SQLite databases, and generated embedded caches are local artifacts and are intentionally not part of the repository.

---

## Setup

The project supports Python:

```text
>=3.11,<3.13
```

Install dependencies:

```powershell
uv sync
```

Run tests:

```powershell
uv run pytest
```

Run Ruff:

```powershell
uv run ruff check .
```

The default real local embedding model is:

```text
Qwen/Qwen3-Embedding-0.6B
```

First use may download model files through Sentence Transformers. No OpenAI API key is required.

---

## Usage

### Import CSV Exercises To SQLite

This imports raw exercises into a local SQLite database. It does not store embeddings in SQLite.

```powershell
uv run import-exercises-to-sqlite `
  --input-csv .\data\raw\calisthenics_exercises.csv `
  --output-db .\data\db\calisthenics_exercises.sqlite
```

### Build A Development JSONL Cache

This uses local deterministic embeddings and is useful for development.

```powershell
uv run build-exercise-cache `
  --input-csv .\data\raw\calisthenics_exercises.csv `
  --output-cache .\data\cache\calisthenics_fake_cache.jsonl `
  --embedding-provider local-deterministic `
  --embedding-model fake-hash-v1 `
  --embedding-dimension 4 `
  --text-builder-version v1
```

### Build A Real Qwen JSONL Cache

This uses Sentence Transformers and the Qwen embedding model. The example reads raw exercises from SQLite and writes embedded records to JSONL.

```powershell
uv run build-exercise-cache `
  --input-db .\data\db\calisthenics_exercises.sqlite `
  --output-cache .\data\cache\calisthenics_qwen_cache.jsonl `
  --embedding-provider sentence-transformer `
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" `
  --text-builder-version v1
```

### Run FastAPI Locally

Use the same embedding provider family at runtime that was used to build the cache.

```powershell
$env:CALISTHENICS_RECOMMENDER_CACHE_PATH = ".\data\cache\calisthenics_qwen_cache.jsonl"
$env:CALISTHENICS_RECOMMENDER_EMBEDDING_PROVIDER = "sentence-transformer"
$env:CALISTHENICS_RECOMMENDER_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
$env:CALISTHENICS_RECOMMENDER_QUERY_PREFIX = ""

uv run uvicorn calisthenics_recommender.api.main:app --reload --host 127.0.0.1 --port 8000
```

### Call `/health`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### Call `/recommend`

```powershell
$body = @{
  target_family = "Pull-up"
  goal = "I want to build pulling strength and improve my strict pull-ups."
  current_level = "I can do 5 strict pull-ups, but the last reps are slow."
  available_equipment = @("Bar")
  limit = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/recommend `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### Run CLI Recommendation Demo

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

Repeat `--available-equipment` to pass multiple equipment options.

### Run Debug Tooling

Inspect query text and selected exercise texts:

```powershell
uv run debug-recommendations `
  --input-csv .\data\raw\calisthenics_exercises.csv `
  --exercise-name "Pull Up" `
  --exercise-name "Row" `
  --target-family "Pull-up" `
  --goal "I want to build pulling strength and improve my strict pull-ups." `
  --current-level "I can do a few strict pull-ups but my last reps are slow." `
  --available-equipment "Bar"
```

Inspect top retrieval candidates from a JSONL embedded cache:

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

---

## Current Limitations

- No SQLite embedded cache yet.
- No search port yet.
- No vector database, vector extension, or approximate nearest neighbor index yet.
- No Docker image yet.
- No frontend yet.
- No cloud deployment yet.
- Recommendation quality has not been fully tuned.
- `target_family` influences semantic retrieval and explanations, but it is not a deterministic hard filter or boost.
- Difficulty/progression filtering is not implemented.
- `current_level` is embedded semantically but is not interpreted as structured progression logic.

---

## v2 Direction

v2 will refactor embedded cache and search responsibility:

- add SQLite embedded cache storage
- add an embedded search port
- move JSONL exact scan/filter/similarity/top-K behavior behind a JSONL search adapter
- add a SQLite exact search adapter
- wire CLI and API runtime config to the new backend options
- Dockerize the FastAPI runtime service after the search/cache refactor

The detailed engineering plan is in `V2_REFACTOR_PLAN.md`.
