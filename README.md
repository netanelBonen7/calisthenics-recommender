# Calisthenics Recommender

`calisthenics-recommender` is a local, embedding-based calisthenics exercise recommender built as a clean architecture backend project.

The stable v1 baseline is tagged as `v1-local-api-mvp`.

The current backend v2 branch keeps the same API request/response shape while adding SQLite embedded cache support, an embedded search port, JSONL and SQLite exact search adapters, TOML-driven runtime configuration, operator CLI `--config` support, and Docker packaging for the FastAPI runtime.

Cloud deployment, vector database integration, and frontend work are not implemented on this main backend branch.

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

Recommendation explanations are deterministic and grounded in dataset fields. The project does not use an LLM to generate explanations.

---

## v1 Baseline

`v1-local-api-mvp` is the stable v1 tag.

v1 includes:

- CSV raw exercise parsing.
- SQLite raw exercise import and repository.
- JSONL embedded exercise cache.
- Runtime query embedding with local deterministic embeddings or Sentence Transformers.
- Deterministic equipment filtering.
- Application-layer exact top-K search.
- CLI commands:
  - `uv run import-exercises-to-sqlite`
  - `uv run build-exercise-cache`
  - `uv run demo-recommend`
  - `uv run debug-recommendations`
- FastAPI local runtime:
  - `GET /health`
  - `POST /recommend`
- Automated tests using fake, local deterministic, or injected embedding providers.

---

## Current Backend v2 State

The current backend v2 line adds:

- SQLite embedded cache writer/reader.
- `EmbeddedExerciseSearchRepository` search port.
- `EmbeddedExerciseSearchResult`.
- JSONL exact search adapter.
- SQLite exact search adapter.
- `recommend_exercises(...)` uses the search port.
- API runtime config through TOML loaded from `CALISTHENICS_RECOMMENDER_CONFIG_PATH`.
- Optional `--config` support for:
  - `build-exercise-cache`
  - `demo-recommend`
  - `debug-recommendations`
- Config selection for:
  - raw exercise source: CSV or SQLite
  - embedded cache backend: JSONL or SQLite
  - embedding provider/model/dimension/prefixes
  - text builder version for cache building
- Docker packaging for the FastAPI runtime service.

Existing explicit CLI workflows are preserved where practical. The API request/response shape is unchanged.

Not included on this branch:

- No vector DB, sqlite-vec, pgvector, or FAISS.
- No cloud deployment yet.
- No frontend merged into this backend branch.

During v2 work, an optional React/Vite demo UI prototype was explored on branch `v2-6c-demo-ui-prototype`. It is intentionally separate and is not part of the main backend v2 line.

---

## Architecture Overview

The project follows a clean / hexagonal architecture style.

For a deeper design explanation, see docs/ARCHITECTURE.md.

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
domain/embedded_exercise_search_result.py
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

The application layer owns recommendation policy, query/text construction, deterministic explanations, and use-case orchestration. In the current v2 branch, backend-specific search mechanics live behind the embedded search port.

### Ports

Small protocol interfaces:

```text
ports/exercise_repository.py
ports/embedded_exercise_repository.py
ports/embedded_exercise_search_repository.py
ports/embedding_provider.py
```

### Adapters

Concrete infrastructure:

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

### API And CLI

The API and CLI layers wire config, adapters, and application use cases together. They should stay thin and should not contain recommendation logic.

---

## Current Pipeline

### Offline Cache Build

```text
CSV or SQLite raw exercises
-> ExerciseRepository
-> build_exercise_text(...)
-> EmbeddingProvider.embed(...)
-> EmbeddedExercise
-> JSONL or SQLite embedded cache writer
```

### Runtime Recommendation

```text
UserRequest
-> build_query_text(...)
-> EmbeddingProvider.embed(query)
-> EmbeddedExerciseSearchRepository
-> JSONL or SQLite exact search adapter
-> deterministic recommendation response
```

Runtime embeds only the user query. It does not re-embed every exercise per request.

---

## Data And Cache Convention

Raw CSV input should live under:

```text
data/raw/
```

Imported raw SQLite databases should live under:

```text
data/db/
```

Embedded JSONL or SQLite caches should live under:

```text
data/cache/
```

Examples:

```text
data/raw/calisthenics_exercises.csv
data/db/calisthenics_exercises.sqlite
data/cache/calisthenics_qwen_cache.jsonl
data/cache/calisthenics_qwen_cache.sqlite
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

This imports raw exercises into a local SQLite database. It does not store embeddings in the raw exercise database.

```powershell
uv run import-exercises-to-sqlite `
  --input-csv .\data\raw\calisthenics_exercises.csv `
  --output-db .\data\db\calisthenics_exercises.sqlite
```

### Build A JSONL Cache With Explicit Flags

```powershell
uv run build-exercise-cache `
  --input-csv .\data\raw\calisthenics_exercises.csv `
  --output-cache .\data\cache\calisthenics_fake_cache.jsonl `
  --embedding-provider local-deterministic `
  --embedding-model fake-hash-v1 `
  --embedding-dimension 4 `
  --text-builder-version v1
```

### Build A Real Qwen JSONL Cache With Explicit Flags

This example reads raw exercises from SQLite and writes embedded records to JSONL.

```powershell
uv run build-exercise-cache `
  --input-db .\data\db\calisthenics_exercises.sqlite `
  --output-cache .\data\cache\calisthenics_qwen_cache.jsonl `
  --embedding-provider sentence-transformer `
  --embedding-model "Qwen/Qwen3-Embedding-0.6B" `
  --text-builder-version v1
```

### Shared TOML Config

`build-exercise-cache`, `demo-recommend`, `debug-recommendations`, and the FastAPI runtime can use TOML config for backend and embedding settings.

```toml
[raw_exercises]
backend = "sqlite"
sqlite_path = "data/db/calisthenics_exercises.sqlite"

[embedded_cache]
backend = "sqlite"
path = "data/cache/calisthenics_qwen_cache.sqlite"

[embedding]
provider = "sentence-transformer"
model = "Qwen/Qwen3-Embedding-0.6B"
query_prefix = ""
text_prefix = ""
text_builder_version = "v1"
```

For CSV raw input, use `backend = "csv"` with `csv_path`. For JSONL embedded cache, use `backend = "jsonl"` with a `.jsonl` cache path.

Config paths are resolved relative to the TOML file. Explicit CLI flags still win over matching config values where those workflows are supported.

### Build A Cache With Config

```powershell
uv run build-exercise-cache --config .\runtime.toml
```

### Run CLI Recommendation Demo With Config

```powershell
uv run demo-recommend `
  --config .\runtime.toml `
  --target-family "Pull-up" `
  --goal "I want to build pulling strength and improve my strict pull-ups." `
  --current-level "I can do a few strict pull-ups but my last reps are slow." `
  --available-equipment "Bar" `
  --limit 5
```

Repeat `--available-equipment` to pass multiple equipment options.

### Run Debug Tooling With Config

Inspect query text and selected exercise texts:

```powershell
uv run debug-recommendations `
  --config .\runtime.toml `
  --exercise-name "Pull Up" `
  --exercise-name "Row"
```

Inspect top retrieval candidates:

```powershell
uv run debug-recommendations `
  --config .\runtime.toml `
  --target-family "Pull-up" `
  --goal "I want to build pulling strength and improve my strict pull-ups." `
  --current-level "I can do a few strict pull-ups but my last reps are slow." `
  --available-equipment "Bar" `
  --limit 10
```

### Run FastAPI Locally

Use a runtime TOML config file and point the API to it with `CALISTHENICS_RECOMMENDER_CONFIG_PATH`.

```powershell
$env:CALISTHENICS_RECOMMENDER_CONFIG_PATH = ".\runtime.toml"

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

### Run FastAPI With Docker

The Docker image runs only the FastAPI runtime. Build embedded caches offline with the CLI, then mount the runtime config and cache into the container. The API container does not rebuild exercise embeddings on startup.

The default Docker smoke test uses local deterministic embeddings and a SQLite embedded cache so it does not depend on Qwen, Hugging Face, internet access, or model cache availability.

Prepare the local runtime config and deterministic SQLite smoke cache on the host:

```powershell
.\scripts\prepare-docker-smoke.ps1
```

If your PowerShell execution policy blocks local scripts, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare-docker-smoke.ps1
```

The script copies `config/docker-runtime.toml.example` to `config/docker-runtime.toml` if needed, then runs `uv run build-exercise-cache --config .\config\docker-runtime.toml`. The generated `config/docker-runtime.toml` and `data/cache/docker_smoke_embeddings.sqlite` are local artifacts and are ignored by Git.

```toml
[raw_exercises]
backend = "csv"
csv_path = "../data/raw/calisthenics_exercises.csv"

[embedded_cache]
backend = "sqlite"
path = "../data/cache/docker_smoke_embeddings.sqlite"

[embedding]
provider = "local-deterministic"
model = "fake-hash-v1"
dimension = 4
query_prefix = ""
text_builder_version = "v1"
```

Start the Dockerized FastAPI runtime:

```powershell
docker compose up --build
```

The Compose service builds from the root `Dockerfile`, maps port `8000:8000`, mounts `./config` to `/app/config:ro`, mounts `./data/cache` to `/app/data/cache:ro`, and sets `CALISTHENICS_RECOMMENDER_CONFIG_PATH=/app/config/docker-runtime.toml`.

You can also run without Compose:

```powershell
docker build -t calisthenics-recommender-api .

docker run --rm -p 8000:8000 `
  -v ${PWD}\config:/app/config:ro `
  -v ${PWD}\data\cache:/app/data/cache:ro `
  -e CALISTHENICS_RECOMMENDER_CONFIG_PATH=/app/config/docker-runtime.toml `
  calisthenics-recommender-api
```

From another terminal, smoke test the running API:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

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

The same image can run with `sentence-transformer` / Qwen if the runtime config selects it and the model is available or downloadable inside the container environment. The default Docker smoke test intentionally avoids that dependency.

---

## Current Limitations

- No vector database, vector extension, or approximate nearest neighbor index yet.
- No frontend merged into the main backend branch.
- No cloud deployment yet.
- No auth, users, or persisted recommendation history.
- Recommendation quality has not been fully tuned.
- `target_family` influences semantic retrieval and explanations, but it is not a deterministic hard filter or boost.
- Difficulty/progression filtering is not implemented.
- `current_level` is embedded semantically but is not interpreted as structured progression logic.
