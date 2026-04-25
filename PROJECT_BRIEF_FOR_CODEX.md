# PROJECT_BRIEF_FOR_CODEX.md

## 1. Purpose of this brief

This file is the source of truth for Codex and for any new chat window that continues the project.

The goal is to keep the implementation structured, low-mistake, and milestone-based.

Codex should use this file to understand:

- what the app is
- what the MVP includes and excludes
- the user input structure
- the system output structure
- the dataset/database strategy
- the file hierarchy
- the system layers
- the code architecture
- the technology stack
- the TDD-critical sections
- the milestone order
- what future changes are planned and how they should fit in

Important rule:

> Do not implement the whole project at once. Implement only the milestone explicitly requested.

---

## 2. Project overview

This project is a **calisthenics exercise recommender**.

The system recommends exercises from a calisthenics exercise dataset using a hybrid approach:

- structured user fields for reliability
- free-text user fields for semantic nuance
- embeddings for semantic matching
- deterministic hard filters for practical constraints
- a modular pure Python core before adding API, database, Docker, or frontend layers

The project is inspired by an embedding-based recommendation assignment, but it is **not** a copy of that assignment.

The goal is to build a realistic, interview-presentable software project with:

- clean architecture
- testable core logic
- replaceable infrastructure
- a clear path from small MVP to expandable product

---

## 3. MVP scope

### 3.1 What the MVP does

The MVP recommends calisthenics exercises based on:

1. the movement/exercise family the user wants to improve
2. the user's goal in natural language
3. the user's current level/progression in natural language
4. the equipment available to the user

The MVP returns a ranked list of exercise recommendations with a match score and a short deterministic explanation.

### 3.2 What the MVP does not do yet

Do not implement these in the first core milestones:

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
- FastAPI
- Docker
- SQLite

These can be added later after the pure Python core works.

---

## 4. MVP user input

The MVP receives 4 user input fields:

```text
target_family
goal
current_level
available_equipment
```

| Field | Type | Purpose |
|---|---|---|
| `target_family` | structured string | The movement/exercise family the user wants to improve |
| `goal` | free text | The user’s goal in natural language |
| `current_level` | free text | The user’s current ability/progression |
| `available_equipment` | list of strings | Equipment the user has available |

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

- `target_family` should be structured because it anchors the recommendation.
- `goal` should be free text because embeddings are good at semantic intent.
- `current_level` should be free text because calisthenics progressions are hard to fit into one rigid format.
- `available_equipment` should be structured because equipment is a hard filter.

---

## 5. MVP system output

Each recommendation returns 5 fields:

```text
exercise_name
match_score
reason
required_equipment
category_family
```

| Field | Source |
|---|---|
| `exercise_name` | dataset `name` |
| `match_score` | computed from vector similarity |
| `reason` | deterministic explanation from user input + dataset fields |
| `required_equipment` | dataset `materials` |
| `category_family` | dataset `categories` and/or `families` |

Example:

```json
{
  "exercise_name": "Pull Up Negative",
  "match_score": 87,
  "reason": "Recommended because it belongs to the Pull-up family, supports your goal, and uses equipment you selected.",
  "required_equipment": ["Bar"],
  "category_family": {
    "categories": ["Upper Body Pull"],
    "families": ["Pull-up"]
  }
}
```

Design reasoning:

- `suggested_role` is not included in MVP because the current dataset does not have role metadata.
- `category_family` is included because it is directly supported by the dataset.
- `reason` should be deterministic and grounded in real fields, not hallucinated by an LLM.

---

## 6. Dataset and database strategy

### 6.1 Dataset fields

The MVP uses a calisthenics exercise dataset with these fields:

```text
name
description
muscle_groups
families
materials
categories
```

These fields should become validated internal `Exercise` objects.

### 6.2 Initial dataset storage

The first implementation should treat the dataset as a **local frozen file**:

```text
data/exercises.csv
```

or:

```text
data/exercises.json
```

Do not make the runtime app depend on downloading the dataset from the internet.

A future script may refresh/download the dataset, but the recommender core should load local data through a repository.

### 6.3 Future storage upgrade

Later, the app may move from a CSV/JSON file to SQLite.

The core recommendation logic must not change when that happens.

Use the Repository Pattern:

```text
ExerciseRepository
    ↓
CsvExerciseRepository now
SQLiteExerciseRepository later
```

Future storage may include:

```text
exercises
exercise_embeddings
embedding_metadata
recommendation_logs later
```

---

## 7. Architecture rules

Use a clean architecture / hexagonal architecture style.

The core logic must be independent of:

- CLI
- FastAPI
- React/frontend
- SQLite
- Docker
- OpenAI/Gemini SDKs
- vector databases
- Hugging Face runtime downloads

The core should depend on interfaces, not concrete external services.

Important boundaries:

```text
ExerciseRepository
EmbeddingProvider
Retriever
```

The recommendation logic should work the same whether:

- exercises come from CSV today or SQLite later
- embeddings come from fake vectors in tests or a real embedding provider later
- the interface is CLI today or FastAPI/frontend later
- retrieval uses brute-force cosine similarity today or a vector index later

---

## 8. Suggested file hierarchy

Use this structure unless there is a strong reason to adjust it:

```text
calisthenics-recommender/
│
├── pyproject.toml
├── README.md
├── .gitignore
├── PROJECT_BRIEF_FOR_CODEX.md
│
├── docs/
│   └── planning documents later
│
├── data/
│   └── exercises.csv or exercises.json
│
├── src/
│   └── calisthenics_recommender/
│       ├── __init__.py
│       │
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── exercise.py
│       │   ├── user_request.py
│       │   └── recommendation.py
│       │
│       ├── application/
│       │   ├── __init__.py
│       │   ├── recommend_exercises.py
│       │   ├── query_builder.py
│       │   ├── exercise_text_builder.py
│       │   ├── filters.py
│       │   ├── similarity.py
│       │   ├── retriever.py
│       │   └── explanation_builder.py
│       │
│       ├── ports/
│       │   ├── __init__.py
│       │   ├── exercise_repository.py
│       │   └── embedding_provider.py
│       │
│       └── adapters/
│           ├── __init__.py
│           ├── csv_exercise_repository.py
│           ├── fake_embedding_provider.py
│           └── local_embedding_cache.py
│
├── scripts/
│   ├── build_exercise_embeddings.py
│   └── demo_recommend.py
│
└── tests/
    ├── domain/
    ├── application/
    └── adapters/
```

The file hierarchy should guide separation of concerns.

Do not place all logic in one file.

---

## 9. System layers and responsibilities

### 9.1 Domain layer

Contains pure data models and core concepts.

Files:

```text
domain/exercise.py
domain/user_request.py
domain/recommendation.py
```

Responsibilities:

- define `Exercise`
- define `UserRequest`
- define `Recommendation`
- validate core data using Pydantic

This layer should not know about CSV, SQLite, APIs, embedding SDKs, or UI.

---

### 9.2 Application layer

Contains use-case logic and deterministic business logic.

Files:

```text
application/recommend_exercises.py
application/query_builder.py
application/exercise_text_builder.py
application/filters.py
application/similarity.py
application/retriever.py
application/explanation_builder.py
```

Responsibilities:

- build user query text
- build exercise embedding text
- filter exercises by equipment
- compute cosine similarity
- retrieve/rank candidates
- build deterministic explanations
- orchestrate the recommendation flow

This layer should not directly read CSV files or call real embedding APIs.

---

### 9.3 Ports layer

Defines interfaces/contracts.

Files:

```text
ports/exercise_repository.py
ports/embedding_provider.py
```

Responsibilities:

- define how the application gets exercises
- define how the application gets embeddings
- keep core logic independent from concrete storage/API implementations

Use `typing.Protocol` or an equivalent clean interface style.

---

### 9.4 Adapters layer

Connects the core to concrete external details.

Files:

```text
adapters/csv_exercise_repository.py
adapters/fake_embedding_provider.py
adapters/local_embedding_cache.py
```

Responsibilities:

- load exercises from local CSV/JSON
- provide fake embeddings in tests
- store/load cached embeddings later

Adapters are replaceable.

---

### 9.5 Scripts layer

Contains runnable utilities, not core logic.

Files:

```text
scripts/build_exercise_embeddings.py
scripts/demo_recommend.py
```

Responsibilities:

- build real exercise embeddings later
- run a CLI/demo later

Scripts should call the core. They should not contain recommendation logic.

---

## 10. Initial technology stack

Use this stack for the first pure Python core.

### 10.1 Runtime dependencies

```text
Python 3.11 or 3.12
Pydantic v2
NumPy
```

Avoid Python 3.14 for now if compatibility issues appear. Prefer Python 3.12 or 3.11 for this project.

### 10.2 Development / test dependencies

```text
pytest
ruff
```

### 10.3 Standard library tools

```text
csv
json
pathlib
logging
typing
argparse later
hashlib later
```

### 10.4 Postponed dependencies

Postpone these until later milestones:

```text
FastAPI
uvicorn
SQLite / SQLAlchemy
Docker
OpenAI / Gemini SDK
python-dotenv
pandas
Hugging Face datasets
FAISS / usearch / pgvector
React / Next.js
```

---

## 11. How postponed dependencies fit later

### FastAPI / uvicorn

Later API wrapper around the same core.

No core rewrite if `UserRequest` and `Recommendation` remain clean Pydantic models.

### SQLite / SQLAlchemy

Later storage upgrade behind `ExerciseRepository`.

No core rewrite if the recommender depends on the repository interface.

### Docker

Later deployment/reproducibility layer.

No core rewrite if paths/config are not hardcoded.

### OpenAI / Gemini SDK

Later real `EmbeddingProvider`.

No core rewrite if embedding calls stay behind the `EmbeddingProvider` interface.

### python-dotenv

Later local config/secrets support.

No core rewrite if keys/model names are passed through config/adapters.

### pandas

Possible offline dataset inspection/preparation tool.

Should not be required in runtime core.

### Hugging Face datasets

Possible dataset download/refresh script.

Runtime core should still use a local frozen dataset file.

### FAISS / usearch / pgvector

Later scalable vector retrieval.

No major core rewrite if retrieval is behind a retriever strategy.

### React / Next.js

Later frontend that calls FastAPI.

No core rewrite if request/response models are stable.

---

## 12. Important implementation principles

### 12.1 Use Pydantic for core models

Use Pydantic models for:

```text
Exercise
UserRequest
Recommendation
```

Why:

- validates external dataset rows
- validates user input
- gives clear error messages
- supports logging validation errors
- maps naturally to FastAPI later

### 12.2 Use logging with validation

Pydantic detects validation errors.

Python logging records them.

Use logging for:

- invalid dataset rows
- missing required fields
- invalid list fields
- missing embeddings
- cache mismatch later
- external API failure later

### 12.3 Use fake embeddings in unit tests

Do not call real embedding APIs in unit tests.

Use a `FakeEmbeddingProvider` with predictable vectors.

Example artificial vector space:

```text
[Pulling, Pushing, Core]

pull-up query:    [1.0, 0.0, 0.0]
pull-up exercise: [0.9, 0.1, 0.0]
push-up exercise: [0.0, 1.0, 0.0]
core exercise:    [0.0, 0.1, 0.9]
```

Fake embeddings are not meant to be realistic. They are meant to make tests deterministic.

### 12.4 Use deterministic text builders

Create deterministic text for embeddings.

User query text example:

```text
The user wants calisthenics exercises related to the Pull-up family.
The user's goal is: I want to build pulling strength and unlock harder pull-up variations.
The user's current level is: I can do 5 strict pull-ups, but the last reps are slow.
Available equipment: Bar.
Recommend suitable calisthenics exercises from the dataset.
```

Exercise text example:

```text
Exercise name: Pull Up
Description: ...
Muscle groups: Back, Biceps
Families: Pull-up
Required equipment: Bar
Categories: Upper Body Pull
```

### 12.5 Hard filters are deterministic

Equipment filtering should not rely on embeddings.

If the user does not have the required equipment, the exercise should be filtered out.

Example:

```text
User equipment: ["Bar"]
Exercise materials: ["Rings"]
Result: filtered out
```

### 12.6 MVP retrieval uses brute-force cosine similarity

The MVP should use brute-force cosine similarity over candidate exercise embeddings.

This is acceptable because the dataset is small and exact search is easier to debug.

Later, the retriever can be replaced with FAISS, usearch, pgvector, sqlite-vec, or another vector index.

### 12.7 Explanations should be deterministic

Do not use an LLM for explanations in the MVP.

The explanation should be template-based and grounded in:

- user request
- exercise family/category
- equipment match
- vector match result

---

## 13. TDD-critical sections

Use TDD especially for deterministic logic.

### Strong TDD required

```text
Domain models and validation
Exercise text builder
User query builder
Equipment filter
Cosine similarity
Brute-force retriever
Explanation/response builder
End-to-end pure Python use case with fake embeddings
```

### Moderate tests required

```text
CSV/local dataset adapter
Embedding cache
FastAPI wrapper later
```

### Mostly manual/integration testing later

```text
real embedding API behavior
CLI demo
Docker runtime
recommendation quality with real embeddings
```

---

## 14. Milestone order

Implement in this order:

```text
Milestone 0 — Repository and project setup
Milestone 1 — Domain models and validation
Milestone 2 — Text builders
Milestone 3 — Ports and fake adapters
Milestone 4 — Filtering and similarity primitives
Milestone 5 — Brute-force retriever
Milestone 6 — Explanation builder and response construction
Milestone 7 — End-to-end pure Python recommender with fake embeddings
Milestone 8 — Local dataset adapter
Milestone 9 — Real embedding integration and cache
Milestone 10 — CLI demo
Milestone 11 — FastAPI wrapper
Milestone 12 — SQLite storage upgrade
Milestone 13 — Docker and project polish
```

Only implement the milestone explicitly requested.

Do not jump ahead.

---

## 15. First implementation milestone rules

When asked to implement a milestone:

1. Use TDD where requested.
2. Create tests first.
3. Keep the scope narrow.
4. Do not implement future milestones.
5. Do not add unapproved dependencies.
6. Keep code modular and aligned with the architecture.
7. Run tests and Ruff before considering the milestone done.

---

## 16. Anti-patterns to avoid

Avoid:

```text
one giant recommender.py file
recommendation logic inside CLI/API routes
direct CSV access inside retrieval logic
real embedding API calls in unit tests
hardcoded absolute paths
hardcoded API keys
mixing filtering, similarity, explanation, and loading in one place
adding Docker/FastAPI/SQLite before the pure core works
letting Codex implement multiple milestones at once
```

---

## 17. Current workflow with user

The user wants a checkpoint-style workflow:

1. Assistant gives one step.
2. User performs it in VS Code/terminal.
3. User reports results.
4. Assistant compares results to expected behavior.
5. Only then move to the next step.

Do not rush ahead.

---

## 18. Interview-oriented architecture explanation

This project should be explainable like this:

> I built the recommender using a clean architecture approach. The domain models and recommendation use case are independent of the interface, database, and embedding provider. I used repository and adapter patterns so I could start with a local CSV dataset, later move to SQLite, and eventually replace brute-force search with a vector index without rewriting the core logic.

Testing explanation:

> I used fake embeddings in unit tests so I could test filtering, ranking, orchestration, and response building without external API calls. Real embeddings are integrated later through an adapter and cached for normal use.

Storage explanation:

> I started with a local frozen dataset file for reproducibility. The core talks to an ExerciseRepository interface, so moving from CSV to SQLite later does not require rewriting the recommendation engine.

MVP explanation:

> I kept the first MVP intentionally narrow: exercise recommendations only, not full workout programming. This made the system easier to test and gave a clean foundation for future features.
