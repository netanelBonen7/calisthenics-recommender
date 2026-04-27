# V2 Refactor Plan

## Purpose

Refactor v1 so embedded cache storage can move from JSONL to SQLite and vector-search mechanics move behind a search port.

v1 is a working local API MVP. v2 should preserve the current user-visible behavior while changing where embedded records are stored and where search mechanics live.

---

## Current v1

- JSONL embedded cache.
- Application-layer exact top-K over streamed embedded records.
- Equipment filtering happens in the application recommendation flow.
- FastAPI runtime works locally.
- Raw exercises can come from CSV or SQLite.
- Runtime embeds only the user query.
- Exercise embeddings are built offline.

---

## Target v2

- SQLite embedded cache.
- Embedded search port.
- Recommender asks for top-K candidates instead of scanning/ranking embedded records directly.
- JSONL exact search adapter preserves backward compatibility.
- SQLite search adapter performs exact search internally first.
- Future adapters can use sqlite-vec, pgvector, FAISS, or a vector database.
- Dockerize the FastAPI runtime service after the search/cache refactor.

---

## Why This Refactor Is Useful

v1 proves the local recommendation flow works, but the recommender currently owns too much of the search process:

```text
recommend_exercises
-> stream embedded records
-> apply equipment filter
-> compute similarity
-> keep exact top-K
```

That is fine for a JSONL MVP, but it makes future storage and search backends harder to add.

v2 should move search responsibility behind a search adapter/search repository:

```text
recommend_exercises
-> build query text
-> embed query
-> ask search port for top-K candidates
-> build recommendations and explanations
```

This keeps recommendation policy in the application layer while allowing different backends to own their own search mechanics.

---

## V2.1 - SQLite Embedded Cache Storage

Goal:

Add SQLite-backed embedded cache writer/reader equivalent to the current JSONL embedded cache.

Scope:

- Add SQLite schema for embedded exercises and embedding metadata.
- Store embedded exercise records and vectors in SQLite.
- Store vectors as JSON or another simple representation for now.
- Keep raw exercises and embedded records logically separate.
- Prove JSONL and SQLite embedded caches round-trip equivalent records.
- No recommender behavior change yet.
- No vector database or SQLite vector extension yet.

---

## V2.2 - Embedded Search Port

Goal:

Introduce a search abstraction for top-K embedded candidate retrieval.

Scope:

- Add a port such as `EmbeddedExerciseSearchRepository`.
- Add a result object containing exercise plus match score / similarity score.
- Preserve existing behavior.
- Do not add sqlite-vec, pgvector, FAISS, or a vector DB yet.

---

## V2.3 - JSONL Exact Search Adapter

Goal:

Keep backward compatibility by moving current JSONL scan/filter/similarity/top-K behavior behind the new search port.

Scope:

- JSONL adapter streams embedded records.
- Applies hard filters such as equipment in Python.
- Computes exact similarity in Python.
- Keeps top-K internally.
- Recommender no longer owns those mechanics directly.

---

## V2.4 - Recommender Uses Search Port

Goal:

Refactor `recommend_exercises` so it embeds the query and asks the search port for top-K candidates.

Scope:

- Query building and query embedding remain in the recommendation flow.
- Search mechanics move behind adapters.
- Recommendation/explanation building stays in application logic.
- Preserve response shape and current behavior.

---

## V2.5 - SQLite Embedded Search Adapter

Goal:

Implement a SQLite-backed search adapter.

Scope:

- Read embedded records from SQLite.
- Apply hard filters as early as practical.
- Compute exact similarity/top-K inside the adapter for now.
- Do not add sqlite-vec yet.
- Keep future sqlite-vec/pgvector/FAISS adapters possible.

---

## V2.6 - CLI/API Wiring For SQLite Embedded Cache

Goal:

Allow build/runtime commands and API runtime config to use the SQLite embedded cache/search backend.

Scope:

- Preserve JSONL compatibility if practical.
- Add backend selection config carefully.
- Tests use fake/local deterministic embeddings only.
- Do not run Qwen in automated tests.

---

## V2.7 - Dockerize FastAPI Runtime

Goal:

Dockerize the FastAPI runtime service.

Scope:

- Dockerize the API/uvicorn runtime.
- Treat the embedding cache as an input artifact.
- Do not rebuild exercise embeddings on container startup.
- Optional future: separate cache-builder worker/container.

---

## How JSONL Backward Compatibility Works

JSONL compatibility should work by implementing the new search port with a JSONL exact-scan adapter.

The JSONL adapter can keep using the current JSONL reader:

```text
JSONL embedded cache
-> stream embedded records
-> apply hard filters
-> compute exact similarity in Python
-> keep top-K
-> return search results
```

This preserves the v1 cache format while allowing `recommend_exercises` to depend on search behavior rather than JSONL repository mechanics.

---

## Where Hard Filters Should Live

After search moves behind adapters, hard retrieval filters such as equipment should be applied as early as practical inside the search adapter.

Examples:

- JSONL exact search adapter filters records while streaming.
- SQLite exact search adapter filters rows as early as practical, then computes similarity/top-K internally.
- Future vector-backed adapters can push filters into the backend when supported.

Product-level recommendation policy and deterministic explanation building should stay in application logic.

---

## What Not To Do In v2 Yet

Do not add these until the v2 search/cache refactor is complete:

- sqlite-vec.
- pgvector.
- FAISS.
- Dedicated vector database.
- Cloud deployment.
- Frontend.
- LLM-generated explanations.
- Large recommendation evaluation framework.
- Rebuilding embeddings on API startup.
- Qwen downloads in automated tests.

Do not collapse raw exercises and embedded records into one logical concept. They may eventually share a physical database, but they should remain separate concepts.

---

## Later - Interview Prep

After v2 refactor and Docker work are complete, create a separate `INTERVIEW_PREP.md` file covering:

- demo script
- architecture explanation
- design decisions
- tradeoffs
- likely interview questions
- future cloud path
