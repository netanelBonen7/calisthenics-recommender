from importlib import import_module
import json
import sqlite3

import pytest


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_model():
    module = import_module("calisthenics_recommender.domain.embedded_exercise")
    return getattr(module, "EmbeddedExercise")


def get_embedded_exercise_search_result_model():
    module = import_module(
        "calisthenics_recommender.domain.embedded_exercise_search_result"
    )
    return getattr(module, "EmbeddedExerciseSearchResult")


def get_embedded_exercise_search_repository_protocol():
    module = import_module(
        "calisthenics_recommender.ports.embedded_exercise_search_repository"
    )
    return getattr(module, "EmbeddedExerciseSearchRepository")


def get_local_embedded_exercise_cache():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "LocalEmbeddedExerciseCache")


def get_local_embedded_exercise_repository():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "LocalEmbeddedExerciseRepository")


def get_cache_metadata_model():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "EmbeddedExerciseCacheMetadata")


def get_jsonl_embedded_exercise_search_repository():
    module = import_module(
        "calisthenics_recommender.adapters.jsonl_embedded_exercise_search_repository"
    )
    return getattr(module, "JsonlEmbeddedExerciseSearchRepository")


def get_sqlite_embedded_exercise_cache():
    module = import_module(
        "calisthenics_recommender.adapters.sqlite_embedded_exercise_cache"
    )
    return getattr(module, "SQLiteEmbeddedExerciseCache")


def get_sqlite_embedded_exercise_search_repository():
    module = import_module(
        "calisthenics_recommender.adapters.sqlite_embedded_exercise_search_repository"
    )
    return getattr(module, "SQLiteEmbeddedExerciseSearchRepository")


def exercise_id_for(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def exercise_named(
    name: str,
    *,
    materials: list[str],
    embedding_family: str = "Pull-up",
    exercise_id: str | None = None,
):
    Exercise = get_exercise_model()
    return Exercise(
        exercise_id=exercise_id_for(name) if exercise_id is None else exercise_id,
        name=name,
        description=f"{name} description.",
        muscle_groups=["Back"],
        families=[embedding_family],
        materials=materials,
        categories=["Upper Body Pull"],
    )


def embedded_exercise_named(
    name: str,
    embedding: list[float],
    *,
    materials: list[str] | None = None,
    exercise_id: str | None = None,
):
    EmbeddedExercise = get_embedded_exercise_model()
    return EmbeddedExercise(
        exercise=exercise_named(
            name,
            materials=["Bar"] if materials is None else materials,
            exercise_id=exercise_id,
        ),
        embedding=embedding,
    )


def build_metadata():
    EmbeddedExerciseCacheMetadata = get_cache_metadata_model()
    return EmbeddedExerciseCacheMetadata(
        embedding_model="test-model",
        embedding_dimension=2,
        text_builder_version="v1",
    )


def write_sqlite_cache(cache_path, embedded_exercises) -> None:
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        embedded_exercises,
        build_metadata(),
    )


def build_sqlite_repository(cache_path, *, batch_size: int = 100):
    SQLiteEmbeddedExerciseSearchRepository = get_sqlite_embedded_exercise_search_repository()
    return SQLiteEmbeddedExerciseSearchRepository(cache_path, batch_size=batch_size)


def result_names(results) -> list[str]:
    return [result.exercise.name for result in results]


def result_similarities(results) -> list[float]:
    return [result.similarity for result in results]


def create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE embedding_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            embedding_model TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
            text_builder_version TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE embedded_exercises (
            id INTEGER PRIMARY KEY,
            exercise_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            muscle_groups TEXT NOT NULL,
            families TEXT NOT NULL,
            materials TEXT NOT NULL,
            categories TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
        """
    )


def insert_metadata(connection: sqlite3.Connection, **overrides) -> None:
    row = {
        "id": 1,
        "embedding_model": "test-model",
        "embedding_dimension": 2,
        "text_builder_version": "v1",
    }
    row.update(overrides)
    connection.execute(
        """
        INSERT INTO embedding_metadata (
            id,
            embedding_model,
            embedding_dimension,
            text_builder_version
        )
        VALUES (
            :id,
            :embedding_model,
            :embedding_dimension,
            :text_builder_version
        )
        """,
        row,
    )


def insert_embedded_exercise_row(connection: sqlite3.Connection, **overrides) -> None:
    row = {
        "exercise_id": "pull-up",
        "name": "Pull Up",
        "description": "Pull Up description.",
        "muscle_groups": json.dumps(["Back"]),
        "families": json.dumps(["Pull-up"]),
        "materials": json.dumps(["Bar"]),
        "categories": json.dumps(["Upper Body Pull"]),
        "embedding": json.dumps([1.0, 0.0]),
    }
    row.update(overrides)
    if "exercise_id" not in overrides:
        row["exercise_id"] = exercise_id_for(row["name"])
    connection.execute(
        """
        INSERT INTO embedded_exercises (
            exercise_id,
            name,
            description,
            muscle_groups,
            families,
            materials,
            categories,
            embedding
        )
        VALUES (
            :exercise_id,
            :name,
            :description,
            :muscle_groups,
            :families,
            :materials,
            :categories,
            :embedding
        )
        """,
        row,
    )


def test_sqlite_embedded_exercise_search_repository_implements_search_protocol(
    tmp_path,
):
    EmbeddedExerciseSearchRepository = get_embedded_exercise_search_repository_protocol()
    cache_path = tmp_path / "embedded_exercises.sqlite"
    write_sqlite_cache(cache_path, [embedded_exercise_named("Pull Up", [1.0, 0.0])])

    repository = build_sqlite_repository(cache_path)
    results = repository.search(
        query_embedding=[1.0, 0.0],
        available_equipment=["Bar"],
        limit=1,
    )

    assert isinstance(repository, EmbeddedExerciseSearchRepository)
    assert not isinstance(results, list)
    assert result_names(list(results)) == ["Pull Up"]


def test_search_returns_search_results_with_raw_cosine_similarity(tmp_path):
    EmbeddedExerciseSearchResult = get_embedded_exercise_search_result_model()
    cache_path = tmp_path / "embedded_exercises.sqlite"
    write_sqlite_cache(
        cache_path,
        [
            embedded_exercise_named("Pull Up", [1.0, 0.0]),
            embedded_exercise_named("Body Row", [0.8, 0.2]),
        ],
    )
    repository = build_sqlite_repository(cache_path)

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=2,
        )
    )

    assert all(isinstance(result, EmbeddedExerciseSearchResult) for result in results)
    assert result_names(results) == ["Pull Up", "Body Row"]
    assert result_similarities(results) == pytest.approx([1.0, 0.9701425])


def test_search_applies_current_equipment_filtering_semantics(tmp_path):
    cache_path = tmp_path / "embedded_exercises.sqlite"
    write_sqlite_cache(
        cache_path,
        [
            embedded_exercise_named("Pull Up", [1.0, 0.0], materials=[" Bar "]),
            embedded_exercise_named("Ring Row", [0.99, 0.01], materials=["Rings"]),
            embedded_exercise_named(
                "Transition",
                [0.98, 0.02],
                materials=["Bar", "Rings"],
            ),
            embedded_exercise_named("Hollow Body Hold", [0.0, 1.0], materials=[]),
        ],
    )
    repository = build_sqlite_repository(cache_path)

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["bar", " Rings "],
            limit=4,
        )
    )

    assert result_names(results) == [
        "Pull Up",
        "Ring Row",
        "Transition",
        "Hollow Body Hold",
    ]


def test_search_returns_exact_top_k_by_cosine_similarity(tmp_path):
    cache_path = tmp_path / "embedded_exercises.sqlite"
    write_sqlite_cache(
        cache_path,
        [
            embedded_exercise_named("Low Match", [0.0, 1.0]),
            embedded_exercise_named("Best Match", [1.0, 0.0]),
            embedded_exercise_named("Second Match", [0.8, 0.2]),
            embedded_exercise_named("Third Match", [0.6, 0.4]),
        ],
    )
    repository = build_sqlite_repository(cache_path)

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=3,
        )
    )

    assert result_names(results) == ["Best Match", "Second Match", "Third Match"]


def test_search_preserves_input_order_for_equal_scores(tmp_path):
    cache_path = tmp_path / "embedded_exercises.sqlite"
    write_sqlite_cache(
        cache_path,
        [
            embedded_exercise_named("Chin Up", [1.0, 0.0]),
            embedded_exercise_named("Neutral Grip Pull Up", [1.0, 0.0]),
            embedded_exercise_named("Body Row", [0.8, 0.2]),
        ],
    )
    repository = build_sqlite_repository(cache_path)

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=3,
        )
    )

    assert result_names(results) == [
        "Chin Up",
        "Neutral Grip Pull Up",
        "Body Row",
    ]


@pytest.mark.parametrize("limit", [0, -1])
def test_search_raises_for_invalid_limit_before_scanning_repository(tmp_path, limit):
    cache_path = tmp_path / "missing_embedded_exercises.sqlite"
    repository = build_sqlite_repository(cache_path)

    with pytest.raises(ValueError, match="limit"):
        list(
            repository.search(
                query_embedding=[1.0, 0.0],
                available_equipment=["Bar"],
                limit=limit,
            )
        )

    assert not cache_path.exists()


def test_search_raises_for_missing_cache_path_without_creating_db(tmp_path):
    cache_path = tmp_path / "missing_embedded_exercises.sqlite"
    repository = build_sqlite_repository(cache_path)

    with pytest.raises(ValueError, match=r"does not exist|missing"):
        list(
            repository.search(
                query_embedding=[1.0, 0.0],
                available_equipment=["Bar"],
                limit=1,
            )
        )

    assert not cache_path.exists()


def test_search_raises_clearly_for_malformed_sqlite_rows(tmp_path):
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        create_schema(connection)
        insert_metadata(connection)
        insert_embedded_exercise_row(connection, embedding="{not json")
    repository = build_sqlite_repository(cache_path)

    with pytest.raises(ValueError, match=r"row id 1|embedding"):
        list(
            repository.search(
                query_embedding=[1.0, 0.0],
                available_equipment=["Bar"],
                limit=1,
            )
        )


def test_search_supports_batch_streaming_via_sqlite_repository_batch_size(tmp_path):
    cache_path = tmp_path / "embedded_exercises.sqlite"
    write_sqlite_cache(
        cache_path,
        [
            embedded_exercise_named("Pull Up", [1.0, 0.0]),
            embedded_exercise_named("Body Row", [0.8, 0.2]),
            embedded_exercise_named("Scap Pull", [0.7, 0.3]),
        ],
    )
    repository = build_sqlite_repository(cache_path, batch_size=1)

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=3,
        )
    )

    assert result_names(results) == ["Pull Up", "Body Row", "Scap Pull"]


def test_jsonl_and_sqlite_search_return_equivalent_results_for_same_embedded_records(
    tmp_path,
):
    LocalEmbeddedExerciseCache = get_local_embedded_exercise_cache()
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    JsonlEmbeddedExerciseSearchRepository = get_jsonl_embedded_exercise_search_repository()
    jsonl_path = tmp_path / "embedded_exercises.jsonl"
    sqlite_path = tmp_path / "embedded_exercises.sqlite"
    embedded_exercises = [
        embedded_exercise_named("Pull Up", [1.0, 0.0], materials=["Bar"]),
        embedded_exercise_named("Ring Row", [0.99, 0.01], materials=["Rings"]),
        embedded_exercise_named("Body Row", [0.8, 0.2], materials=["Bar"]),
        embedded_exercise_named("Hollow Body Hold", [0.0, 1.0], materials=[]),
    ]

    LocalEmbeddedExerciseCache(jsonl_path).write_embedded_exercises(
        embedded_exercises,
        build_metadata(),
    )
    write_sqlite_cache(sqlite_path, embedded_exercises)

    jsonl_results = list(
        JsonlEmbeddedExerciseSearchRepository(
            LocalEmbeddedExerciseRepository(jsonl_path)
        ).search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=3,
        )
    )
    sqlite_results = list(
        build_sqlite_repository(sqlite_path).search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=3,
        )
    )

    assert result_names(sqlite_results) == result_names(jsonl_results)
    assert result_similarities(sqlite_results) == pytest.approx(
        result_similarities(jsonl_results)
    )
