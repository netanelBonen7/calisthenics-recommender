from importlib import import_module
import json
import math
import sqlite3

import pytest


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_model():
    module = import_module("calisthenics_recommender.domain.embedded_exercise")
    return getattr(module, "EmbeddedExercise")


def get_embedded_exercise_repository_protocol():
    module = import_module(
        "calisthenics_recommender.ports.embedded_exercise_repository"
    )
    return getattr(module, "EmbeddedExerciseRepository")


def get_cache_metadata_model():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "EmbeddedExerciseCacheMetadata")


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


def get_read_jsonl_metadata():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "read_embedded_exercise_cache_metadata")


def get_sqlite_embedded_exercise_cache():
    module = import_module(
        "calisthenics_recommender.adapters.sqlite_embedded_exercise_cache"
    )
    return getattr(module, "SQLiteEmbeddedExerciseCache")


def get_sqlite_embedded_exercise_repository():
    module = import_module(
        "calisthenics_recommender.adapters.sqlite_embedded_exercise_cache"
    )
    return getattr(module, "SQLiteEmbeddedExerciseRepository")


def get_read_sqlite_metadata():
    module = import_module(
        "calisthenics_recommender.adapters.sqlite_embedded_exercise_cache"
    )
    return getattr(module, "read_sqlite_embedded_exercise_cache_metadata")


def exercise_id_for(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def make_embedded_exercise(
    *,
    exercise_id: str | None = None,
    name: str = "Pull Up Negative",
    description: str = "A controlled eccentric pull-up variation.",
    muscle_groups: list[str] | None = None,
    families: list[str] | None = None,
    materials: list[str] | None = None,
    categories: list[str] | None = None,
    embedding: list[float] | None = None,
):
    Exercise = get_exercise_model()
    EmbeddedExercise = get_embedded_exercise_model()
    exercise = Exercise(
        exercise_id=exercise_id_for(name) if exercise_id is None else exercise_id,
        name=name,
        description=description,
        muscle_groups=["Back", "Biceps"] if muscle_groups is None else muscle_groups,
        families=["Pull-up"] if families is None else families,
        materials=["Bar"] if materials is None else materials,
        categories=["Upper Body Pull"] if categories is None else categories,
    )
    return EmbeddedExercise(
        exercise=exercise,
        embedding=[1.0, 0.0, 0.0] if embedding is None else embedding,
    )


def make_metadata(**overrides):
    EmbeddedExerciseCacheMetadata = get_cache_metadata_model()
    payload = {
        "embedding_model": "test-model",
        "embedding_dimension": 3,
        "text_builder_version": "v1",
    }
    payload.update(overrides)
    return EmbeddedExerciseCacheMetadata(**payload)


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
        "embedding_dimension": 3,
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
        "exercise_id": "pull-up-negative",
        "name": "Pull Up Negative",
        "description": "A controlled eccentric pull-up variation.",
        "muscle_groups": json.dumps(["Back", "Biceps"]),
        "families": json.dumps(["Pull-up"]),
        "materials": json.dumps(["Bar"]),
        "categories": json.dumps(["Upper Body Pull"]),
        "embedding": json.dumps([1.0, 0.0, 0.0]),
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


def test_sqlite_embedded_exercise_repository_implements_protocol(tmp_path):
    EmbeddedExerciseRepository = get_embedded_exercise_repository_protocol()
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.sqlite"

    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        [make_embedded_exercise()],
        make_metadata(),
    )
    repository = SQLiteEmbeddedExerciseRepository(str(cache_path))

    assert isinstance(repository, EmbeddedExerciseRepository)
    assert not isinstance(repository.iter_embedded_exercises(), list)


def test_sqlite_embedded_exercise_cache_accepts_string_cache_path(tmp_path):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.sqlite"

    SQLiteEmbeddedExerciseCache(str(cache_path)).write_embedded_exercises(
        [make_embedded_exercise()],
        make_metadata(),
    )

    loaded = list(SQLiteEmbeddedExerciseRepository(str(cache_path)).iter_embedded_exercises())

    assert [item.exercise.name for item in loaded] == ["Pull Up Negative"]


def test_sqlite_embedded_exercise_cache_writes_and_reads_metadata(tmp_path):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    read_sqlite_metadata = get_read_sqlite_metadata()
    cache_path = tmp_path / "embedded_exercises.sqlite"
    metadata = make_metadata(
        embedding_model="custom-model",
        embedding_dimension=3,
        text_builder_version="v2-test",
    )

    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises([], metadata)

    assert read_sqlite_metadata(cache_path) == metadata


def test_sqlite_embedded_exercise_cache_creates_unique_exercise_id_column(tmp_path):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    cache_path = tmp_path / "embedded_exercises.sqlite"

    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        [make_embedded_exercise()],
        make_metadata(),
    )

    with sqlite3.connect(cache_path) as connection:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(embedded_exercises)")
        ]
        indexes = [
            tuple(row) for row in connection.execute("PRAGMA index_list(embedded_exercises)")
        ]

    assert columns == [
        "id",
        "exercise_id",
        "name",
        "description",
        "muscle_groups",
        "families",
        "materials",
        "categories",
        "embedding",
    ]
    assert any(index[2] == 1 for index in indexes)


def test_sqlite_embedded_exercise_cache_writes_and_repository_reads_records(tmp_path):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.sqlite"
    embedded_exercises = [
        make_embedded_exercise(
            exercise_id="pull-up",
            name="Pull Up",
            description="A strict vertical pulling movement.",
            embedding=[1.0, 0.0, 0.0],
        ),
        make_embedded_exercise(
            exercise_id="paused-pull-up",
            name="Pull Up",
            description="A paused pull-up variation with the same name.",
            embedding=[0.9, 0.1, 0.0],
        ),
        make_embedded_exercise(
            exercise_id="body-row",
            name="Body Row",
            description="A horizontal pulling variation.",
            families=["Row"],
            materials=["Bar", "Rings"],
            embedding=[0.8, 0.2, 0.0],
        ),
    ]

    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        embedded_exercises,
        make_metadata(),
    )
    loaded = list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())

    assert loaded == embedded_exercises
    assert [item.exercise.name for item in loaded] == ["Pull Up", "Pull Up", "Body Row"]
    assert [item.exercise.exercise_id for item in loaded] == [
        "pull-up",
        "paused-pull-up",
        "body-row",
    ]
    assert loaded[0].embedding == (1.0, 0.0, 0.0)
    assert isinstance(loaded[0].embedding, tuple)
    assert loaded[2].exercise.materials == ["Bar", "Rings"]


def test_sqlite_embedded_exercise_cache_reads_empty_cache(tmp_path):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.sqlite"

    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises([], make_metadata())

    assert list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises()) == []


def test_sqlite_embedded_exercise_cache_preserves_vector_dimension_and_list_fields(
    tmp_path,
):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.sqlite"
    embedded_exercise = make_embedded_exercise(
        muscle_groups=[" Back ", ""],
        families=["Pull-up", "Strength"],
        materials=[],
        categories=["Upper Body Pull"],
        embedding=[1, 2.5, -3],
    )

    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        [embedded_exercise],
        make_metadata(),
    )
    loaded = list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())

    assert loaded[0].embedding == (1.0, 2.5, -3.0)
    assert loaded[0].exercise.muscle_groups == [" Back ", ""]
    assert loaded[0].exercise.materials == []


def test_sqlite_embedded_exercise_cache_replaces_existing_contents(tmp_path):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    read_sqlite_metadata = get_read_sqlite_metadata()
    cache_path = tmp_path / "embedded_exercises.sqlite"

    cache = SQLiteEmbeddedExerciseCache(cache_path)
    cache.write_embedded_exercises(
        [
            make_embedded_exercise(name="Pull Up", embedding=[1.0, 0.0, 0.0]),
            make_embedded_exercise(
                exercise_id="body-row",
                name="Body Row",
                embedding=[0.8, 0.2, 0.0],
            ),
        ],
        make_metadata(embedding_model="first-model"),
    )
    replacement_metadata = make_metadata(embedding_model="replacement-model")
    cache.write_embedded_exercises(
        [make_embedded_exercise(name="Dip", embedding=[0.0, 1.0, 0.0])],
        replacement_metadata,
    )

    loaded = list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())

    assert read_sqlite_metadata(cache_path) == replacement_metadata
    assert [item.exercise.name for item in loaded] == ["Dip"]


def test_sqlite_embedded_exercise_cache_replaces_legacy_schema_without_exercise_id(
    tmp_path,
):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.sqlite"
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            """
            CREATE TABLE embedded_exercises (
                id INTEGER PRIMARY KEY,
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

    expected_exercise = make_embedded_exercise()
    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        [expected_exercise],
        make_metadata(),
    )

    assert list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises()) == [
        expected_exercise
    ]


def test_sqlite_embedded_exercise_repository_rejects_non_positive_batch_size(tmp_path):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()

    with pytest.raises(ValueError, match="batch_size"):
        SQLiteEmbeddedExerciseRepository(tmp_path / "embedded.sqlite", batch_size=0)


def test_sqlite_embedded_exercise_cache_rejects_invalid_metadata(tmp_path):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()

    with pytest.raises(ValueError, match="metadata"):
        SQLiteEmbeddedExerciseCache(
            tmp_path / "embedded.sqlite"
        ).write_embedded_exercises([], metadata=object())


@pytest.mark.parametrize("non_finite_value", [math.nan, math.inf])
def test_sqlite_embedded_exercise_cache_rejects_non_finite_embedding_values_on_write(
    tmp_path, non_finite_value
):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()

    with pytest.raises(ValueError, match=r"row 1|embedding"):
        SQLiteEmbeddedExerciseCache(
            tmp_path / "embedded.sqlite"
        ).write_embedded_exercises(
            [make_embedded_exercise(embedding=[1.0, non_finite_value, 0.0])],
            make_metadata(),
        )


def test_sqlite_embedded_exercise_cache_rejects_dimension_mismatch_on_write(tmp_path):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()

    with pytest.raises(ValueError, match=r"row 1|dimension"):
        SQLiteEmbeddedExerciseCache(
            tmp_path / "embedded.sqlite"
        ).write_embedded_exercises(
            [make_embedded_exercise(embedding=[1.0, 0.0])],
            make_metadata(),
        )


def test_sqlite_embedded_exercise_cache_rejects_duplicate_exercise_ids(tmp_path):
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()

    with pytest.raises(sqlite3.IntegrityError, match="exercise_id|UNIQUE"):
        SQLiteEmbeddedExerciseCache(
            tmp_path / "embedded.sqlite"
        ).write_embedded_exercises(
            [
                make_embedded_exercise(
                    exercise_id="duplicate-id",
                    name="Pull Up",
                    embedding=[1.0, 0.0, 0.0],
                ),
                make_embedded_exercise(
                    exercise_id="duplicate-id",
                    name="Body Row",
                    embedding=[0.8, 0.2, 0.0],
                ),
            ],
            make_metadata(),
        )


def test_sqlite_embedded_exercise_repository_raises_for_malformed_vector_json(
    tmp_path,
):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        create_schema(connection)
        insert_metadata(connection)
        insert_embedded_exercise_row(connection, embedding="{not json")

    with pytest.raises(ValueError, match=r"row id 1|embedding"):
        list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_sqlite_embedded_exercise_repository_raises_for_malformed_list_field_json(
    tmp_path,
):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        create_schema(connection)
        insert_metadata(connection)
        insert_embedded_exercise_row(connection, families="{not json")

    with pytest.raises(ValueError, match=r"row id 1|families"):
        list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_sqlite_embedded_exercise_repository_raises_for_non_list_list_field(
    tmp_path,
):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        create_schema(connection)
        insert_metadata(connection)
        insert_embedded_exercise_row(connection, materials='"not-a-list"')

    with pytest.raises(ValueError, match=r"row id 1|materials"):
        list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_sqlite_embedded_exercise_repository_raises_for_non_string_list_item(
    tmp_path,
):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        create_schema(connection)
        insert_metadata(connection)
        insert_embedded_exercise_row(connection, categories=json.dumps(["Pull", 123]))

    with pytest.raises(ValueError, match=r"row id 1|categories"):
        list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_sqlite_embedded_exercise_repository_raises_for_invalid_exercise_payload(
    tmp_path,
):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        create_schema(connection)
        insert_metadata(connection)
        insert_embedded_exercise_row(connection, name="   ")

    with pytest.raises(ValueError, match=r"row id 1|name"):
        list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_sqlite_embedded_exercise_repository_raises_for_missing_metadata_table(
    tmp_path,
):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        connection.execute(
            """
            CREATE TABLE embedded_exercises (
                id INTEGER PRIMARY KEY,
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

    with pytest.raises(ValueError, match="metadata"):
        list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_read_sqlite_embedded_exercise_cache_metadata_raises_for_missing_cache_path(
    tmp_path,
):
    read_sqlite_metadata = get_read_sqlite_metadata()
    cache_path = tmp_path / "missing_embedded_exercises.sqlite"

    with pytest.raises(ValueError, match=r"does not exist|missing"):
        read_sqlite_metadata(cache_path)

    assert not cache_path.exists()


def test_sqlite_embedded_exercise_repository_raises_for_missing_cache_path(tmp_path):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "missing_embedded_exercises.sqlite"

    with pytest.raises(ValueError, match=r"does not exist|missing"):
        list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())

    assert not cache_path.exists()


def test_sqlite_embedded_exercise_repository_raises_for_missing_metadata_row(
    tmp_path,
):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        create_schema(connection)

    with pytest.raises(ValueError, match="metadata"):
        list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_sqlite_embedded_exercise_repository_raises_for_malformed_metadata(tmp_path):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        create_schema(connection)
        insert_metadata(connection, embedding_dimension="not-an-int")

    with pytest.raises(ValueError, match=r"metadata|embedding_dimension"):
        list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_sqlite_embedded_exercise_repository_yields_first_row_before_later_error(
    tmp_path,
):
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    cache_path = tmp_path / "embedded.sqlite"
    with sqlite3.connect(cache_path) as connection:
        create_schema(connection)
        insert_metadata(connection)
        insert_embedded_exercise_row(connection, name="Pull Up Negative")
        insert_embedded_exercise_row(connection, name="Body Row", embedding="{not json")

    iterator = iter(
        SQLiteEmbeddedExerciseRepository(
            cache_path, batch_size=1
        ).iter_embedded_exercises()
    )
    first_embedded_exercise = next(iterator)

    assert first_embedded_exercise.exercise.name == "Pull Up Negative"

    with pytest.raises(ValueError, match=r"row id 2|embedding"):
        next(iterator)


def test_jsonl_and_sqlite_embedded_caches_round_trip_equivalent_records(tmp_path):
    LocalEmbeddedExerciseCache = get_local_embedded_exercise_cache()
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    SQLiteEmbeddedExerciseCache = get_sqlite_embedded_exercise_cache()
    SQLiteEmbeddedExerciseRepository = get_sqlite_embedded_exercise_repository()
    read_jsonl_metadata = get_read_jsonl_metadata()
    read_sqlite_metadata = get_read_sqlite_metadata()
    jsonl_path = tmp_path / "embedded_exercises.jsonl"
    sqlite_path = tmp_path / "embedded_exercises.sqlite"
    metadata = make_metadata()
    embedded_exercises = [
        make_embedded_exercise(name="Pull Up", embedding=[1.0, 0.0, 0.0]),
        make_embedded_exercise(
            name="Body Row",
            families=["Row"],
            materials=["Bar", "Rings"],
            embedding=[0.8, 0.2, 0.0],
        ),
    ]

    LocalEmbeddedExerciseCache(jsonl_path).write_embedded_exercises(
        embedded_exercises,
        metadata,
    )
    SQLiteEmbeddedExerciseCache(sqlite_path).write_embedded_exercises(
        embedded_exercises,
        metadata,
    )

    assert read_sqlite_metadata(sqlite_path) == read_jsonl_metadata(jsonl_path)
    assert list(
        SQLiteEmbeddedExerciseRepository(sqlite_path).iter_embedded_exercises()
    ) == list(LocalEmbeddedExerciseRepository(jsonl_path).iter_embedded_exercises())
