from importlib import import_module
import json
import logging
import sqlite3

import pytest


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_exercise_repository_protocol():
    module = import_module("calisthenics_recommender.ports.exercise_repository")
    return getattr(module, "ExerciseRepository")


def get_sqlite_exercise_repository():
    module = import_module(
        "calisthenics_recommender.adapters.sqlite_exercise_repository"
    )
    return getattr(module, "SQLiteExerciseRepository")


def get_write_exercises_to_sqlite():
    module = import_module(
        "calisthenics_recommender.adapters.sqlite_exercise_repository"
    )
    return getattr(module, "write_exercises_to_sqlite")


def make_exercise(**overrides):
    Exercise = get_exercise_model()
    payload = {
        "name": "Pull Up Negative",
        "description": "A controlled eccentric pull-up variation.",
        "muscle_groups": ["Back", "Biceps"],
        "families": ["Pull-up"],
        "materials": ["Bar"],
        "categories": ["Upper Body Pull"],
    }
    payload.update(overrides)
    return Exercise(**payload)


def write_malformed_sqlite_row(sqlite_path, **overrides):
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            CREATE TABLE exercises (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                muscle_groups TEXT NOT NULL,
                families TEXT NOT NULL,
                materials TEXT NOT NULL,
                categories TEXT NOT NULL
            )
            """
        )
        row = {
            "name": "Pull Up Negative",
            "description": "A controlled eccentric pull-up variation.",
            "muscle_groups": json.dumps(["Back", "Biceps"]),
            "families": json.dumps(["Pull-up"]),
            "materials": json.dumps(["Bar"]),
            "categories": json.dumps(["Upper Body Pull"]),
        }
        row.update(overrides)
        connection.execute(
            """
            INSERT INTO exercises (
                name,
                description,
                muscle_groups,
                families,
                materials,
                categories
            )
            VALUES (
                :name,
                :description,
                :muscle_groups,
                :families,
                :materials,
                :categories
            )
            """,
            row,
        )


def test_sqlite_exercise_repository_implements_exercise_repository_protocol(tmp_path):
    ExerciseRepository = get_exercise_repository_protocol()
    SQLiteExerciseRepository = get_sqlite_exercise_repository()
    write_exercises_to_sqlite = get_write_exercises_to_sqlite()
    sqlite_path = tmp_path / "exercises.sqlite"

    write_exercises_to_sqlite(sqlite_path, [make_exercise()])
    repository = SQLiteExerciseRepository(sqlite_path)

    assert isinstance(repository, ExerciseRepository)
    assert not isinstance(repository.iter_exercises(), list)


def test_sqlite_exercise_repository_accepts_string_sqlite_path(tmp_path):
    SQLiteExerciseRepository = get_sqlite_exercise_repository()
    write_exercises_to_sqlite = get_write_exercises_to_sqlite()
    sqlite_path = tmp_path / "exercises.sqlite"
    write_exercises_to_sqlite(sqlite_path, [make_exercise()])

    repository = SQLiteExerciseRepository(str(sqlite_path))

    exercises = list(repository.iter_exercises())

    assert [exercise.name for exercise in exercises] == ["Pull Up Negative"]


def test_sqlite_exercise_repository_loads_exercises_in_insert_order(tmp_path):
    SQLiteExerciseRepository = get_sqlite_exercise_repository()
    write_exercises_to_sqlite = get_write_exercises_to_sqlite()
    sqlite_path = tmp_path / "exercises.sqlite"
    expected_exercises = [
        make_exercise(
            name="Pull Up",
            description="A strict vertical pulling movement.",
            materials=["Bar"],
        ),
        make_exercise(
            name="Pull Up",
            description="A paused pull-up variation with the same name.",
            materials=["Bar"],
        ),
        make_exercise(
            name="Body Row",
            description="A horizontal pulling variation.",
            families=["Row"],
            materials=[],
        ),
    ]

    write_exercises_to_sqlite(sqlite_path, expected_exercises)
    exercises = list(SQLiteExerciseRepository(sqlite_path).iter_exercises())

    assert exercises == expected_exercises
    assert [exercise.name for exercise in exercises] == ["Pull Up", "Pull Up", "Body Row"]
    assert exercises[2].materials == []


def test_sqlite_exercise_repository_streams_rows_and_logs_only_safe_messages(
    tmp_path, caplog
):
    SQLiteExerciseRepository = get_sqlite_exercise_repository()
    write_exercises_to_sqlite = get_write_exercises_to_sqlite()
    sqlite_path = tmp_path / "exercises.sqlite"
    write_exercises_to_sqlite(
        sqlite_path,
        [
            make_exercise(
                description="A controlled eccentric pull-up variation for building strength."
            )
        ],
    )
    repository = SQLiteExerciseRepository(sqlite_path, batch_size=1)

    with caplog.at_level(logging.INFO):
        iterator = iter(repository.iter_exercises())
        first_exercise = next(iterator)

        assert first_exercise.name == "Pull Up Negative"
        assert "Starting exercise SQLite scan" in caplog.text
        assert str(sqlite_path) in caplog.text
        assert "Finished exercise SQLite scan" not in caplog.text

        remaining_exercises = list(iterator)

    assert remaining_exercises == []
    assert "Finished exercise SQLite scan" in caplog.text
    assert "with 1 exercises" in caplog.text
    assert "A controlled eccentric pull-up variation for building strength." not in caplog.text


def test_sqlite_exercise_repository_yields_first_valid_row_before_later_row_error(
    tmp_path,
):
    SQLiteExerciseRepository = get_sqlite_exercise_repository()
    sqlite_path = tmp_path / "exercises.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            CREATE TABLE exercises (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                muscle_groups TEXT NOT NULL,
                families TEXT NOT NULL,
                materials TEXT NOT NULL,
                categories TEXT NOT NULL
            )
            """
        )
        for row in [
            {
                "name": "Pull Up Negative",
                "description": "A controlled eccentric pull-up variation.",
                "muscle_groups": json.dumps(["Back", "Biceps"]),
                "families": json.dumps(["Pull-up"]),
                "materials": json.dumps(["Bar"]),
                "categories": json.dumps(["Upper Body Pull"]),
            },
            {
                "name": "Body Row",
                "description": "A horizontal pulling variation.",
                "muscle_groups": json.dumps(["Back", "Biceps"]),
                "families": json.dumps(["Row"]),
                "materials": json.dumps(["Bar"]),
                "categories": '"not-a-list"',
            },
        ]:
            connection.execute(
                """
                INSERT INTO exercises (
                    name,
                    description,
                    muscle_groups,
                    families,
                    materials,
                    categories
                )
                VALUES (
                    :name,
                    :description,
                    :muscle_groups,
                    :families,
                    :materials,
                    :categories
                )
                """,
                row,
            )

    iterator = iter(SQLiteExerciseRepository(sqlite_path, batch_size=1).iter_exercises())
    first_exercise = next(iterator)

    assert first_exercise.name == "Pull Up Negative"

    with pytest.raises(ValueError, match=r"row id 2|categories"):
        next(iterator)


@pytest.mark.parametrize(
    ("field_name", "stored_value"),
    [
        ("muscle_groups", "{not json"),
        ("families", '"not-a-list"'),
        ("materials", json.dumps(["Bar", 123])),
        ("categories", json.dumps([])),
    ],
)
def test_sqlite_exercise_repository_raises_clear_errors_for_invalid_list_fields(
    tmp_path, field_name, stored_value
):
    SQLiteExerciseRepository = get_sqlite_exercise_repository()
    sqlite_path = tmp_path / "exercises.sqlite"
    write_malformed_sqlite_row(sqlite_path, **{field_name: stored_value})

    with pytest.raises(ValueError, match=rf"row id 1|{field_name}"):
        list(SQLiteExerciseRepository(sqlite_path).iter_exercises())


def test_sqlite_exercise_repository_raises_clear_error_for_invalid_domain_payload(
    tmp_path,
):
    SQLiteExerciseRepository = get_sqlite_exercise_repository()
    sqlite_path = tmp_path / "exercises.sqlite"
    write_malformed_sqlite_row(sqlite_path, name="   ")

    with pytest.raises(ValueError, match=r"row id 1|name"):
        list(SQLiteExerciseRepository(sqlite_path).iter_exercises())


def test_sqlite_exercise_repository_rejects_non_positive_batch_size(tmp_path):
    SQLiteExerciseRepository = get_sqlite_exercise_repository()

    with pytest.raises(ValueError, match="batch_size"):
        SQLiteExerciseRepository(tmp_path / "exercises.sqlite", batch_size=0)
