import json
import sqlite3

from calisthenics_recommender.adapters.sqlite_exercise_repository import (
    SQLiteExerciseRepository,
    write_exercises_to_sqlite,
)
from calisthenics_recommender.adapters.sqlite_pending_embedding_update_repository import (
    SQLitePendingEmbeddingUpdateRepository,
)
from calisthenics_recommender.domain.exercise import Exercise


def make_exercise(**overrides):
    payload = {
        "exercise_id": "pull-up",
        "name": "Pull Up",
        "description": "A strict vertical pulling movement.",
        "muscle_groups": ["Back", "Biceps"],
        "families": ["Pull-up"],
        "materials": ["Bar"],
        "categories": ["Upper Body Pull"],
    }
    payload.update(overrides)
    return Exercise(**payload)


def insert_raw_exercise(connection: sqlite3.Connection, **overrides) -> None:
    row = {
        "exercise_id": "pull-up",
        "name": "Pull Up",
        "description": "A strict vertical pulling movement.",
        "muscle_groups": json.dumps(["Back", "Biceps"]),
        "families": json.dumps(["Pull-up"]),
        "materials": json.dumps(["Bar"]),
        "categories": json.dumps(["Upper Body Pull"]),
    }
    row.update(overrides)
    connection.execute(
        """
        INSERT INTO exercises (
            exercise_id,
            name,
            description,
            muscle_groups,
            families,
            materials,
            categories
        )
        VALUES (
            :exercise_id,
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


def pending_rows(sqlite_path):
    with sqlite3.connect(sqlite_path) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT exercise_id, operation, version, attempt_count, last_error
                FROM pending_embedding_updates
                ORDER BY exercise_id
                """
            )
        ]


def test_raw_sqlite_insert_update_delete_coalesce_to_final_pending_state(tmp_path):
    sqlite_path = tmp_path / "exercises.sqlite"
    write_exercises_to_sqlite(sqlite_path, [])

    with sqlite3.connect(sqlite_path) as connection:
        insert_raw_exercise(connection)
        connection.execute(
            """
            UPDATE pending_embedding_updates
            SET attempt_count = 3, last_error = 'old failure'
            WHERE exercise_id = 'pull-up'
            """
        )
        connection.execute(
            """
            UPDATE exercises
            SET description = 'A stricter vertical pulling movement.'
            WHERE exercise_id = 'pull-up'
            """
        )
        connection.execute("DELETE FROM exercises WHERE exercise_id = 'pull-up'")

    assert pending_rows(sqlite_path) == [
        {
            "exercise_id": "pull-up",
            "operation": "delete",
            "version": 3,
            "attempt_count": 0,
            "last_error": None,
        }
    ]


def test_raw_sqlite_exercise_id_change_records_delete_and_upsert(tmp_path):
    sqlite_path = tmp_path / "exercises.sqlite"
    write_exercises_to_sqlite(sqlite_path, [])

    with sqlite3.connect(sqlite_path) as connection:
        insert_raw_exercise(connection)
        connection.execute(
            """
            UPDATE exercises
            SET exercise_id = 'strict-pull-up'
            WHERE exercise_id = 'pull-up'
            """
        )

    assert pending_rows(sqlite_path) == [
        {
            "exercise_id": "pull-up",
            "operation": "delete",
            "version": 2,
            "attempt_count": 0,
            "last_error": None,
        },
        {
            "exercise_id": "strict-pull-up",
            "operation": "upsert",
            "version": 1,
            "attempt_count": 0,
            "last_error": None,
        },
    ]


def test_raw_sqlite_import_rebuild_recreates_triggers_and_allows_pending_upserts(
    tmp_path,
):
    sqlite_path = tmp_path / "exercises.sqlite"

    write_exercises_to_sqlite(
        sqlite_path,
        [
            make_exercise(exercise_id="pull-up", name="Pull Up"),
            make_exercise(exercise_id="body-row", name="Body Row", families=["Row"]),
        ],
    )

    with sqlite3.connect(sqlite_path) as connection:
        triggers = [
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'trigger'
                ORDER BY name
                """
            )
        ]

    assert pending_rows(sqlite_path) == [
        {
            "exercise_id": "body-row",
            "operation": "upsert",
            "version": 1,
            "attempt_count": 0,
            "last_error": None,
        },
        {
            "exercise_id": "pull-up",
            "operation": "upsert",
            "version": 1,
            "attempt_count": 0,
            "last_error": None,
        },
    ]
    assert triggers == [
        "trg_exercises_pending_embedding_delete",
        "trg_exercises_pending_embedding_insert",
        "trg_exercises_pending_embedding_update_changed_id",
        "trg_exercises_pending_embedding_update_same_id",
    ]


def test_raw_sqlite_import_rebuild_records_pending_deletes_for_removed_exercises(
    tmp_path,
):
    sqlite_path = tmp_path / "exercises.sqlite"
    write_exercises_to_sqlite(
        sqlite_path,
        [
            make_exercise(exercise_id="pull-up", name="Pull Up"),
            make_exercise(exercise_id="dip", name="Dip", families=["Dip"]),
        ],
    )

    write_exercises_to_sqlite(
        sqlite_path,
        [make_exercise(exercise_id="pull-up", name="Pull Up")],
    )

    rows_by_id = {row["exercise_id"]: row for row in pending_rows(sqlite_path)}
    assert rows_by_id["pull-up"]["operation"] == "upsert"
    assert rows_by_id["dip"]["operation"] == "delete"


def test_pending_repository_only_clears_matching_version(tmp_path):
    sqlite_path = tmp_path / "exercises.sqlite"
    write_exercises_to_sqlite(sqlite_path, [])
    repository = SQLitePendingEmbeddingUpdateRepository(sqlite_path)

    with sqlite3.connect(sqlite_path) as connection:
        insert_raw_exercise(connection)
    update = next(iter(repository.iter_pending_updates()))
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            UPDATE exercises
            SET description = 'Updated while processing.'
            WHERE exercise_id = 'pull-up'
            """
        )

    assert repository.mark_processed(update) is False
    assert [(row["exercise_id"], row["operation"], row["version"]) for row in pending_rows(sqlite_path)] == [
        ("pull-up", "upsert", 2)
    ]


def test_sqlite_exercise_repository_get_by_exercise_id(tmp_path):
    sqlite_path = tmp_path / "exercises.sqlite"
    expected_exercise = make_exercise()
    write_exercises_to_sqlite(sqlite_path, [expected_exercise])

    repository = SQLiteExerciseRepository(sqlite_path)

    assert repository.get_by_exercise_id("pull-up") == expected_exercise
    assert repository.get_by_exercise_id("missing") is None
