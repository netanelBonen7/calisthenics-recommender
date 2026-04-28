from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
from json import JSONDecodeError
import logging
from pathlib import Path
import sqlite3
from typing import Any

from pydantic import ValidationError

from calisthenics_recommender.domain.exercise import Exercise


logger = logging.getLogger(__name__)

_LIST_FIELDS = ("muscle_groups", "families", "materials", "categories")
_REQUIRED_NON_EMPTY_LIST_FIELDS = ("muscle_groups", "families", "categories")
_DEFAULT_BATCH_SIZE = 100


class SQLiteExerciseRepository:
    def __init__(
        self, sqlite_path: Path | str, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self._sqlite_path = Path(sqlite_path)
        self._batch_size = batch_size

    def iter_exercises(self) -> Iterable[Exercise]:
        return self._iter_exercises()

    def get_by_exercise_id(self, exercise_id: str) -> Exercise | None:
        with sqlite3.connect(self._sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT
                    id,
                    exercise_id,
                    name,
                    description,
                    muscle_groups,
                    families,
                    materials,
                    categories
                FROM exercises
                WHERE exercise_id = ?
                """,
                (exercise_id,),
            ).fetchone()

        if row is None:
            return None
        return _build_exercise_from_sqlite_row(row)

    def _iter_exercises(self) -> Iterator[Exercise]:
        logger.info("Starting exercise SQLite scan from path %s", self._sqlite_path)

        with sqlite3.connect(self._sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(
                """
                SELECT
                    id,
                    exercise_id,
                    name,
                    description,
                    muscle_groups,
                    families,
                    materials,
                    categories
                FROM exercises
                ORDER BY id
                """
            )
            exercise_count = 0

            while rows := cursor.fetchmany(self._batch_size):
                for row in rows:
                    yield _build_exercise_from_sqlite_row(row)
                    exercise_count += 1

        logger.info(
            "Finished exercise SQLite scan from path %s with %s exercises",
            self._sqlite_path,
            exercise_count,
        )


def write_exercises_to_sqlite(
    sqlite_path: Path | str,
    exercises: Iterable[Exercise],
) -> None:
    normalized_sqlite_path = Path(sqlite_path)
    normalized_sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Writing exercises to SQLite path %s", normalized_sqlite_path)

    written_count = 0
    with sqlite3.connect(normalized_sqlite_path) as connection:
        previous_exercise_ids = _read_existing_exercise_ids(connection)
        connection.execute("DROP TABLE IF EXISTS exercises")
        _ensure_schema(connection)

        written_exercise_ids: set[str] = set()
        for exercise in exercises:
            _insert_exercise(connection, exercise)
            written_exercise_ids.add(exercise.exercise_id)
            written_count += 1

        removed_exercise_ids = previous_exercise_ids - written_exercise_ids
        for exercise_id in removed_exercise_ids:
            _record_pending_embedding_update(
                connection=connection,
                exercise_id=exercise_id,
                operation="delete",
            )

    logger.info(
        "Finished writing exercises to SQLite path %s with %s exercises",
        normalized_sqlite_path,
        written_count,
    )


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS exercises (
            id INTEGER PRIMARY KEY,
            exercise_id TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            muscle_groups TEXT NOT NULL,
            families TEXT NOT NULL,
            materials TEXT NOT NULL,
            categories TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_exercises_name ON exercises(name)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_embedding_updates (
            exercise_id TEXT PRIMARY KEY,
            operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
            version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
            created_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            ),
            updated_at TEXT NOT NULL DEFAULT (
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            ),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
            last_attempted_at TEXT,
            last_error TEXT
        )
        """
    )
    _ensure_pending_embedding_update_triggers(connection)


def _ensure_pending_embedding_update_triggers(connection: sqlite3.Connection) -> None:
    _drop_pending_embedding_update_triggers(connection)
    connection.execute(
        """
        CREATE TRIGGER trg_exercises_pending_embedding_insert
        AFTER INSERT ON exercises
        BEGIN
            INSERT INTO pending_embedding_updates (exercise_id, operation)
            VALUES (NEW.exercise_id, 'upsert')
            ON CONFLICT(exercise_id) DO UPDATE SET
                operation = excluded.operation,
                version = version + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                attempt_count = 0,
                last_attempted_at = NULL,
                last_error = NULL;
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_exercises_pending_embedding_update_same_id
        AFTER UPDATE ON exercises
        WHEN OLD.exercise_id = NEW.exercise_id
        BEGIN
            INSERT INTO pending_embedding_updates (exercise_id, operation)
            VALUES (NEW.exercise_id, 'upsert')
            ON CONFLICT(exercise_id) DO UPDATE SET
                operation = excluded.operation,
                version = version + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                attempt_count = 0,
                last_attempted_at = NULL,
                last_error = NULL;
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_exercises_pending_embedding_update_changed_id
        AFTER UPDATE ON exercises
        WHEN OLD.exercise_id != NEW.exercise_id
        BEGIN
            INSERT INTO pending_embedding_updates (exercise_id, operation)
            VALUES (OLD.exercise_id, 'delete')
            ON CONFLICT(exercise_id) DO UPDATE SET
                operation = excluded.operation,
                version = version + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                attempt_count = 0,
                last_attempted_at = NULL,
                last_error = NULL;

            INSERT INTO pending_embedding_updates (exercise_id, operation)
            VALUES (NEW.exercise_id, 'upsert')
            ON CONFLICT(exercise_id) DO UPDATE SET
                operation = excluded.operation,
                version = version + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                attempt_count = 0,
                last_attempted_at = NULL,
                last_error = NULL;
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_exercises_pending_embedding_delete
        AFTER DELETE ON exercises
        BEGIN
            INSERT INTO pending_embedding_updates (exercise_id, operation)
            VALUES (OLD.exercise_id, 'delete')
            ON CONFLICT(exercise_id) DO UPDATE SET
                operation = excluded.operation,
                version = version + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                attempt_count = 0,
                last_attempted_at = NULL,
                last_error = NULL;
        END
        """
    )


def _drop_pending_embedding_update_triggers(connection: sqlite3.Connection) -> None:
    for trigger_name in (
        "trg_exercises_pending_embedding_insert",
        "trg_exercises_pending_embedding_update_same_id",
        "trg_exercises_pending_embedding_update_changed_id",
        "trg_exercises_pending_embedding_delete",
    ):
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")


def _read_existing_exercise_ids(connection: sqlite3.Connection) -> set[str]:
    table_row = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = 'exercises'
        """
    ).fetchone()
    if table_row is None:
        return set()

    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(exercises)")
    }
    if "exercise_id" not in columns:
        return set()

    return {
        row[0]
        for row in connection.execute(
            "SELECT exercise_id FROM exercises WHERE exercise_id IS NOT NULL"
        )
    }


def _record_pending_embedding_update(
    connection: sqlite3.Connection,
    exercise_id: str,
    operation: str,
) -> None:
    connection.execute(
        """
        INSERT INTO pending_embedding_updates (exercise_id, operation)
        VALUES (?, ?)
        ON CONFLICT(exercise_id) DO UPDATE SET
            operation = excluded.operation,
            version = version + 1,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
            attempt_count = 0,
            last_attempted_at = NULL,
            last_error = NULL
        """,
        (exercise_id, operation),
    )


def _insert_exercise(connection: sqlite3.Connection, exercise: Exercise) -> None:
    if not isinstance(exercise, Exercise):
        raise ValueError("exercises must contain Exercise objects")

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
        {
            "exercise_id": exercise.exercise_id,
            "name": exercise.name,
            "description": exercise.description,
            "muscle_groups": json.dumps(exercise.muscle_groups),
            "families": json.dumps(exercise.families),
            "materials": json.dumps(exercise.materials),
            "categories": json.dumps(exercise.categories),
        },
    )


def _build_exercise_from_sqlite_row(row: sqlite3.Row) -> Exercise:
    row_id = row["id"]
    parsed_row = {
        "exercise_id": row["exercise_id"],
        "name": row["name"],
        "description": row["description"],
        "muscle_groups": _parse_json_list_field(
            row_id=row_id,
            field_name="muscle_groups",
            raw_value=row["muscle_groups"],
        ),
        "families": _parse_json_list_field(
            row_id=row_id,
            field_name="families",
            raw_value=row["families"],
        ),
        "materials": _parse_json_list_field(
            row_id=row_id,
            field_name="materials",
            raw_value=row["materials"],
        ),
        "categories": _parse_json_list_field(
            row_id=row_id,
            field_name="categories",
            raw_value=row["categories"],
        ),
    }
    _validate_required_list_fields(row_id=row_id, parsed_row=parsed_row)

    try:
        return Exercise(**parsed_row)
    except ValidationError as error:
        field_names = ", ".join(
            str(issue["loc"][-1]) for issue in error.errors() if issue.get("loc")
        )
        detail = field_names or "validation error"
        logger.warning(
            "Invalid SQLite exercise row id %s (%s)",
            row_id,
            detail,
        )
        raise ValueError(f"Invalid SQLite row id {row_id}: {detail}") from error


def _parse_json_list_field(
    row_id: int,
    field_name: str,
    raw_value: Any,
) -> list[str]:
    if field_name not in _LIST_FIELDS:
        raise ValueError(f"Unsupported list field: {field_name}")
    if not isinstance(raw_value, str):
        logger.warning("Invalid SQLite exercise row id %s (%s)", row_id, field_name)
        raise ValueError(f"Invalid SQLite row id {row_id}: {field_name}")

    try:
        parsed_value = json.loads(raw_value)
    except JSONDecodeError as error:
        logger.warning("Invalid SQLite exercise row id %s (%s)", row_id, field_name)
        raise ValueError(f"Invalid SQLite row id {row_id}: {field_name}") from error

    if not isinstance(parsed_value, list) or any(
        not isinstance(item, str) for item in parsed_value
    ):
        logger.warning("Invalid SQLite exercise row id %s (%s)", row_id, field_name)
        raise ValueError(f"Invalid SQLite row id {row_id}: {field_name}")

    return [item.strip() for item in parsed_value if item.strip()]


def _validate_required_list_fields(
    row_id: int, parsed_row: dict[str, str | list[str]]
) -> None:
    for field_name in _REQUIRED_NON_EMPTY_LIST_FIELDS:
        field_value = parsed_row[field_name]
        if isinstance(field_value, list) and field_value:
            continue

        logger.warning("Invalid SQLite exercise row id %s (%s)", row_id, field_name)
        raise ValueError(f"Invalid SQLite row id {row_id}: {field_name}")
