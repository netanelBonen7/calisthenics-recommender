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

    def _iter_exercises(self) -> Iterator[Exercise]:
        logger.info("Starting exercise SQLite scan from path %s", self._sqlite_path)

        with sqlite3.connect(self._sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(
                """
                SELECT
                    id,
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
        _ensure_schema(connection)
        connection.execute("DELETE FROM exercises")

        for exercise in exercises:
            _insert_exercise(connection, exercise)
            written_count += 1

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


def _insert_exercise(connection: sqlite3.Connection, exercise: Exercise) -> None:
    if not isinstance(exercise, Exercise):
        raise ValueError("exercises must contain Exercise objects")

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
        {
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
