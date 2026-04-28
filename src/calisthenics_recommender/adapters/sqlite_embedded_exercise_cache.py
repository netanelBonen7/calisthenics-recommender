from __future__ import annotations

from collections.abc import Iterable, Iterator
import json
from json import JSONDecodeError
import logging
import math
from pathlib import Path
import sqlite3
from typing import Any

from pydantic import ValidationError

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
)
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise


logger = logging.getLogger(__name__)

_LIST_FIELDS = ("muscle_groups", "families", "materials", "categories")
_DEFAULT_BATCH_SIZE = 100


class SQLiteEmbeddedExerciseCache:
    def __init__(self, cache_path: Path | str) -> None:
        self._cache_path = Path(cache_path)

    def write_embedded_exercises(
        self,
        embedded_exercises: Iterable[EmbeddedExercise],
        metadata: EmbeddedExerciseCacheMetadata,
    ) -> None:
        if not isinstance(metadata, EmbeddedExerciseCacheMetadata):
            raise ValueError("metadata must be an EmbeddedExerciseCacheMetadata")

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Writing embedded exercise SQLite cache to path %s", self._cache_path)

        written_count = 0
        with sqlite3.connect(self._cache_path) as connection:
            connection.execute("DROP TABLE IF EXISTS embedded_exercises")
            connection.execute("DROP TABLE IF EXISTS embedding_metadata")
            _ensure_schema(connection)
            _insert_metadata(connection=connection, metadata=metadata)

            for row_number, embedded_exercise in enumerate(embedded_exercises, start=1):
                _insert_embedded_exercise(
                    connection=connection,
                    embedded_exercise=embedded_exercise,
                    expected_dimension=metadata.embedding_dimension,
                    row_number=row_number,
                )
                written_count += 1

        logger.info(
            "Finished writing embedded exercise SQLite cache to path %s with %s embedded exercises",
            self._cache_path,
            written_count,
        )

    def upsert_embedded_exercise(
        self,
        embedded_exercise: EmbeddedExercise,
        metadata: EmbeddedExerciseCacheMetadata,
    ) -> None:
        if not isinstance(metadata, EmbeddedExerciseCacheMetadata):
            raise ValueError("metadata must be an EmbeddedExerciseCacheMetadata")
        _raise_if_cache_path_missing(self._cache_path)

        with sqlite3.connect(self._cache_path) as connection:
            connection.row_factory = sqlite3.Row
            actual_metadata = _read_metadata_from_connection(
                connection=connection,
                cache_path=self._cache_path,
            )
            _validate_metadata_matches(
                actual_metadata=actual_metadata,
                expected_metadata=metadata,
                cache_path=self._cache_path,
            )
            _upsert_embedded_exercise(
                connection=connection,
                embedded_exercise=embedded_exercise,
                expected_dimension=metadata.embedding_dimension,
            )

    def delete_embedded_exercise(self, exercise_id: str) -> None:
        _raise_if_cache_path_missing(self._cache_path)

        with sqlite3.connect(self._cache_path) as connection:
            connection.execute(
                "DELETE FROM embedded_exercises WHERE exercise_id = ?",
                (exercise_id,),
            )


class SQLiteEmbeddedExerciseRepository:
    def __init__(
        self, cache_path: Path | str, batch_size: int = _DEFAULT_BATCH_SIZE
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self._cache_path = Path(cache_path)
        self._batch_size = batch_size

    def iter_embedded_exercises(self) -> Iterable[EmbeddedExercise]:
        return self._iter_embedded_exercises()

    def _iter_embedded_exercises(self) -> Iterator[EmbeddedExercise]:
        logger.info(
            "Starting embedded exercise SQLite cache scan from path %s",
            self._cache_path,
        )
        _raise_if_cache_path_missing(self._cache_path)

        with sqlite3.connect(self._cache_path) as connection:
            connection.row_factory = sqlite3.Row
            metadata = _read_metadata_from_connection(
                connection=connection,
                cache_path=self._cache_path,
            )
            cursor = _execute_embedded_exercises_query(
                connection=connection,
                cache_path=self._cache_path,
            )
            embedded_exercise_count = 0

            while rows := cursor.fetchmany(self._batch_size):
                for row in rows:
                    yield _build_embedded_exercise_from_sqlite_row(
                        row=row,
                        expected_dimension=metadata.embedding_dimension,
                    )
                    embedded_exercise_count += 1

        logger.info(
            "Finished embedded exercise SQLite cache scan from path %s with %s embedded exercises",
            self._cache_path,
            embedded_exercise_count,
        )


def read_sqlite_embedded_exercise_cache_metadata(
    cache_path: Path | str,
) -> EmbeddedExerciseCacheMetadata:
    normalized_cache_path = Path(cache_path)
    _raise_if_cache_path_missing(normalized_cache_path)

    with sqlite3.connect(normalized_cache_path) as connection:
        connection.row_factory = sqlite3.Row
        return _read_metadata_from_connection(
            connection=connection,
            cache_path=normalized_cache_path,
        )


def _raise_if_cache_path_missing(cache_path: Path) -> None:
    if not cache_path.exists():
        raise ValueError(f"SQLite embedded exercise cache file does not exist: {cache_path}")


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_metadata (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            embedding_model TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
            text_builder_version TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embedded_exercises (
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


def _insert_metadata(
    connection: sqlite3.Connection,
    metadata: EmbeddedExerciseCacheMetadata,
) -> None:
    connection.execute(
        """
        INSERT INTO embedding_metadata (
            id,
            embedding_model,
            embedding_dimension,
            text_builder_version
        )
        VALUES (
            1,
            :embedding_model,
            :embedding_dimension,
            :text_builder_version
        )
        """,
        {
            "embedding_model": metadata.embedding_model,
            "embedding_dimension": metadata.embedding_dimension,
            "text_builder_version": metadata.text_builder_version,
        },
    )


def _insert_embedded_exercise(
    connection: sqlite3.Connection,
    embedded_exercise: EmbeddedExercise,
    expected_dimension: int,
    row_number: int,
) -> None:
    if not isinstance(embedded_exercise, EmbeddedExercise):
        raise ValueError(
            f"Invalid SQLite embedded exercise row {row_number}: expected EmbeddedExercise"
        )

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
        _embedded_exercise_parameters(
            embedded_exercise=embedded_exercise,
            expected_dimension=expected_dimension,
            row_label=f"row {row_number}",
        ),
    )


def _upsert_embedded_exercise(
    connection: sqlite3.Connection,
    embedded_exercise: EmbeddedExercise,
    expected_dimension: int,
) -> None:
    if not isinstance(embedded_exercise, EmbeddedExercise):
        raise ValueError("embedded_exercise must be an EmbeddedExercise")

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
        ON CONFLICT(exercise_id) DO UPDATE SET
            name = excluded.name,
            description = excluded.description,
            muscle_groups = excluded.muscle_groups,
            families = excluded.families,
            materials = excluded.materials,
            categories = excluded.categories,
            embedding = excluded.embedding
        """,
        _embedded_exercise_parameters(
            embedded_exercise=embedded_exercise,
            expected_dimension=expected_dimension,
            row_label=f"exercise_id {embedded_exercise.exercise.exercise_id}",
        ),
    )


def _embedded_exercise_parameters(
    embedded_exercise: EmbeddedExercise,
    expected_dimension: int,
    row_label: str,
) -> dict[str, str]:
    embedding = _parse_embedding_payload(
        payload=list(embedded_exercise.embedding),
        expected_dimension=expected_dimension,
        row_label=row_label,
    )
    exercise = embedded_exercise.exercise
    return {
        "exercise_id": exercise.exercise_id,
        "name": exercise.name,
        "description": exercise.description,
        "muscle_groups": json.dumps(exercise.muscle_groups),
        "families": json.dumps(exercise.families),
        "materials": json.dumps(exercise.materials),
        "categories": json.dumps(exercise.categories),
        "embedding": json.dumps(list(embedding)),
    }


def _validate_metadata_matches(
    actual_metadata: EmbeddedExerciseCacheMetadata,
    expected_metadata: EmbeddedExerciseCacheMetadata,
    cache_path: Path,
) -> None:
    if actual_metadata == expected_metadata:
        return

    raise ValueError(
        "SQLite embedded exercise cache metadata is incompatible with the "
        f"current embedding config for {cache_path}; run a full cache rebuild"
    )


def _read_metadata_from_connection(
    connection: sqlite3.Connection,
    cache_path: Path,
) -> EmbeddedExerciseCacheMetadata:
    try:
        row = connection.execute(
            """
            SELECT
                embedding_model,
                embedding_dimension,
                text_builder_version
            FROM embedding_metadata
            WHERE id = 1
            """
        ).fetchone()
    except sqlite3.OperationalError as error:
        raise ValueError(
            f"Missing metadata in SQLite embedded exercise cache: {cache_path}"
        ) from error

    if row is None:
        raise ValueError(
            f"Missing metadata in SQLite embedded exercise cache: {cache_path}"
        )

    try:
        return EmbeddedExerciseCacheMetadata(
            embedding_model=row["embedding_model"],
            embedding_dimension=row["embedding_dimension"],
            text_builder_version=row["text_builder_version"],
        )
    except ValueError as error:
        raise ValueError(f"Invalid SQLite embedded exercise metadata: {error}") from error


def _execute_embedded_exercises_query(
    connection: sqlite3.Connection,
    cache_path: Path,
) -> sqlite3.Cursor:
    try:
        return connection.execute(
            """
            SELECT
                id,
                exercise_id,
                name,
                description,
                muscle_groups,
                families,
                materials,
                categories,
                embedding
            FROM embedded_exercises
            ORDER BY id
            """
        )
    except sqlite3.OperationalError as error:
        raise ValueError(
            f"Invalid SQLite embedded exercise cache at {cache_path}: embedded_exercises"
        ) from error


def _build_embedded_exercise_from_sqlite_row(
    row: sqlite3.Row,
    expected_dimension: int,
) -> EmbeddedExercise:
    row_id = row["id"]
    parsed_exercise = {
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
    exercise = _parse_exercise_payload(row_id=row_id, payload=parsed_exercise)
    embedding_payload = _parse_json_field(
        row_id=row_id,
        field_name="embedding",
        raw_value=row["embedding"],
    )
    embedding = _parse_embedding_payload(
        payload=embedding_payload,
        expected_dimension=expected_dimension,
        row_label=f"row id {row_id}",
    )
    return EmbeddedExercise(exercise=exercise, embedding=embedding)


def _parse_exercise_payload(row_id: int, payload: dict[str, object]) -> Exercise:
    try:
        return Exercise(**payload)
    except ValidationError as error:
        field_names = ", ".join(
            str(issue["loc"][-1]) for issue in error.errors() if issue.get("loc")
        )
        detail = field_names or "exercise"
        logger.warning(
            "Invalid SQLite embedded exercise row id %s (%s)",
            row_id,
            detail,
        )
        raise ValueError(
            f"Invalid SQLite embedded exercise row id {row_id}: {detail}"
        ) from error


def _parse_json_list_field(
    row_id: int,
    field_name: str,
    raw_value: Any,
) -> list[str]:
    if field_name not in _LIST_FIELDS:
        raise ValueError(f"Unsupported list field: {field_name}")

    parsed_value = _parse_json_field(
        row_id=row_id,
        field_name=field_name,
        raw_value=raw_value,
    )
    if not isinstance(parsed_value, list) or any(
        not isinstance(item, str) for item in parsed_value
    ):
        logger.warning(
            "Invalid SQLite embedded exercise row id %s (%s)",
            row_id,
            field_name,
        )
        raise ValueError(
            f"Invalid SQLite embedded exercise row id {row_id}: {field_name}"
        )

    return parsed_value


def _parse_json_field(row_id: int, field_name: str, raw_value: Any) -> object:
    if not isinstance(raw_value, str):
        logger.warning(
            "Invalid SQLite embedded exercise row id %s (%s)",
            row_id,
            field_name,
        )
        raise ValueError(
            f"Invalid SQLite embedded exercise row id {row_id}: {field_name}"
        )

    try:
        return json.loads(raw_value)
    except JSONDecodeError as error:
        logger.warning(
            "Invalid SQLite embedded exercise row id %s (%s)",
            row_id,
            field_name,
        )
        raise ValueError(
            f"Invalid SQLite embedded exercise row id {row_id}: {field_name}"
        ) from error


def _parse_embedding_payload(
    payload: object,
    expected_dimension: int,
    row_label: str,
) -> tuple[float, ...]:
    if not isinstance(payload, list):
        logger.warning("Invalid SQLite embedded exercise %s (%s)", row_label, "embedding")
        raise ValueError(f"Invalid SQLite embedded exercise {row_label}: embedding")

    values: list[float] = []
    for value in payload:
        if isinstance(value, bool) or not isinstance(value, int | float):
            logger.warning(
                "Invalid SQLite embedded exercise %s (%s)",
                row_label,
                "embedding",
            )
            raise ValueError(f"Invalid SQLite embedded exercise {row_label}: embedding")

        normalized_value = float(value)
        if not math.isfinite(normalized_value):
            logger.warning(
                "Invalid SQLite embedded exercise %s (%s)",
                row_label,
                "embedding",
            )
            raise ValueError(f"Invalid SQLite embedded exercise {row_label}: embedding")

        values.append(normalized_value)

    if len(values) != expected_dimension:
        raise ValueError(
            "SQLite embedded exercise embedding dimension mismatch at "
            f"{row_label}: expected {expected_dimension}, got {len(values)}"
        )

    return tuple(values)
