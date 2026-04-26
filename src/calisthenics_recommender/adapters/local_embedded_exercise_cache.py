from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import json
from json import JSONDecodeError
import logging
import math
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise


logger = logging.getLogger(__name__)

_METADATA_RECORD_TYPE = "metadata"
_EMBEDDED_EXERCISE_RECORD_TYPE = "embedded_exercise"


@dataclass(frozen=True)
class EmbeddedExerciseCacheMetadata:
    embedding_model: str
    embedding_dimension: int
    text_builder_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "embedding_model",
            _validate_non_empty_string(
                value=self.embedding_model,
                field_name="embedding_model",
            ),
        )
        object.__setattr__(
            self,
            "embedding_dimension",
            _validate_positive_int(
                value=self.embedding_dimension,
                field_name="embedding_dimension",
            ),
        )
        object.__setattr__(
            self,
            "text_builder_version",
            _validate_non_empty_string(
                value=self.text_builder_version,
                field_name="text_builder_version",
            ),
        )

    def to_record(self) -> dict[str, str | int]:
        return {
            "type": _METADATA_RECORD_TYPE,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "text_builder_version": self.text_builder_version,
        }


class LocalEmbeddedExerciseCache:
    def __init__(self, cache_path: Path | str) -> None:
        self._cache_path = Path(cache_path)

    def write_embedded_exercises(
        self,
        embedded_exercises: Iterable[EmbeddedExercise],
        metadata: EmbeddedExerciseCacheMetadata,
    ) -> None:
        if not isinstance(metadata, EmbeddedExerciseCacheMetadata):
            raise ValueError("metadata must be an EmbeddedExerciseCacheMetadata")

        logger.info("Writing embedded exercise cache to path %s", self._cache_path)

        written_count = 0
        with self._cache_path.open("w", encoding="utf-8", newline="\n") as cache_file:
            cache_file.write(json.dumps(metadata.to_record()) + "\n")

            for line_number, embedded_exercise in enumerate(embedded_exercises, start=2):
                record = _serialize_embedded_exercise_record(
                    embedded_exercise=embedded_exercise,
                    expected_dimension=metadata.embedding_dimension,
                    line_number=line_number,
                )
                cache_file.write(json.dumps(record) + "\n")
                written_count += 1

        logger.info(
            "Finished writing embedded exercise cache to path %s with %s embedded exercises",
            self._cache_path,
            written_count,
        )


class LocalEmbeddedExerciseRepository:
    def __init__(self, cache_path: Path | str) -> None:
        self._cache_path = Path(cache_path)

    def iter_embedded_exercises(self) -> Iterable[EmbeddedExercise]:
        return self._iter_embedded_exercises()

    def _iter_embedded_exercises(self) -> Iterator[EmbeddedExercise]:
        logger.info("Starting embedded exercise cache scan from path %s", self._cache_path)

        with self._cache_path.open("r", encoding="utf-8") as cache_file:
            metadata = self._read_metadata(cache_file)
            embedded_exercise_count = 0

            for line_number, line in enumerate(cache_file, start=2):
                record = _parse_json_line(line=line, line_number=line_number)
                yield _parse_embedded_exercise_record(
                    record=record,
                    expected_dimension=metadata.embedding_dimension,
                    line_number=line_number,
                )
                embedded_exercise_count += 1

        logger.info(
            "Finished embedded exercise cache scan from path %s with %s embedded exercises",
            self._cache_path,
            embedded_exercise_count,
        )

    def _read_metadata(self, cache_file: Any) -> EmbeddedExerciseCacheMetadata:
        return _read_embedded_exercise_cache_metadata_from_file(
            cache_file=cache_file,
            cache_path=self._cache_path,
        )


def read_embedded_exercise_cache_metadata(
    cache_path: Path | str,
) -> EmbeddedExerciseCacheMetadata:
    normalized_cache_path = Path(cache_path)

    with normalized_cache_path.open("r", encoding="utf-8") as cache_file:
        return _read_embedded_exercise_cache_metadata_from_file(
            cache_file=cache_file,
            cache_path=normalized_cache_path,
        )


def _read_embedded_exercise_cache_metadata_from_file(
    cache_file: Any,
    cache_path: Path,
) -> EmbeddedExerciseCacheMetadata:
    first_line = cache_file.readline()
    if first_line == "":
        raise ValueError(f"Embedded exercise cache file is empty: {cache_path}")

    record = _parse_json_line(line=first_line, line_number=1)
    if not isinstance(record, dict):
        raise ValueError("Invalid metadata at line 1: expected object")
    if record.get("type") != _METADATA_RECORD_TYPE:
        raise ValueError("Missing metadata at line 1")

    return _parse_metadata_record(record=record, line_number=1)


def _serialize_embedded_exercise_record(
    embedded_exercise: EmbeddedExercise,
    expected_dimension: int,
    line_number: int,
) -> dict[str, object]:
    if not isinstance(embedded_exercise, EmbeddedExercise):
        raise ValueError(
            f"Invalid embedded exercise payload at line {line_number}: expected EmbeddedExercise"
        )

    embedding = _parse_embedding_payload(
        payload=list(embedded_exercise.embedding),
        expected_dimension=expected_dimension,
        line_number=line_number,
    )
    return {
        "type": _EMBEDDED_EXERCISE_RECORD_TYPE,
        "exercise": embedded_exercise.exercise.model_dump(mode="json"),
        "embedding": list(embedding),
    }


def _parse_metadata_record(
    record: dict[str, object], line_number: int
) -> EmbeddedExerciseCacheMetadata:
    try:
        return EmbeddedExerciseCacheMetadata(
            embedding_model=record["embedding_model"],
            embedding_dimension=record["embedding_dimension"],
            text_builder_version=record["text_builder_version"],
        )
    except KeyError as error:
        missing_field = error.args[0]
        raise ValueError(
            f"Invalid metadata at line {line_number}: missing {missing_field}"
        ) from error
    except ValueError as error:
        raise ValueError(f"Invalid metadata at line {line_number}: {error}") from error


def _parse_embedded_exercise_record(
    record: object,
    expected_dimension: int,
    line_number: int,
) -> EmbeddedExercise:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid embedded exercise record at line {line_number}: expected object")

    record_type = record.get("type")
    if record_type != _EMBEDDED_EXERCISE_RECORD_TYPE:
        raise ValueError(f"Unknown record type at line {line_number}: {record_type!r}")

    exercise = _parse_exercise_payload(
        payload=record.get("exercise"),
        line_number=line_number,
    )
    embedding = _parse_embedding_payload(
        payload=record.get("embedding"),
        expected_dimension=expected_dimension,
        line_number=line_number,
    )
    return EmbeddedExercise(exercise=exercise, embedding=embedding)


def _parse_exercise_payload(payload: object, line_number: int) -> Exercise:
    if not isinstance(payload, dict):
        logger.warning(
            "Invalid embedded exercise cache record at line %s (%s)",
            line_number,
            "exercise",
        )
        raise ValueError(f"Invalid exercise payload at line {line_number}: exercise")

    try:
        return Exercise(**payload)
    except ValidationError as error:
        field_names = ", ".join(
            str(issue["loc"][-1]) for issue in error.errors() if issue.get("loc")
        )
        logger.warning(
            "Invalid embedded exercise cache record at line %s (%s)",
            line_number,
            field_names or "exercise",
        )
        detail = field_names or "exercise"
        raise ValueError(
            f"Invalid exercise payload at line {line_number}: {detail}"
        ) from error


def _parse_embedding_payload(
    payload: object,
    expected_dimension: int,
    line_number: int,
) -> tuple[float, ...]:
    if not isinstance(payload, list):
        logger.warning(
            "Invalid embedded exercise cache record at line %s (%s)",
            line_number,
            "embedding",
        )
        raise ValueError(f"Invalid embedding payload at line {line_number}: embedding")

    values: list[float] = []
    for value in payload:
        if isinstance(value, bool) or not isinstance(value, int | float):
            logger.warning(
                "Invalid embedded exercise cache record at line %s (%s)",
                line_number,
                "embedding",
            )
            raise ValueError(
                f"Invalid embedding payload at line {line_number}: embedding"
            )
        normalized_value = float(value)
        if not math.isfinite(normalized_value):
            logger.warning(
                "Invalid embedded exercise cache record at line %s (%s)",
                line_number,
                "embedding",
            )
            raise ValueError(
                f"Invalid embedding payload at line {line_number}: embedding"
            )
        values.append(normalized_value)

    if len(values) != expected_dimension:
        raise ValueError(
            "Embedding dimension mismatch at line "
            f"{line_number}: expected {expected_dimension}, got {len(values)}"
        )

    return tuple(values)


def _parse_json_line(line: str, line_number: int) -> object:
    try:
        return json.loads(line)
    except JSONDecodeError as error:
        raise ValueError(f"Malformed JSON at line {line_number}") from error


def _validate_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")

    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must be a non-empty string")

    return normalized_value


def _validate_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")

    return value
