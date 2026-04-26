from __future__ import annotations

import csv
import logging
from pathlib import Path

from pydantic import ValidationError

from calisthenics_recommender.domain.exercise import Exercise


logger = logging.getLogger(__name__)

_REQUIRED_HEADERS = (
    "name",
    "description",
    "muscle_groups",
    "families",
    "materials",
    "categories",
)
_LIST_FIELDS = ("muscle_groups", "families", "materials", "categories")
_REQUIRED_NON_EMPTY_LIST_FIELDS = ("muscle_groups", "families", "categories")


class CsvExerciseRepository:
    def __init__(self, csv_path: Path | str) -> None:
        self._csv_path = Path(csv_path)

    def list_exercises(self) -> list[Exercise]:
        logger.info("Loading exercises from CSV path %s", self._csv_path)

        with self._csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self._validate_headers(reader.fieldnames)

            exercises = [
                self._build_exercise(row_number, row)
                for row_number, row in enumerate(reader, start=2)
            ]

        logger.info("Loaded %s exercises from CSV path %s", len(exercises), self._csv_path)
        return exercises

    def _validate_headers(self, fieldnames: list[str] | None) -> None:
        actual_headers = [] if fieldnames is None else fieldnames
        missing_headers = [
            header for header in _REQUIRED_HEADERS if header not in actual_headers
        ]
        if missing_headers:
            missing_headers_text = ", ".join(missing_headers)
            raise ValueError(f"Missing required headers: {missing_headers_text}")

    def _build_exercise(self, row_number: int, row: dict[str, str | None]) -> Exercise:
        parsed_row = {
            "name": row.get("name"),
            "description": row.get("description"),
            "muscle_groups": self._parse_list_field(
                row_number=row_number,
                field_name="muscle_groups",
                raw_value=row.get("muscle_groups"),
            ),
            "families": self._parse_list_field(
                row_number=row_number,
                field_name="families",
                raw_value=row.get("families"),
            ),
            "materials": self._parse_list_field(
                row_number=row_number,
                field_name="materials",
                raw_value=row.get("materials"),
            ),
            "categories": self._parse_list_field(
                row_number=row_number,
                field_name="categories",
                raw_value=row.get("categories"),
            ),
        }
        self._validate_required_list_fields(row_number, parsed_row)

        try:
            return Exercise(**parsed_row)
        except ValidationError as error:
            field_names = ", ".join(
                str(issue["loc"][-1]) for issue in error.errors() if issue.get("loc")
            )
            logger.warning(
                "Invalid CSV row %s in %s (%s)",
                row_number,
                self._csv_path,
                field_names or "validation error",
            )
            detail = field_names or "validation error"
            raise ValueError(f"Invalid row {row_number}: {detail}") from error

    def _validate_required_list_fields(
        self, row_number: int, parsed_row: dict[str, str | list[str] | None]
    ) -> None:
        for field_name in _REQUIRED_NON_EMPTY_LIST_FIELDS:
            field_value = parsed_row[field_name]
            if isinstance(field_value, list) and field_value:
                continue

            logger.warning(
                "Invalid CSV row %s in %s (%s)",
                row_number,
                self._csv_path,
                field_name,
            )
            raise ValueError(f"Invalid row {row_number}: {field_name}")

    def _parse_list_field(
        self, row_number: int, field_name: str, raw_value: str | None
    ) -> list[str]:
        if field_name not in _LIST_FIELDS:
            raise ValueError(f"Unsupported list field: {field_name}")
        if raw_value is None:
            logger.warning(
                "Invalid CSV row %s in %s (%s)",
                row_number,
                self._csv_path,
                field_name,
            )
            raise ValueError(f"Invalid row {row_number}: {field_name}")

        return [item.strip() for item in raw_value.split(";") if item.strip()]
