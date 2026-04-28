import csv
import importlib
import sqlite3
from pathlib import Path

import pytest

from calisthenics_recommender.adapters.csv_exercise_repository import (
    CsvExerciseRepository,
)
from calisthenics_recommender.adapters.sqlite_exercise_repository import (
    SQLiteExerciseRepository,
)


def load_import_exercises_to_sqlite_module():
    return importlib.import_module(
        "calisthenics_recommender.cli.import_exercises_to_sqlite"
    )


def write_exercise_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "exercise_id",
                "name",
                "description",
                "muscle_groups",
                "families",
                "materials",
                "categories",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_import_exercises_to_sqlite_main_creates_readable_sqlite_database(tmp_path):
    module = load_import_exercises_to_sqlite_module()
    csv_path = tmp_path / "exercises.csv"
    sqlite_path = tmp_path / "db" / "exercises.sqlite"
    write_exercise_csv(
        csv_path,
        [
            {
                "exercise_id": "pull-up-negative",
                "name": "Pull Up Negative",
                "description": "A controlled eccentric pull-up variation.",
                "muscle_groups": "Back;Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            },
            {
                "exercise_id": "body-row",
                "name": "Body Row",
                "description": "A horizontal pulling variation.",
                "muscle_groups": "Back;Rear Delts",
                "families": "Row",
                "materials": "",
                "categories": "Upper Body Pull",
            },
        ],
    )

    exit_code = module.main(
        [
            "--input-csv",
            str(csv_path),
            "--output-db",
            str(sqlite_path),
        ]
    )

    assert exit_code == 0
    assert sqlite_path.exists()
    assert list(SQLiteExerciseRepository(sqlite_path).iter_exercises()) == list(
        CsvExerciseRepository(csv_path).iter_exercises()
    )


def test_import_exercises_to_sqlite_main_preserves_real_dataset_json_list_cells(
    tmp_path,
):
    module = load_import_exercises_to_sqlite_module()
    csv_path = tmp_path / "exercises.csv"
    sqlite_path = tmp_path / "exercises.sqlite"
    write_exercise_csv(
        csv_path,
        [
            {
                "exercise_id": "360-pull",
                "name": "360° Pull",
                "description": "A dynamic explosive movement rotating around the bar.",
                "muscle_groups": '["Back", "Shoulders", "Biceps", "Core"]',
                "families": '["Pull-up"]',
                "materials": '["Bar"]',
                "categories": '["Upper Body Pull"]',
            }
        ],
    )

    exit_code = module.main(
        [
            "--input-csv",
            str(csv_path),
            "--output-db",
            str(sqlite_path),
        ]
    )

    assert exit_code == 0
    assert list(SQLiteExerciseRepository(sqlite_path).iter_exercises()) == list(
        CsvExerciseRepository(csv_path).iter_exercises()
    )


def test_import_exercises_to_sqlite_main_creates_expected_schema_and_index(tmp_path):
    module = load_import_exercises_to_sqlite_module()
    csv_path = tmp_path / "exercises.csv"
    sqlite_path = tmp_path / "exercises.sqlite"
    write_exercise_csv(
        csv_path,
        [
            {
                "exercise_id": "pull-up-negative",
                "name": "Pull Up Negative",
                "description": "A controlled eccentric pull-up variation.",
                "muscle_groups": "Back;Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            }
        ],
    )

    exit_code = module.main(
        [
            "--input-csv",
            str(csv_path),
            "--output-db",
            str(sqlite_path),
        ]
    )

    with sqlite3.connect(sqlite_path) as connection:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(exercises)")
        ]
        indexes = [
            row[1] for row in connection.execute("PRAGMA index_list(exercises)")
        ]

    assert exit_code == 0
    assert columns == [
        "id",
        "exercise_id",
        "name",
        "description",
        "muscle_groups",
        "families",
        "materials",
        "categories",
    ]
    assert "idx_exercises_name" in indexes


def test_import_exercises_to_sqlite_main_rejects_duplicate_exercise_ids(tmp_path):
    module = load_import_exercises_to_sqlite_module()
    csv_path = tmp_path / "exercises.csv"
    sqlite_path = tmp_path / "exercises.sqlite"
    write_exercise_csv(
        csv_path,
        [
            {
                "exercise_id": "duplicate-id",
                "name": "Pull Up Negative",
                "description": "A controlled eccentric pull-up variation.",
                "muscle_groups": "Back;Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            },
            {
                "exercise_id": "duplicate-id",
                "name": "Body Row",
                "description": "A horizontal pulling variation.",
                "muscle_groups": "Back;Rear Delts",
                "families": "Row",
                "materials": "",
                "categories": "Upper Body Pull",
            },
        ],
    )

    with pytest.raises(ValueError, match=r"Duplicate exercise_id|row 3|row 2"):
        module.main(
            [
                "--input-csv",
                str(csv_path),
                "--output-db",
                str(sqlite_path),
            ]
        )


REAL_DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "calisthenics_exercises.csv"
)


@pytest.mark.skipif(
    not REAL_DATASET_PATH.exists(),
    reason="Local real dataset file is not available.",
)
def test_import_exercises_to_sqlite_main_smoke_imports_local_real_dataset(tmp_path):
    module = load_import_exercises_to_sqlite_module()
    sqlite_path = tmp_path / "exercises.sqlite"

    exit_code = module.main(
        [
            "--input-csv",
            str(REAL_DATASET_PATH),
            "--output-db",
            str(sqlite_path),
        ]
    )

    exercises = list(SQLiteExerciseRepository(sqlite_path).iter_exercises())

    assert exit_code == 0
    assert len(exercises) == 104
    assert exercises[0].exercise_id == "360-pull"
    assert exercises[0].name.startswith("360")
    assert exercises[0].families == ["Pull-up"]
    assert exercises[0].materials == ["Bar"]
