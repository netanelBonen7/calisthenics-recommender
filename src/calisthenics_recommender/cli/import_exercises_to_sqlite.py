from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.adapters.csv_exercise_repository import (
    CsvExerciseRepository,
)
from calisthenics_recommender.adapters.sqlite_exercise_repository import (
    write_exercises_to_sqlite,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import raw exercises from a CSV file into a local SQLite database."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-db", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    exercise_repository = CsvExerciseRepository(args.input_csv)

    write_exercises_to_sqlite(
        sqlite_path=args.output_db,
        exercises=exercise_repository.iter_exercises(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
