import csv
import importlib.util
from pathlib import Path

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    read_embedded_exercise_cache_metadata,
)


def load_build_exercise_cache_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "build_exercise_cache.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_exercise_cache_script", script_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_demo_recommend_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "demo_recommend.py"
    spec = importlib.util.spec_from_file_location("demo_recommend_script", script_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_exercise_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "name",
                "description",
                "muscle_groups",
                "families",
                "materials",
                "categories",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "name": "Pull Up Negative",
                    "description": "A controlled eccentric pull-up variation.",
                    "muscle_groups": "Back;Biceps",
                    "families": "Pull-up",
                    "materials": "Bar",
                    "categories": "Upper Body Pull",
                },
                {
                    "name": "Body Row",
                    "description": "A horizontal pulling variation with a bar.",
                    "muscle_groups": "Back;Biceps",
                    "families": "Pull-up",
                    "materials": "Bar",
                    "categories": "Upper Body Pull",
                },
                {
                    "name": "Ring Pull Up",
                    "description": "A pull-up variation that requires rings.",
                    "muscle_groups": "Back;Biceps",
                    "families": "Pull-up",
                    "materials": "Rings",
                    "categories": "Upper Body Pull",
                },
            ]
        )


def build_cache(csv_path: Path, cache_path: Path) -> None:
    build_module = load_build_exercise_cache_module()
    exit_code = build_module.main(
        [
            "--input-csv",
            str(csv_path),
            "--output-cache",
            str(cache_path),
            "--embedding-model",
            "fake-hash-v1",
            "--embedding-dimension",
            "4",
            "--text-builder-version",
            "v1",
        ]
    )

    assert exit_code == 0


def test_demo_recommend_main_prints_human_readable_recommendations_from_existing_cache(
    tmp_path, capsys
):
    csv_path = tmp_path / "exercises.csv"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_exercise_csv(csv_path)
    build_cache(csv_path, cache_path)
    demo_module = load_demo_recommend_module()

    exit_code = demo_module.main(
        [
            "--cache-path",
            str(cache_path),
            "--target-family",
            "Pull-up",
            "--goal",
            "I want to build pulling strength and improve pull-ups.",
            "--current-level",
            "I can do a few strict pull-ups.",
            "--available-equipment",
            "Bar",
            "--limit",
            "3",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Pull Up Negative" in output
    assert "Ring Pull Up" not in output
    assert "Match score:" in output
    assert "Reason:" in output
    assert "Required equipment:" in output
    assert "Categories:" in output
    assert "Families:" in output
    assert "matched your Pull-up target family through retrieval" in output
    assert "Recommendation(" not in output
    assert "CategoryFamily(" not in output


def test_read_embedded_exercise_cache_metadata_reads_metadata_from_script_built_cache(
    tmp_path,
):
    csv_path = tmp_path / "exercises.csv"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_exercise_csv(csv_path)
    build_cache(csv_path, cache_path)

    metadata = read_embedded_exercise_cache_metadata(cache_path)

    assert metadata.embedding_model == "fake-hash-v1"
    assert metadata.embedding_dimension == 4
    assert metadata.text_builder_version == "v1"
