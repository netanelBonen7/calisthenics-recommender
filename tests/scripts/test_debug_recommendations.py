import csv
import importlib.util
from pathlib import Path

import pytest

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
    LocalEmbeddedExerciseCache,
)
from calisthenics_recommender.application.exercise_text_builder import (
    build_exercise_text,
)
from calisthenics_recommender.application.query_builder import build_query_text
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise
from calisthenics_recommender.domain.user_request import UserRequest


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


def load_debug_recommendations_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "debug_recommendations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "debug_recommendations_script", script_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_exercise_csv(path: Path, rows: list[dict[str, str]]) -> None:
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
        writer.writerows(rows)


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


def make_user_request() -> UserRequest:
    return UserRequest(
        target_family="Pull-up",
        goal="I want to build pulling strength and improve my strict pull-ups.",
        current_level="I can do a few strict pull-ups but my last reps are slow.",
        available_equipment=["Bar"],
    )


def make_user_request_args() -> list[str]:
    return [
        "--target-family",
        "Pull-up",
        "--goal",
        "I want to build pulling strength and improve my strict pull-ups.",
        "--current-level",
        "I can do a few strict pull-ups but my last reps are slow.",
        "--available-equipment",
        "Bar",
    ]


def make_embedded_exercise(
    *,
    name: str,
    description: str,
    families: list[str],
    materials: list[str],
    categories: list[str],
    embedding: list[float],
) -> EmbeddedExercise:
    exercise = Exercise(
        name=name,
        description=description,
        muscle_groups=["Back", "Biceps"],
        families=families,
        materials=materials,
        categories=categories,
    )
    return EmbeddedExercise(exercise=exercise, embedding=embedding)


def test_debug_recommendations_main_prints_query_text_duplicate_matches_and_missing_name(
    tmp_path, capsys
):
    module = load_debug_recommendations_module()
    csv_path = tmp_path / "exercises.csv"
    first_pull_up = Exercise(
        name="Pull Up",
        description="A strict vertical pulling movement on a bar.",
        muscle_groups=["Back", "Biceps"],
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    second_pull_up = Exercise(
        name="Pull Up",
        description="A pull-up variation with a paused top position.",
        muscle_groups=["Back", "Biceps"],
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    write_exercise_csv(
        csv_path,
        [
            {
                "name": first_pull_up.name,
                "description": first_pull_up.description,
                "muscle_groups": "Back;Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            },
            {
                "name": second_pull_up.name,
                "description": second_pull_up.description,
                "muscle_groups": "Back;Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            },
        ],
    )

    exit_code = module.main(
        [
            "--input-csv",
            str(csv_path),
            "--exercise-name",
            "Pull Up",
            "--exercise-name",
            "Missing Move",
            *make_user_request_args(),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "=== QUERY TEXT ===" in output
    assert build_query_text(make_user_request()) in output
    assert "--- Pull Up (match 1) ---" in output
    assert "--- Pull Up (match 2) ---" in output
    assert build_exercise_text(first_pull_up) in output
    assert build_exercise_text(second_pull_up) in output
    assert output.index(first_pull_up.description) < output.index(
        second_pull_up.description
    )
    assert "--- Missing Move ---" in output
    assert "Exercise not found." in output


def test_debug_recommendations_main_prints_filtered_candidates_from_local_deterministic_cache(
    tmp_path, capsys
):
    module = load_debug_recommendations_module()
    csv_path = tmp_path / "exercises.csv"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_exercise_csv(
        csv_path,
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
                "description": "A pulling variation that requires rings.",
                "muscle_groups": "Back;Biceps",
                "families": "Pull-up",
                "materials": "Rings",
                "categories": "Upper Body Pull",
            },
        ],
    )
    build_cache(csv_path, cache_path)

    exit_code = module.main(
        [
            "--cache-path",
            str(cache_path),
            *make_user_request_args(),
            "--limit",
            "3",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "=== QUERY TEXT ===" in output
    assert build_query_text(make_user_request()) in output
    assert "=== TOP CANDIDATES ===" in output
    assert "1. Exercise:" in output
    assert "Score:" in output
    assert "Exercise text:" in output
    assert "Pull Up Negative" in output
    assert "Body Row" in output
    assert "Ring Pull Up" not in output


def test_debug_recommendations_main_supports_sentence_transformer_cache_debug_without_real_model(
    tmp_path, monkeypatch, capsys
):
    module = load_debug_recommendations_module()
    cache_path = tmp_path / "embedded_exercises.jsonl"
    init_calls: list[tuple[str, str, bool]] = []

    class FakeSentenceTransformerEmbeddingProvider:
        def __init__(
            self,
            model_name: str = "unused",
            text_prefix: str = "",
            normalize_embeddings: bool = True,
            model=None,
        ) -> None:
            init_calls.append((model_name, text_prefix, normalize_embeddings))

        def get_embedding_dimension(self) -> int:
            return 3

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        module,
        "SentenceTransformerEmbeddingProvider",
        FakeSentenceTransformerEmbeddingProvider,
    )
    LocalEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        [
            make_embedded_exercise(
                name="Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[1.0, 0.0, 0.0],
            ),
            make_embedded_exercise(
                name="Body Row",
                description="A horizontal pull that builds pulling volume.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.8, 0.2, 0.0],
            ),
            make_embedded_exercise(
                name="Ring Pull Up",
                description="A pulling variation that requires rings.",
                families=["Pull-up"],
                materials=["Rings"],
                categories=["Upper Body Pull"],
                embedding=[1.0, 0.0, 0.0],
            ),
        ],
        EmbeddedExerciseCacheMetadata(
            embedding_model="custom/local-model",
            embedding_dimension=3,
            text_builder_version="v1",
        ),
    )

    exit_code = module.main(
        [
            "--cache-path",
            str(cache_path),
            "--embedding-provider",
            "sentence-transformer",
            "--embedding-model",
            "custom/local-model",
            "--query-prefix",
            "query: ",
            *make_user_request_args(),
            "--limit",
            "2",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert init_calls == [("custom/local-model", "query: ", True)]
    assert "1. Exercise: Pull Up" in output
    assert "2. Exercise: Body Row" in output
    assert output.index("1. Exercise: Pull Up") < output.index(
        "2. Exercise: Body Row"
    )
    assert "Ring Pull Up" not in output


def test_debug_recommendations_main_raises_for_incomplete_cache_request_arguments(
    tmp_path,
):
    module = load_debug_recommendations_module()
    cache_path = tmp_path / "embedded_exercises.jsonl"

    with pytest.raises(
        ValueError,
        match="--cache-path requires --target-family, --goal, --current-level, and at least one --available-equipment",
    ):
        module.main(
            [
                "--cache-path",
                str(cache_path),
                "--target-family",
                "Pull-up",
            ]
        )
