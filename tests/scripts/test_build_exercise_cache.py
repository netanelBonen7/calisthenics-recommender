import csv
import importlib
import json
from pathlib import Path

import pytest

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    LocalEmbeddedExerciseRepository,
)


def load_build_exercise_cache_module():
    return importlib.import_module(
        "calisthenics_recommender.cli.build_exercise_cache"
    )


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
            ]
        )


def test_build_exercise_cache_main_creates_a_readable_local_cache(tmp_path):
    module = load_build_exercise_cache_module()
    csv_path = tmp_path / "exercises.csv"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_exercise_csv(csv_path)

    exit_code = module.main(
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
    assert cache_path.exists()

    first_line = cache_path.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first_line) == {
        "type": "metadata",
        "embedding_model": "fake-hash-v1",
        "embedding_dimension": 4,
        "text_builder_version": "v1",
    }

    embedded_exercises = list(
        LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises()
    )

    assert [item.exercise.name for item in embedded_exercises] == [
        "Pull Up Negative",
        "Body Row",
    ]
    assert len(embedded_exercises) == 2
    assert all(len(item.embedding) == 4 for item in embedded_exercises)
    assert embedded_exercises[0].embedding != embedded_exercises[1].embedding


def test_build_exercise_cache_main_supports_sentence_transformer_mode_without_real_model(
    tmp_path, monkeypatch
):
    module = load_build_exercise_cache_module()
    csv_path = tmp_path / "exercises.csv"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_exercise_csv(csv_path)
    init_calls: list[tuple[str, str, bool]] = []
    embed_calls: list[str] = []

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
            embed_calls.append(text)
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        module,
        "SentenceTransformerEmbeddingProvider",
        FakeSentenceTransformerEmbeddingProvider,
    )

    exit_code = module.main(
        [
            "--input-csv",
            str(csv_path),
            "--output-cache",
            str(cache_path),
            "--embedding-provider",
            "sentence-transformer",
            "--embedding-model",
            "custom/local-model",
            "--text-prefix",
            "passage: ",
            "--text-builder-version",
            "v1",
        ]
    )

    assert exit_code == 0
    assert init_calls == [("custom/local-model", "passage: ", True)]
    assert len(embed_calls) == 2
    assert json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0]) == {
        "type": "metadata",
        "embedding_model": "custom/local-model",
        "embedding_dimension": 3,
        "text_builder_version": "v1",
    }


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
def test_build_exercise_cache_main_smoke_builds_cache_from_local_real_dataset(tmp_path):
    module = load_build_exercise_cache_module()
    cache_path = tmp_path / "embedded_exercises.jsonl"

    exit_code = module.main(
        [
            "--input-csv",
            str(REAL_DATASET_PATH),
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
    assert cache_path.exists()

    embedded_exercises = list(
        LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises()
    )

    assert len(embedded_exercises) == 104

    exercises_by_name = {
        item.exercise.name: item.exercise for item in embedded_exercises
    }

    assert exercises_by_name["360° Pull"].muscle_groups == [
        "Back",
        "Shoulders",
        "Biceps",
        "Core",
    ]
    assert exercises_by_name["360° Pull"].families == ["Pull-up"]
    assert exercises_by_name["360° Pull"].materials == ["Bar"]
    assert exercises_by_name["360° Pull"].categories == ["Upper Body Pull"]

    assert exercises_by_name["Adv Tuck Flag"].muscle_groups == [
        "Obliques",
        "Core",
        "Shoulders",
    ]
    assert exercises_by_name["Adv Tuck Flag"].families == ["Human Flag"]
    assert exercises_by_name["Adv Tuck Flag"].materials == ["Vertical Bar"]
    assert exercises_by_name["Adv Tuck Flag"].categories == ["Core"]
