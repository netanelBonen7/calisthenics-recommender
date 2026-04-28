import csv
import importlib
from pathlib import Path

from calisthenics_recommender.adapters.local_deterministic_embedding_provider import (
    LocalDeterministicEmbeddingProvider,
)
from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
    read_embedded_exercise_cache_metadata,
)
from calisthenics_recommender.adapters.sqlite_embedded_exercise_cache import (
    SQLiteEmbeddedExerciseCache,
)
from calisthenics_recommender.application.exercise_text_builder import (
    build_exercise_text,
)
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise


def load_build_exercise_cache_module():
    return importlib.import_module(
        "calisthenics_recommender.cli.build_exercise_cache"
    )


def load_wiring_module():
    return importlib.import_module("calisthenics_recommender.wiring")


def load_demo_recommend_module():
    return importlib.import_module("calisthenics_recommender.cli.demo_recommend")


def write_exercise_csv(path: Path) -> None:
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
        writer.writerows(
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
                    "description": "A horizontal pulling variation with a bar.",
                    "muscle_groups": "Back;Biceps",
                    "families": "Pull-up",
                    "materials": "Bar",
                    "categories": "Upper Body Pull",
                },
                {
                    "exercise_id": "ring-pull-up",
                    "name": "Ring Pull Up",
                    "description": "A pull-up variation that requires rings.",
                    "muscle_groups": "Back;Biceps",
                    "families": "Pull-up",
                    "materials": "Rings",
                    "categories": "Upper Body Pull",
                },
            ]
        )


def make_exercises() -> list[Exercise]:
    return [
        Exercise(
            exercise_id="pull-up-negative",
            name="Pull Up Negative",
            description="A controlled eccentric pull-up variation.",
            muscle_groups=["Back", "Biceps"],
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
        ),
        Exercise(
            exercise_id="body-row",
            name="Body Row",
            description="A horizontal pulling variation with a bar.",
            muscle_groups=["Back", "Biceps"],
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
        ),
        Exercise(
            exercise_id="ring-pull-up",
            name="Ring Pull Up",
            description="A pull-up variation that requires rings.",
            muscle_groups=["Back", "Biceps"],
            families=["Pull-up"],
            materials=["Rings"],
            categories=["Upper Body Pull"],
        ),
    ]


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


def write_sqlite_cache(cache_path: Path) -> None:
    embedding_provider = LocalDeterministicEmbeddingProvider(dimension=4)
    embedded_exercises = [
        EmbeddedExercise(
            exercise=exercise,
            embedding=tuple(embedding_provider.embed(build_exercise_text(exercise))),
        )
        for exercise in make_exercises()
    ]
    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        embedded_exercises,
        EmbeddedExerciseCacheMetadata(
            embedding_model="fake-hash-v1",
            embedding_dimension=4,
            text_builder_version="v1",
        ),
    )


def write_demo_config(path: Path, *, cache_backend: str, cache_path: Path) -> None:
    path.write_text(
        (
            "[embedded_cache]\n"
            f'backend = "{cache_backend}"\n'
            f'path = "{cache_path.name}"\n'
            "\n"
            "[embedding]\n"
            'provider = "local-deterministic"\n'
            'model = "fake-hash-v1"\n'
            'query_prefix = ""\n'
        ),
        encoding="utf-8",
    )


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


def test_demo_recommend_main_supports_sentence_transformer_mode_without_real_model(
    tmp_path, monkeypatch, capsys
):
    csv_path = tmp_path / "exercises.csv"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_exercise_csv(csv_path)
    build_module = load_build_exercise_cache_module()
    demo_module = load_demo_recommend_module()
    build_init_calls: list[tuple[str, str, bool]] = []
    demo_init_calls: list[tuple[str, str, bool]] = []

    class FakeSentenceTransformerEmbeddingProvider:
        def __init__(
            self,
            model_name: str = "unused",
            text_prefix: str = "",
            normalize_embeddings: bool = True,
            model=None,
        ) -> None:
            init_calls = build_init_calls if text_prefix == "passage: " else demo_init_calls
            init_calls.append((model_name, text_prefix, normalize_embeddings))
            self._text_prefix = text_prefix

        def get_embedding_dimension(self) -> int:
            return 3

        def embed(self, text: str) -> list[float]:
            if "Ring Pull Up" in text or "Rings" in text:
                return [0.0, 1.0, 0.0]
            if "Body Row" in text:
                return [0.8, 0.2, 0.0]
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        load_wiring_module(),
        "SentenceTransformerEmbeddingProvider",
        FakeSentenceTransformerEmbeddingProvider,
    )

    build_exit_code = build_module.main(
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
    demo_exit_code = demo_module.main(
        [
            "--cache-path",
            str(cache_path),
            "--embedding-provider",
            "sentence-transformer",
            "--embedding-model",
            "custom/local-model",
            "--target-family",
            "Pull-up",
            "--goal",
            "I want to build pulling strength and improve pull-ups.",
            "--current-level",
            "I can do a few strict pull-ups.",
            "--available-equipment",
            "Bar",
            "--query-prefix",
            "query: ",
            "--limit",
            "3",
        ]
    )
    output = capsys.readouterr().out

    assert build_exit_code == 0
    assert demo_exit_code == 0
    assert build_init_calls == [("custom/local-model", "passage: ", True)]
    assert demo_init_calls == [("custom/local-model", "query: ", True)]
    assert "Pull Up Negative" in output
    assert "Ring Pull Up" not in output
    assert "Match score:" in output


def test_demo_recommend_main_supports_jsonl_config(tmp_path, capsys):
    csv_path = tmp_path / "exercises.csv"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    config_path = tmp_path / "demo.toml"
    write_exercise_csv(csv_path)
    build_cache(csv_path, cache_path)
    write_demo_config(
        config_path,
        cache_backend="jsonl",
        cache_path=cache_path,
    )
    demo_module = load_demo_recommend_module()

    exit_code = demo_module.main(
        [
            "--config",
            str(config_path),
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


def test_demo_recommend_main_supports_sqlite_config(tmp_path, capsys):
    cache_path = tmp_path / "embedded_exercises.sqlite"
    config_path = tmp_path / "demo.toml"
    write_sqlite_cache(cache_path)
    write_demo_config(
        config_path,
        cache_backend="sqlite",
        cache_path=cache_path,
    )
    demo_module = load_demo_recommend_module()

    exit_code = demo_module.main(
        [
            "--config",
            str(config_path),
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
