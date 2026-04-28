import csv
import importlib
import json
from pathlib import Path

import pytest

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    LocalEmbeddedExerciseRepository,
)
from calisthenics_recommender.adapters.sqlite_embedded_exercise_cache import (
    SQLiteEmbeddedExerciseRepository,
    read_sqlite_embedded_exercise_cache_metadata,
)
from calisthenics_recommender.adapters.sqlite_exercise_repository import (
    SQLiteExerciseRepository,
)


def load_build_exercise_cache_module():
    return importlib.import_module(
        "calisthenics_recommender.cli.build_exercise_cache"
    )


def load_wiring_module():
    return importlib.import_module("calisthenics_recommender.wiring")


def load_import_exercises_to_sqlite_module():
    return importlib.import_module(
        "calisthenics_recommender.cli.import_exercises_to_sqlite"
    )


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
            ]
        )


def write_build_config(
    path: Path,
    *,
    raw_backend: str,
    raw_path: Path,
    cache_backend: str,
    cache_path: Path,
    embedding_provider: str = "local-deterministic",
    embedding_model: str = "fake-hash-v1",
    embedding_dimension: int | None = 4,
    text_prefix: str = "",
    text_builder_version: str = "v1",
) -> None:
    raw_path_key = "csv_path" if raw_backend == "csv" else "sqlite_path"
    dimension_line = (
        f"dimension = {embedding_dimension}\n"
        if embedding_dimension is not None
        else ""
    )
    text_prefix_line = f'text_prefix = "{text_prefix}"\n'

    path.write_text(
        (
            "[raw_exercises]\n"
            f'backend = "{raw_backend}"\n'
            f'{raw_path_key} = "{raw_path.name}"\n'
            "\n"
            "[embedded_cache]\n"
            f'backend = "{cache_backend}"\n'
            f'path = "{cache_path.name}"\n'
            "\n"
            "[embedding]\n"
            f'provider = "{embedding_provider}"\n'
            f'model = "{embedding_model}"\n'
            f"{dimension_line}"
            f"{text_prefix_line}"
            f'text_builder_version = "{text_builder_version}"\n'
        ),
        encoding="utf-8",
    )


def import_csv_to_sqlite(csv_path: Path, sqlite_path: Path) -> int:
    module = load_import_exercises_to_sqlite_module()
    return module.main(
        [
            "--input-csv",
            str(csv_path),
            "--output-db",
            str(sqlite_path),
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


def test_build_exercise_cache_main_creates_a_readable_local_cache_from_sqlite(
    tmp_path,
):
    module = load_build_exercise_cache_module()
    csv_path = tmp_path / "exercises.csv"
    sqlite_path = tmp_path / "exercises.sqlite"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_exercise_csv(csv_path)

    import_exit_code = import_csv_to_sqlite(csv_path, sqlite_path)
    exit_code = module.main(
        [
            "--input-db",
            str(sqlite_path),
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

    assert import_exit_code == 0
    assert exit_code == 0
    assert cache_path.exists()
    assert len(list(SQLiteExerciseRepository(sqlite_path).iter_exercises())) == 2

    embedded_exercises = list(
        LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises()
    )

    assert [item.exercise.name for item in embedded_exercises] == [
        "Pull Up Negative",
        "Body Row",
    ]
    assert len(embedded_exercises) == 2
    assert all(len(item.embedding) == 4 for item in embedded_exercises)


def test_build_exercise_cache_main_requires_exactly_one_raw_input_source(tmp_path):
    module = load_build_exercise_cache_module()
    cache_path = tmp_path / "embedded_exercises.jsonl"

    with pytest.raises(SystemExit):
        module.main(
            [
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


def test_build_exercise_cache_main_rejects_multiple_raw_input_sources(tmp_path):
    module = load_build_exercise_cache_module()
    csv_path = tmp_path / "exercises.csv"
    sqlite_path = tmp_path / "exercises.sqlite"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_exercise_csv(csv_path)

    with pytest.raises(SystemExit):
        module.main(
            [
                "--input-csv",
                str(csv_path),
                "--input-db",
                str(sqlite_path),
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
        load_wiring_module(),
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


def test_build_exercise_cache_main_uses_jsonl_config_when_flags_are_omitted(tmp_path):
    module = load_build_exercise_cache_module()
    csv_path = tmp_path / "exercises.csv"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    config_path = tmp_path / "build.toml"
    write_exercise_csv(csv_path)
    write_build_config(
        config_path,
        raw_backend="csv",
        raw_path=csv_path,
        cache_backend="jsonl",
        cache_path=cache_path,
    )

    exit_code = module.main(["--config", str(config_path)])

    assert exit_code == 0
    assert cache_path.exists()
    embedded_exercises = list(
        LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises()
    )
    assert [item.exercise.name for item in embedded_exercises] == [
        "Pull Up Negative",
        "Body Row",
    ]


def test_build_exercise_cache_main_uses_sqlite_config_when_flags_are_omitted(tmp_path):
    module = load_build_exercise_cache_module()
    csv_path = tmp_path / "exercises.csv"
    sqlite_cache_path = tmp_path / "embedded_exercises.sqlite"
    config_path = tmp_path / "build.toml"
    write_exercise_csv(csv_path)
    write_build_config(
        config_path,
        raw_backend="csv",
        raw_path=csv_path,
        cache_backend="sqlite",
        cache_path=sqlite_cache_path,
    )

    exit_code = module.main(["--config", str(config_path)])

    assert exit_code == 0
    metadata = read_sqlite_embedded_exercise_cache_metadata(sqlite_cache_path)
    assert metadata.embedding_model == "fake-hash-v1"
    assert metadata.embedding_dimension == 4
    assert metadata.text_builder_version == "v1"
    embedded_exercises = list(
        SQLiteEmbeddedExerciseRepository(sqlite_cache_path).iter_embedded_exercises()
    )
    assert [item.exercise.name for item in embedded_exercises] == [
        "Pull Up Negative",
        "Body Row",
    ]


def test_build_exercise_cache_main_explicit_flags_override_config_values(tmp_path):
    module = load_build_exercise_cache_module()
    csv_path = tmp_path / "exercises.csv"
    sqlite_path = tmp_path / "exercises.sqlite"
    configured_cache_path = tmp_path / "configured.jsonl"
    explicit_cache_path = tmp_path / "explicit.jsonl"
    config_path = tmp_path / "build.toml"
    write_exercise_csv(csv_path)
    import_exit_code = import_csv_to_sqlite(csv_path, sqlite_path)
    write_build_config(
        config_path,
        raw_backend="csv",
        raw_path=tmp_path / "missing.csv",
        cache_backend="jsonl",
        cache_path=configured_cache_path,
        embedding_model="configured-model",
        embedding_dimension=8,
        text_prefix="configured: ",
        text_builder_version="v1",
    )

    exit_code = module.main(
        [
            "--config",
            str(config_path),
            "--input-db",
            str(sqlite_path),
            "--output-cache",
            str(explicit_cache_path),
            "--embedding-model",
            "fake-hash-v1",
            "--embedding-dimension",
            "4",
            "--text-prefix",
            "",
            "--text-builder-version",
            "v1",
        ]
    )

    assert import_exit_code == 0
    assert exit_code == 0
    assert explicit_cache_path.exists()
    assert not configured_cache_path.exists()
    first_line = explicit_cache_path.read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(first_line) == {
        "type": "metadata",
        "embedding_model": "fake-hash-v1",
        "embedding_dimension": 4,
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
