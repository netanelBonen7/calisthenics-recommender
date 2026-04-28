import importlib
import sqlite3

import pytest

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
)
from calisthenics_recommender.adapters.sqlite_embedded_exercise_cache import (
    SQLiteEmbeddedExerciseCache,
    SQLiteEmbeddedExerciseRepository,
)
from calisthenics_recommender.adapters.sqlite_exercise_repository import (
    write_exercises_to_sqlite,
)
from calisthenics_recommender.domain.exercise import Exercise


def load_module():
    return importlib.import_module(
        "calisthenics_recommender.cli.process_pending_embedding_updates"
    )


def make_exercise(**overrides):
    payload = {
        "exercise_id": "pull-up",
        "name": "Pull Up",
        "description": "A strict vertical pulling movement.",
        "muscle_groups": ["Back", "Biceps"],
        "families": ["Pull-up"],
        "materials": ["Bar"],
        "categories": ["Upper Body Pull"],
    }
    payload.update(overrides)
    return Exercise(**payload)


def write_config(
    config_path,
    *,
    raw_path,
    cache_path,
    raw_backend="sqlite",
    embedded_backend="sqlite",
):
    if raw_backend == "sqlite":
        raw_path_field = f'sqlite_path = "{raw_path.as_posix()}"'
    else:
        raw_path_field = f'csv_path = "{raw_path.as_posix()}"'

    config_path.write_text(
        "\n".join(
            [
                "[raw_exercises]",
                f'backend = "{raw_backend}"',
                raw_path_field,
                "",
                "[embedded_cache]",
                f'backend = "{embedded_backend}"',
                f'path = "{cache_path.as_posix()}"',
                "",
                "[embedding]",
                'provider = "local-deterministic"',
                'model = "test-model"',
                "dimension = 3",
                'text_builder_version = "v1"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def pending_count(sqlite_path):
    with sqlite3.connect(sqlite_path) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM pending_embedding_updates"
        ).fetchone()[0]


def test_process_pending_embedding_updates_main_processes_sqlite_pending_upsert(
    tmp_path,
    capsys,
):
    module = load_module()
    raw_path = tmp_path / "exercises.sqlite"
    cache_path = tmp_path / "embedded.sqlite"
    config_path = tmp_path / "config.toml"
    exercise = make_exercise()
    write_exercises_to_sqlite(raw_path, [exercise])
    SQLiteEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        [],
        EmbeddedExerciseCacheMetadata(
            embedding_model="test-model",
            embedding_dimension=3,
            text_builder_version="v1",
        ),
    )
    write_config(
        config_path,
        raw_path=raw_path,
        cache_path=cache_path,
    )

    exit_code = module.main(["--config", str(config_path)])

    loaded = list(SQLiteEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())
    assert exit_code == 0
    assert pending_count(raw_path) == 0
    assert [item.exercise for item in loaded] == [exercise]
    assert "processed=1" in capsys.readouterr().out


def test_process_pending_embedding_updates_main_rejects_non_sqlite_cache(tmp_path):
    module = load_module()
    raw_path = tmp_path / "exercises.sqlite"
    cache_path = tmp_path / "embedded.jsonl"
    config_path = tmp_path / "config.toml"
    write_exercises_to_sqlite(raw_path, [])
    write_config(
        config_path,
        raw_path=raw_path,
        cache_path=cache_path,
        embedded_backend="jsonl",
    )

    with pytest.raises(ValueError, match="SQLite embedded cache"):
        module.main(["--config", str(config_path)])
