from importlib import import_module
import json
import logging
import math
from pathlib import Path

import pytest


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_model():
    module = import_module("calisthenics_recommender.domain.embedded_exercise")
    return getattr(module, "EmbeddedExercise")


def get_embedded_exercise_repository_protocol():
    module = import_module(
        "calisthenics_recommender.ports.embedded_exercise_repository"
    )
    return getattr(module, "EmbeddedExerciseRepository")


def get_cache_metadata_model():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "EmbeddedExerciseCacheMetadata")


def get_local_embedded_exercise_cache():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "LocalEmbeddedExerciseCache")


def get_local_embedded_exercise_repository():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "LocalEmbeddedExerciseRepository")


def exercise_id_for(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def build_embedded_exercise(
    name: str, description: str, materials: list[str], embedding: list[float]
):
    Exercise = get_exercise_model()
    EmbeddedExercise = get_embedded_exercise_model()
    exercise = Exercise(
        exercise_id=exercise_id_for(name),
        name=name,
        description=description,
        muscle_groups=["Back", "Biceps"],
        families=["Pull-up"],
        materials=materials,
        categories=["Upper Body Pull"],
    )
    return EmbeddedExercise(exercise=exercise, embedding=embedding)


def build_metadata():
    EmbeddedExerciseCacheMetadata = get_cache_metadata_model()
    return EmbeddedExerciseCacheMetadata(
        embedding_model="test-model",
        embedding_dimension=3,
        text_builder_version="v1",
    )


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return path


def metadata_record(**overrides):
    record = {
        "type": "metadata",
        "embedding_model": "test-model",
        "embedding_dimension": 3,
        "text_builder_version": "v1",
    }
    record.update(overrides)
    return record


def embedded_exercise_record(
    *,
    name: str = "Pull Up Negative",
    description: str = "A controlled eccentric pull-up variation for building strength.",
    materials: list[str] | None = None,
    embedding: list[float] | object = None,
):
    if materials is None:
        materials = ["Bar"]
    if embedding is None:
        embedding = [1.0, 0.0, 0.0]

    return {
        "type": "embedded_exercise",
        "exercise": {
            "exercise_id": exercise_id_for(name),
            "name": name,
            "description": description,
            "muscle_groups": ["Back", "Biceps"],
            "families": ["Pull-up"],
            "materials": materials,
            "categories": ["Upper Body Pull"],
        },
        "embedding": embedding,
    }


def test_local_embedded_exercise_repository_implements_protocol(tmp_path):
    EmbeddedExerciseRepository = get_embedded_exercise_repository_protocol()
    LocalEmbeddedExerciseCache = get_local_embedded_exercise_cache()
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.jsonl"
    cache = LocalEmbeddedExerciseCache(cache_path)
    cache.write_embedded_exercises(
        [build_embedded_exercise("Pull Up Negative", "Controlled eccentric pulling.", ["Bar"], [1.0, 0.0, 0.0])],
        build_metadata(),
    )

    repository = LocalEmbeddedExerciseRepository(str(cache_path))

    assert isinstance(repository, EmbeddedExerciseRepository)
    assert not isinstance(repository.iter_embedded_exercises(), list)


def test_local_embedded_exercise_cache_writes_and_repository_reads_embedded_exercises(
    tmp_path,
):
    LocalEmbeddedExerciseCache = get_local_embedded_exercise_cache()
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.jsonl"
    embedded_exercises = [
        build_embedded_exercise(
            "Pull Up Negative",
            "A controlled eccentric pull-up variation.",
            ["Bar"],
            [1.0, 0.0, 0.0],
        ),
        build_embedded_exercise(
            "Body Row",
            "A horizontal pulling variation.",
            ["Bar", "Rings"],
            [0.9, 0.1, 0.0],
        ),
    ]

    LocalEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        embedded_exercises,
        build_metadata(),
    )
    loaded_embedded_exercises = list(
        LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises()
    )

    assert loaded_embedded_exercises == embedded_exercises
    assert loaded_embedded_exercises[0].embedding == (1.0, 0.0, 0.0)
    assert isinstance(loaded_embedded_exercises[0].embedding, tuple)
    assert [item.exercise.name for item in loaded_embedded_exercises] == [
        "Pull Up Negative",
        "Body Row",
    ]
    assert [item.exercise.exercise_id for item in loaded_embedded_exercises] == [
        "pull-up-negative",
        "body-row",
    ]


def test_local_embedded_exercise_cache_serializes_exercise_id(tmp_path):
    LocalEmbeddedExerciseCache = get_local_embedded_exercise_cache()
    cache_path = tmp_path / "embedded_exercises.jsonl"

    LocalEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        [
            build_embedded_exercise(
                "Pull Up Negative",
                "A controlled eccentric pull-up variation.",
                ["Bar"],
                [1.0, 0.0, 0.0],
            )
        ],
        build_metadata(),
    )

    record = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[1])

    assert record["exercise"]["exercise_id"] == "pull-up-negative"


def test_local_embedded_exercise_repository_rejects_old_records_without_exercise_id(
    tmp_path,
):
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    record = embedded_exercise_record()
    del record["exercise"]["exercise_id"]
    cache_path = write_jsonl(
        tmp_path / "embedded_exercises.jsonl",
        [metadata_record(), record],
    )

    with pytest.raises(ValueError, match=r"line 2|exercise_id"):
        list(LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_local_embedded_exercise_repository_streams_and_logs_only_safe_messages(
    tmp_path, caplog
):
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = write_jsonl(
        tmp_path / "embedded_exercises.jsonl",
        [
            metadata_record(),
            embedded_exercise_record(),
        ],
    )
    repository = LocalEmbeddedExerciseRepository(cache_path)

    with caplog.at_level(logging.INFO):
        iterator = iter(repository.iter_embedded_exercises())
        first_embedded_exercise = next(iterator)

        assert first_embedded_exercise.exercise.name == "Pull Up Negative"
        assert "Starting embedded exercise cache scan" in caplog.text
        assert str(cache_path) in caplog.text
        assert "Finished embedded exercise cache scan" not in caplog.text

        remaining_embedded_exercises = list(iterator)

    assert remaining_embedded_exercises == []
    assert "Finished embedded exercise cache scan" in caplog.text
    assert "with 1 embedded exercises" in caplog.text
    assert "A controlled eccentric pull-up variation for building strength." not in caplog.text
    assert "[1.0, 0.0, 0.0]" not in caplog.text


def test_local_embedded_exercise_repository_yields_first_record_before_later_error(
    tmp_path,
):
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = write_jsonl(
        tmp_path / "embedded_exercises.jsonl",
        [
            metadata_record(),
            embedded_exercise_record(name="Pull Up Negative"),
            embedded_exercise_record(name="Body Row", embedding="not-a-vector"),
        ],
    )
    repository = LocalEmbeddedExerciseRepository(cache_path)
    iterator = iter(repository.iter_embedded_exercises())

    first_embedded_exercise = next(iterator)

    assert first_embedded_exercise.exercise.name == "Pull Up Negative"

    with pytest.raises(ValueError, match=r"line 3|embedding"):
        next(iterator)


def test_local_embedded_exercise_cache_writes_metadata_header(tmp_path):
    LocalEmbeddedExerciseCache = get_local_embedded_exercise_cache()
    cache_path = tmp_path / "embedded_exercises.jsonl"

    LocalEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        [build_embedded_exercise("Pull Up Negative", "Controlled eccentric pulling.", ["Bar"], [1.0, 0.0, 0.0])],
        build_metadata(),
    )

    first_line = cache_path.read_text(encoding="utf-8").splitlines()[0]

    assert json.loads(first_line) == metadata_record()


def test_local_embedded_exercise_repository_raises_for_missing_metadata_header(
    tmp_path,
):
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = write_jsonl(
        tmp_path / "embedded_exercises.jsonl",
        [embedded_exercise_record()],
    )

    with pytest.raises(ValueError, match=r"metadata|line 1"):
        list(LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_local_embedded_exercise_repository_raises_for_invalid_metadata(
    tmp_path,
):
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = write_jsonl(
        tmp_path / "embedded_exercises.jsonl",
        [metadata_record(embedding_dimension=0)],
    )

    with pytest.raises(ValueError, match=r"metadata|embedding_dimension|line 1"):
        list(LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_local_embedded_exercise_cache_rejects_invalid_metadata():
    EmbeddedExerciseCacheMetadata = get_cache_metadata_model()

    with pytest.raises(ValueError, match="embedding_dimension"):
        EmbeddedExerciseCacheMetadata(
            embedding_model="test-model",
            embedding_dimension=0,
            text_builder_version="v1",
        )


def test_local_embedded_exercise_repository_raises_for_embedding_dimension_mismatch(
    tmp_path,
):
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = write_jsonl(
        tmp_path / "embedded_exercises.jsonl",
        [
            metadata_record(embedding_dimension=3),
            embedded_exercise_record(embedding=[1.0, 0.0]),
        ],
    )

    with pytest.raises(ValueError, match=r"line 2|dimension"):
        list(LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


@pytest.mark.parametrize("non_finite_value", [math.nan, math.inf])
def test_local_embedded_exercise_repository_raises_for_non_finite_embedding_values(
    tmp_path, non_finite_value
):
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = write_jsonl(
        tmp_path / "embedded_exercises.jsonl",
        [
            metadata_record(),
            embedded_exercise_record(embedding=[1.0, non_finite_value, 0.0]),
        ],
    )

    with pytest.raises(ValueError, match=r"line 2|embedding"):
        list(LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


@pytest.mark.parametrize("non_finite_value", [math.nan, math.inf])
def test_local_embedded_exercise_cache_rejects_non_finite_embedding_values_on_write(
    tmp_path, non_finite_value
):
    LocalEmbeddedExerciseCache = get_local_embedded_exercise_cache()
    cache_path = tmp_path / "embedded_exercises.jsonl"

    with pytest.raises(ValueError, match=r"line 2|embedding"):
        LocalEmbeddedExerciseCache(cache_path).write_embedded_exercises(
            [
                build_embedded_exercise(
                    "Pull Up Negative",
                    "Controlled eccentric pulling.",
                    ["Bar"],
                    [1.0, non_finite_value, 0.0],
                )
            ],
            build_metadata(),
        )


@pytest.mark.parametrize(
    ("records", "expected_match"),
    [
        ([metadata_record(), "{not json"], r"line 2|JSON"),
        (
            [metadata_record(), {"type": "something_else"}],
            r"line 2|Unknown record type",
        ),
        (
            [
                metadata_record(),
                embedded_exercise_record(),
                {
                    "type": "embedded_exercise",
                    "exercise": {
                        "exercise_id": "invalid-exercise",
                        "name": "   ",
                        "description": "Invalid exercise.",
                        "muscle_groups": ["Back"],
                        "families": ["Pull-up"],
                        "materials": ["Bar"],
                        "categories": ["Upper Body Pull"],
                    },
                    "embedding": [1.0, 0.0, 0.0],
                },
            ],
            r"line 3|exercise|name",
        ),
        (
            [
                metadata_record(),
                {
                    "type": "embedded_exercise",
                    "exercise": embedded_exercise_record()["exercise"],
                    "embedding": ["bad", 0.0, 0.0],
                },
            ],
            r"line 2|embedding",
        ),
    ],
)
def test_local_embedded_exercise_repository_raises_clear_errors_for_invalid_records(
    tmp_path, records, expected_match
):
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.jsonl"
    rendered_lines = [
        record if isinstance(record, str) else json.dumps(record) for record in records
    ]
    cache_path.write_text("\n".join(rendered_lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=expected_match):
        list(LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())


def test_local_embedded_exercise_repository_raises_for_empty_cache_file(tmp_path):
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    cache_path = tmp_path / "embedded_exercises.jsonl"
    cache_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty|Empty"):
        list(LocalEmbeddedExerciseRepository(cache_path).iter_embedded_exercises())
