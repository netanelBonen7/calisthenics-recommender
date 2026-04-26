from importlib import import_module
import csv
import inspect
from pathlib import Path
import socket

import pytest


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_model():
    module = import_module("calisthenics_recommender.domain.embedded_exercise")
    return getattr(module, "EmbeddedExercise")


def get_build_exercise_text():
    module = import_module("calisthenics_recommender.application.exercise_text_builder")
    return getattr(module, "build_exercise_text")


def get_build_embedded_exercise_cache():
    module = import_module(
        "calisthenics_recommender.application.embedded_exercise_cache_workflow"
    )
    return getattr(module, "build_embedded_exercise_cache")


def exercise_named(name: str):
    Exercise = get_exercise_model()
    return Exercise(
        name=name,
        description=f"{name} description.",
        muscle_groups=["Back"],
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )


class RecordingEmbeddingProvider:
    def __init__(
        self,
        embeddings: dict[str, list[float]] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self._embeddings = {} if embeddings is None else dict(embeddings)
        self._errors = {} if errors is None else dict(errors)
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)

        if text in self._errors:
            raise self._errors[text]

        try:
            return list(self._embeddings[text])
        except KeyError as error:
            raise KeyError(f"Unknown text: {text}") from error


class OnePassExerciseIterable:
    def __init__(self, exercises):
        self._exercises = tuple(exercises)
        self.iteration_count = 0
        self.yielded_names: list[str] = []

    def __iter__(self):
        self.iteration_count += 1
        if self.iteration_count > 1:
            raise AssertionError("exercise iterable should be consumed only once")

        for exercise in self._exercises:
            self.yielded_names.append(exercise.name)
            yield exercise

    def __len__(self) -> int:
        raise AssertionError("len() should not be used on the exercise stream")


class OnePassExerciseRepository:
    def __init__(self, exercises):
        self._iterable = OnePassExerciseIterable(exercises)
        self.iter_exercises_call_count = 0

    @property
    def yielded_names(self) -> list[str]:
        return self._iterable.yielded_names

    def iter_exercises(self):
        self.iter_exercises_call_count += 1
        if self.iter_exercises_call_count > 1:
            raise AssertionError("iter_exercises() should be called only once")
        return self._iterable


class RecordingCacheWriter:
    def __init__(self) -> None:
        self.received_embedded_exercises = None
        self.received_metadata = None
        self.written_embedded_exercises = []

    def write_embedded_exercises(self, embedded_exercises, metadata) -> None:
        self.received_embedded_exercises = embedded_exercises
        self.received_metadata = metadata
        self.written_embedded_exercises = list(embedded_exercises)


class StreamingRecordingCacheWriter:
    def __init__(self, exercise_repository: OnePassExerciseRepository) -> None:
        self._exercise_repository = exercise_repository
        self.received_embedded_exercises = None
        self.raw_yielded_when_first_seen: list[str] | None = None
        self.first_embedded_exercise = None
        self.remaining_embedded_exercises = []

    def write_embedded_exercises(self, embedded_exercises, metadata) -> None:
        self.received_embedded_exercises = embedded_exercises

        iterator = iter(embedded_exercises)
        self.first_embedded_exercise = next(iterator)
        self.raw_yielded_when_first_seen = list(self._exercise_repository.yielded_names)
        self.remaining_embedded_exercises = list(iterator)


class FailingCacheWriter:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def write_embedded_exercises(self, embedded_exercises, metadata) -> None:
        raise self._error


def test_build_embedded_exercise_cache_exists_and_accepts_expected_arguments():
    build_embedded_exercise_cache = get_build_embedded_exercise_cache()

    assert list(inspect.signature(build_embedded_exercise_cache).parameters) == [
        "exercise_repository",
        "embedding_provider",
        "cache_writer",
        "metadata",
    ]


def test_build_embedded_exercise_cache_writes_embedded_exercises_in_encounter_order():
    EmbeddedExercise = get_embedded_exercise_model()
    build_embedded_exercise_cache = get_build_embedded_exercise_cache()
    build_exercise_text = get_build_exercise_text()
    exercises = [exercise_named("Pull Up"), exercise_named("Body Row")]
    exercise_repository = OnePassExerciseRepository(exercises)
    provider = RecordingEmbeddingProvider(
        embeddings={
            build_exercise_text(exercises[0]): [1.0, 0.0],
            build_exercise_text(exercises[1]): [0.8, 0.2],
        }
    )
    cache_writer = RecordingCacheWriter()
    metadata = object()

    build_embedded_exercise_cache(
        exercise_repository=exercise_repository,
        embedding_provider=provider,
        cache_writer=cache_writer,
        metadata=metadata,
    )

    assert cache_writer.received_metadata is metadata
    assert not isinstance(cache_writer.received_embedded_exercises, list)
    assert len(cache_writer.written_embedded_exercises) == 2
    assert all(
        isinstance(embedded_exercise, EmbeddedExercise)
        for embedded_exercise in cache_writer.written_embedded_exercises
    )
    assert [item.exercise for item in cache_writer.written_embedded_exercises] == exercises
    assert cache_writer.written_embedded_exercises[0].exercise is exercises[0]
    assert cache_writer.written_embedded_exercises[1].exercise is exercises[1]
    assert [item.embedding for item in cache_writer.written_embedded_exercises] == [
        (1.0, 0.0),
        (0.8, 0.2),
    ]


def test_build_embedded_exercise_cache_uses_build_exercise_text_outputs_in_order():
    build_embedded_exercise_cache = get_build_embedded_exercise_cache()
    build_exercise_text = get_build_exercise_text()
    exercises = [exercise_named("Pull Up"), exercise_named("Body Row")]
    expected_texts = [build_exercise_text(exercise) for exercise in exercises]
    exercise_repository = OnePassExerciseRepository(exercises)
    provider = RecordingEmbeddingProvider(
        embeddings={
            expected_texts[0]: [1.0, 0.0],
            expected_texts[1]: [0.8, 0.2],
        }
    )

    build_embedded_exercise_cache(
        exercise_repository=exercise_repository,
        embedding_provider=provider,
        cache_writer=RecordingCacheWriter(),
        metadata={"cache": "v1"},
    )

    assert provider.calls == expected_texts
    assert all("Exercise(" not in call for call in provider.calls)


def test_build_embedded_exercise_cache_streams_from_repository_to_writer():
    build_embedded_exercise_cache = get_build_embedded_exercise_cache()
    build_exercise_text = get_build_exercise_text()
    exercises = [
        exercise_named("Pull Up"),
        exercise_named("Body Row"),
        exercise_named("Chin Up"),
    ]
    exercise_repository = OnePassExerciseRepository(exercises)
    provider = RecordingEmbeddingProvider(
        embeddings={
            build_exercise_text(exercises[0]): [1.0, 0.0],
            build_exercise_text(exercises[1]): [0.8, 0.2],
            build_exercise_text(exercises[2]): [0.9, 0.1],
        }
    )
    cache_writer = StreamingRecordingCacheWriter(exercise_repository)

    build_embedded_exercise_cache(
        exercise_repository=exercise_repository,
        embedding_provider=provider,
        cache_writer=cache_writer,
        metadata={"cache": "v1"},
    )

    assert exercise_repository.iter_exercises_call_count == 1
    assert not isinstance(cache_writer.received_embedded_exercises, list)
    assert cache_writer.first_embedded_exercise.exercise.name == "Pull Up"
    assert cache_writer.raw_yielded_when_first_seen == ["Pull Up"]
    assert [item.exercise.name for item in cache_writer.remaining_embedded_exercises] == [
        "Body Row",
        "Chin Up",
    ]
    assert exercise_repository.yielded_names == ["Pull Up", "Body Row", "Chin Up"]


def test_build_embedded_exercise_cache_propagates_embedding_errors_during_writer_consumption():
    build_embedded_exercise_cache = get_build_embedded_exercise_cache()
    build_exercise_text = get_build_exercise_text()
    exercises = [exercise_named("Pull Up"), exercise_named("Body Row")]
    first_text = build_exercise_text(exercises[0])
    second_text = build_exercise_text(exercises[1])
    exercise_repository = OnePassExerciseRepository(exercises)
    provider = RecordingEmbeddingProvider(
        embeddings={first_text: [1.0, 0.0]},
        errors={second_text: RuntimeError("embedding failed")},
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        build_embedded_exercise_cache(
            exercise_repository=exercise_repository,
            embedding_provider=provider,
            cache_writer=RecordingCacheWriter(),
            metadata={"cache": "v1"},
        )


def test_build_embedded_exercise_cache_propagates_cache_writer_errors():
    build_embedded_exercise_cache = get_build_embedded_exercise_cache()
    exercise_repository = OnePassExerciseRepository([exercise_named("Pull Up")])
    provider = RecordingEmbeddingProvider(
        embeddings={get_build_exercise_text()(exercise_named("Pull Up")): [1.0, 0.0]}
    )

    with pytest.raises(RuntimeError, match="cache write failed"):
        build_embedded_exercise_cache(
            exercise_repository=exercise_repository,
            embedding_provider=provider,
            cache_writer=FailingCacheWriter(RuntimeError("cache write failed")),
            metadata={"cache": "v1"},
        )


def test_build_embedded_exercise_cache_has_no_direct_file_csv_network_or_recommender_side_effects(
    monkeypatch,
):
    build_embedded_exercise_cache = get_build_embedded_exercise_cache()
    build_exercise_text = get_build_exercise_text()
    exercises = [exercise_named("Pull Up"), exercise_named("Body Row")]
    exercise_repository = OnePassExerciseRepository(exercises)
    provider = RecordingEmbeddingProvider(
        embeddings={
            build_exercise_text(exercises[0]): [1.0, 0.0],
            build_exercise_text(exercises[1]): [0.8, 0.2],
        }
    )
    cache_writer = RecordingCacheWriter()

    def fail(*args, **kwargs):
        raise AssertionError("unexpected boundary crossing")

    recommend_module = import_module(
        "calisthenics_recommender.application.recommend_exercises"
    )

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(Path, "write_text", fail)
    monkeypatch.setattr(csv, "reader", fail)
    monkeypatch.setattr(csv, "DictReader", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(recommend_module, "recommend_exercises", fail)

    build_embedded_exercise_cache(
        exercise_repository=exercise_repository,
        embedding_provider=provider,
        cache_writer=cache_writer,
        metadata={"cache": "v1"},
    )

    assert [item.exercise.name for item in cache_writer.written_embedded_exercises] == [
        "Pull Up",
        "Body Row",
    ]
