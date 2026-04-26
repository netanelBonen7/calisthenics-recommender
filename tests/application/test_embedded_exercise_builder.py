from importlib import import_module
import inspect
from pathlib import Path
import socket
from typing import Iterable, Iterator

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


def get_build_embedded_exercises():
    module = import_module(
        "calisthenics_recommender.application.embedded_exercise_builder"
    )
    return getattr(module, "build_embedded_exercises")


def exercise_named(name: str, materials: list[str] | None = None):
    Exercise = get_exercise_model()
    return Exercise(
        name=name,
        description=f"{name} description.",
        muscle_groups=["Back"],
        families=["Pull-up"],
        materials=["Bar"] if materials is None else materials,
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


class LenExplodingIterable(Iterable):
    def __init__(self, values):
        self._values = tuple(values)

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __len__(self) -> int:
        raise AssertionError("len() should not be used")


def test_build_embedded_exercises_exists_and_accepts_expected_arguments():
    build_embedded_exercises = get_build_embedded_exercises()

    assert list(inspect.signature(build_embedded_exercises).parameters) == [
        "exercises",
        "embedding_provider",
    ]


def test_build_embedded_exercises_yields_embedded_exercises_with_original_exercises():
    EmbeddedExercise = get_embedded_exercise_model()
    build_embedded_exercises = get_build_embedded_exercises()
    build_exercise_text = get_build_exercise_text()
    exercises = [exercise_named("Pull Up"), exercise_named("Body Row")]
    provider = RecordingEmbeddingProvider(
        embeddings={
            build_exercise_text(exercises[0]): [1.0, 0.0],
            build_exercise_text(exercises[1]): [0.8, 0.2],
        }
    )

    results = list(build_embedded_exercises(exercises, provider))

    assert len(results) == 2
    assert all(isinstance(result, EmbeddedExercise) for result in results)
    assert results[0].exercise is exercises[0]
    assert results[1].exercise is exercises[1]
    assert results[0].embedding == (1.0, 0.0)
    assert results[1].embedding == (0.8, 0.2)
    assert isinstance(results[0].embedding, tuple)
    assert isinstance(results[1].embedding, tuple)


def test_build_embedded_exercises_uses_build_exercise_text_outputs_in_encounter_order():
    build_embedded_exercises = get_build_embedded_exercises()
    build_exercise_text = get_build_exercise_text()
    exercises = [exercise_named("Pull Up"), exercise_named("Body Row")]
    expected_texts = [build_exercise_text(exercise) for exercise in exercises]
    provider = RecordingEmbeddingProvider(
        embeddings={
            expected_texts[0]: [1.0, 0.0],
            expected_texts[1]: [0.8, 0.2],
        }
    )

    results = list(build_embedded_exercises(exercises, provider))

    assert len(results) == 2
    assert provider.calls == expected_texts
    assert all("Exercise(" not in call for call in provider.calls)


def test_build_embedded_exercises_accepts_a_one_pass_generator():
    build_embedded_exercises = get_build_embedded_exercises()
    build_exercise_text = get_build_exercise_text()
    exercises = [exercise_named("Pull Up"), exercise_named("Body Row")]
    provider = RecordingEmbeddingProvider(
        embeddings={
            build_exercise_text(exercises[0]): [1.0, 0.0],
            build_exercise_text(exercises[1]): [0.8, 0.2],
        }
    )

    results = list(
        build_embedded_exercises((exercise for exercise in exercises), provider)
    )

    assert [result.exercise.name for result in results] == ["Pull Up", "Body Row"]


def test_build_embedded_exercises_does_not_require_len_on_exercises():
    build_embedded_exercises = get_build_embedded_exercises()
    build_exercise_text = get_build_exercise_text()
    exercises = LenExplodingIterable([exercise_named("Pull Up"), exercise_named("Body Row")])
    expected_texts = [build_exercise_text(exercise) for exercise in exercises]
    provider = RecordingEmbeddingProvider(
        embeddings={
            expected_texts[0]: [1.0, 0.0],
            expected_texts[1]: [0.8, 0.2],
        }
    )

    results = list(build_embedded_exercises(exercises, provider))

    assert [result.exercise.name for result in results] == ["Pull Up", "Body Row"]


def test_build_embedded_exercises_yields_the_first_result_without_consuming_all_inputs():
    build_embedded_exercises = get_build_embedded_exercises()
    build_exercise_text = get_build_exercise_text()
    exercises = [
        exercise_named("Pull Up"),
        exercise_named("Body Row"),
        exercise_named("Chin Up"),
    ]
    yielded_names: list[str] = []

    def generate_exercises():
        for exercise in exercises:
            yielded_names.append(exercise.name)
            yield exercise

    provider = RecordingEmbeddingProvider(
        embeddings={
            build_exercise_text(exercises[0]): [1.0, 0.0],
            build_exercise_text(exercises[1]): [0.8, 0.2],
            build_exercise_text(exercises[2]): [0.9, 0.1],
        }
    )

    iterator = iter(build_embedded_exercises(generate_exercises(), provider))
    first = next(iterator)

    assert first.exercise.name == "Pull Up"
    assert yielded_names == ["Pull Up"]

    remaining = list(iterator)

    assert [result.exercise.name for result in remaining] == ["Body Row", "Chin Up"]
    assert yielded_names == ["Pull Up", "Body Row", "Chin Up"]


@pytest.mark.parametrize("error_type", [KeyError, RuntimeError])
def test_build_embedded_exercises_propagates_embedding_provider_errors(error_type):
    build_embedded_exercises = get_build_embedded_exercises()
    build_exercise_text = get_build_exercise_text()
    exercises = [exercise_named("Pull Up"), exercise_named("Body Row")]
    failing_text = build_exercise_text(exercises[1])
    provider = RecordingEmbeddingProvider(
        embeddings={build_exercise_text(exercises[0]): [1.0, 0.0]},
        errors={failing_text: error_type("embedding failed")},
    )

    iterator = build_embedded_exercises(exercises, provider)

    first = next(iterator)

    assert first.exercise.name == "Pull Up"
    with pytest.raises(error_type, match="embedding failed"):
        next(iterator)


def test_build_embedded_exercises_is_pure_and_does_not_mutate_input_exercises(
    monkeypatch,
):
    build_embedded_exercises = get_build_embedded_exercises()
    build_exercise_text = get_build_exercise_text()
    exercises = [
        exercise_named("Pull Up", materials=["Bar", "Rings"]),
        exercise_named("Body Row", materials=["Bar"]),
    ]
    original_dumps = [exercise.model_dump() for exercise in exercises]
    provider = RecordingEmbeddingProvider(
        embeddings={
            build_exercise_text(exercises[0]): [1.0, 0.0],
            build_exercise_text(exercises[1]): [0.8, 0.2],
        }
    )

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    results = list(build_embedded_exercises(exercises, provider))

    assert [result.exercise.name for result in results] == ["Pull Up", "Body Row"]
    assert [exercise.model_dump() for exercise in exercises] == original_dumps
