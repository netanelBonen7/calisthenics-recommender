from importlib import import_module
import inspect
from typing import get_type_hints


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


def valid_embedded_exercise():
    Exercise = get_exercise_model()
    EmbeddedExercise = get_embedded_exercise_model()
    exercise = Exercise(
        name="Pull Up Negative",
        description="A controlled eccentric pull-up variation for building pulling strength.",
        muscle_groups=["Back", "Biceps"],
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    return EmbeddedExercise(exercise=exercise, embedding=[1.0, 0.0, 0.0])


def test_embedded_exercise_repository_is_a_runtime_checkable_protocol():
    EmbeddedExercise = get_embedded_exercise_model()
    EmbeddedExerciseRepository = get_embedded_exercise_repository_protocol()

    assert getattr(EmbeddedExerciseRepository, "_is_protocol", False) is True
    assert getattr(EmbeddedExerciseRepository, "_is_runtime_protocol", False) is True
    assert list(
        inspect.signature(
            EmbeddedExerciseRepository.list_embedded_exercises
        ).parameters
    ) == ["self"]
    assert get_type_hints(EmbeddedExerciseRepository.list_embedded_exercises)[
        "return"
    ] == list[EmbeddedExercise]


def test_embedded_exercise_repository_can_be_implemented_by_a_simple_fake_class():
    EmbeddedExerciseRepository = get_embedded_exercise_repository_protocol()
    embedded_exercise = valid_embedded_exercise()

    class InMemoryEmbeddedExerciseRepository:
        def __init__(self, embedded_exercises):
            self._embedded_exercises = embedded_exercises

        def list_embedded_exercises(self):
            return self._embedded_exercises

    repository = InMemoryEmbeddedExerciseRepository([embedded_exercise])

    assert isinstance(repository, EmbeddedExerciseRepository)
    assert repository.list_embedded_exercises() == [embedded_exercise]
