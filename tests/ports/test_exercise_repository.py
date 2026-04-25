from importlib import import_module
import inspect
from typing import get_type_hints


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_exercise_repository_protocol():
    module = import_module("calisthenics_recommender.ports.exercise_repository")
    return getattr(module, "ExerciseRepository")


def valid_exercise():
    Exercise = get_exercise_model()
    return Exercise(
        name="Pull Up Negative",
        description="A controlled eccentric pull-up variation for building pulling strength.",
        muscle_groups=["Back", "Biceps"],
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )


def test_exercise_repository_is_a_runtime_checkable_protocol_with_list_exercises():
    Exercise = get_exercise_model()
    ExerciseRepository = get_exercise_repository_protocol()

    assert getattr(ExerciseRepository, "_is_protocol", False) is True
    assert getattr(ExerciseRepository, "_is_runtime_protocol", False) is True
    assert list(inspect.signature(ExerciseRepository.list_exercises).parameters) == ["self"]
    assert get_type_hints(ExerciseRepository.list_exercises)["return"] == list[Exercise]


def test_exercise_repository_can_be_implemented_by_a_simple_fake_class():
    ExerciseRepository = get_exercise_repository_protocol()
    exercise = valid_exercise()

    class InMemoryExerciseRepository:
        def __init__(self, exercises):
            self._exercises = exercises

        def list_exercises(self):
            return self._exercises

    repository = InMemoryExerciseRepository([exercise])

    assert isinstance(repository, ExerciseRepository)
    assert repository.list_exercises() == [exercise]
    assert all(item.__class__ is exercise.__class__ for item in repository.list_exercises())
