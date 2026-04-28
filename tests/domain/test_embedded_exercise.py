from importlib import import_module

import pytest
from pydantic import ValidationError


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_model():
    module = import_module("calisthenics_recommender.domain.embedded_exercise")
    return getattr(module, "EmbeddedExercise")


def valid_exercise():
    Exercise = get_exercise_model()
    return Exercise(
        exercise_id="pull-up-negative",
        name="Pull Up Negative",
        description="A controlled eccentric pull-up variation for building pulling strength.",
        muscle_groups=["Back", "Biceps"],
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )


def test_embedded_exercise_can_be_created_with_an_exercise_and_embedding():
    EmbeddedExercise = get_embedded_exercise_model()
    exercise = valid_exercise()

    embedded_exercise = EmbeddedExercise(exercise=exercise, embedding=[1.0, 0.0, 0.0])

    assert embedded_exercise.exercise == exercise
    assert embedded_exercise.embedding == (1.0, 0.0, 0.0)
    assert isinstance(embedded_exercise.embedding, tuple)


def test_embedded_exercise_rejects_plain_string_for_embedding():
    EmbeddedExercise = get_embedded_exercise_model()

    with pytest.raises(ValidationError):
        EmbeddedExercise(exercise=valid_exercise(), embedding="not-a-vector")


def test_embedded_exercise_is_immutable():
    EmbeddedExercise = get_embedded_exercise_model()
    embedded_exercise = EmbeddedExercise(
        exercise=valid_exercise(), embedding=[1.0, 0.0, 0.0]
    )

    with pytest.raises(ValidationError):
        embedded_exercise.embedding = [0.0, 1.0, 0.0]


def test_embedded_exercise_embedding_cannot_be_mutated_by_item_assignment():
    EmbeddedExercise = get_embedded_exercise_model()
    embedded_exercise = EmbeddedExercise(
        exercise=valid_exercise(), embedding=[1.0, 0.0, 0.0]
    )

    with pytest.raises(TypeError):
        embedded_exercise.embedding[0] = 0.0
