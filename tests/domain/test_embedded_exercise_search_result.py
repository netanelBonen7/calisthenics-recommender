from importlib import import_module

import pytest
from pydantic import ValidationError


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_search_result_model():
    module = import_module(
        "calisthenics_recommender.domain.embedded_exercise_search_result"
    )
    return getattr(module, "EmbeddedExerciseSearchResult")


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


def test_embedded_exercise_search_result_can_be_created_with_exercise_and_similarity():
    EmbeddedExerciseSearchResult = get_embedded_exercise_search_result_model()
    exercise = valid_exercise()

    result = EmbeddedExerciseSearchResult(exercise=exercise, similarity=0.87)

    assert result.exercise == exercise
    assert result.similarity == 0.87


def test_embedded_exercise_search_result_is_immutable():
    EmbeddedExerciseSearchResult = get_embedded_exercise_search_result_model()
    result = EmbeddedExerciseSearchResult(exercise=valid_exercise(), similarity=0.87)

    with pytest.raises(ValidationError):
        result.similarity = 0.5


def test_embedded_exercise_search_result_rejects_invalid_exercise_payload():
    EmbeddedExerciseSearchResult = get_embedded_exercise_search_result_model()

    with pytest.raises(ValidationError):
        EmbeddedExerciseSearchResult(exercise="not-an-exercise", similarity=0.87)


@pytest.mark.parametrize("similarity", [-1.1, 1.1, float("nan"), float("inf"), float("-inf")])
def test_embedded_exercise_search_result_rejects_invalid_similarity(similarity):
    EmbeddedExerciseSearchResult = get_embedded_exercise_search_result_model()

    with pytest.raises(ValidationError, match="similarity"):
        EmbeddedExerciseSearchResult(exercise=valid_exercise(), similarity=similarity)
