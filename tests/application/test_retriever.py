from copy import deepcopy
from importlib import import_module
import inspect
from pathlib import Path
import socket

import pytest


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_retrieve_top_matches():
    module = import_module("calisthenics_recommender.application.retriever")
    return getattr(module, "retrieve_top_matches")


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


def result_names(results) -> list[str]:
    return [result.exercise.name for result in results]


def result_scores(results) -> list[float]:
    return [result.score for result in results]


def test_retrieve_top_matches_accepts_expected_arguments():
    retrieve_top_matches = get_retrieve_top_matches()

    assert list(inspect.signature(retrieve_top_matches).parameters) == [
        "query_embedding",
        "exercises",
        "exercise_embeddings",
        "limit",
    ]


def test_retrieve_top_matches_returns_exercises_sorted_by_cosine_similarity_descending():
    retrieve_top_matches = get_retrieve_top_matches()
    exercises = [
        exercise_named("Pull Up"),
        exercise_named("Body Row"),
        exercise_named("Push Up"),
    ]
    exercise_embeddings = {
        "Pull Up": [1.0, 0.0],
        "Body Row": [0.8, 0.2],
        "Push Up": [0.0, 1.0],
    }

    results = retrieve_top_matches([1.0, 0.0], exercises, exercise_embeddings, limit=3)

    assert result_names(results) == ["Pull Up", "Body Row", "Push Up"]
    assert results[0].exercise == exercises[0]
    assert results[0].score == pytest.approx(1.0)
    assert results[-1].score == pytest.approx(0.0)
    assert result_scores(results) == sorted(result_scores(results), reverse=True)


def test_retrieve_top_matches_respects_the_requested_limit():
    retrieve_top_matches = get_retrieve_top_matches()
    exercises = [
        exercise_named("Pull Up"),
        exercise_named("Body Row"),
        exercise_named("Push Up"),
    ]
    exercise_embeddings = {
        "Pull Up": [1.0, 0.0],
        "Body Row": [0.8, 0.2],
        "Push Up": [0.0, 1.0],
    }

    results = retrieve_top_matches([1.0, 0.0], exercises, exercise_embeddings, limit=2)

    assert result_names(results) == ["Pull Up", "Body Row"]


def test_retrieve_top_matches_returns_all_matches_when_limit_exceeds_candidate_count():
    retrieve_top_matches = get_retrieve_top_matches()
    exercises = [
        exercise_named("Pull Up"),
        exercise_named("Body Row"),
    ]
    exercise_embeddings = {
        "Pull Up": [1.0, 0.0],
        "Body Row": [0.8, 0.2],
    }

    results = retrieve_top_matches([1.0, 0.0], exercises, exercise_embeddings, limit=10)

    assert result_names(results) == ["Pull Up", "Body Row"]


def test_retrieve_top_matches_is_deterministic_for_equal_scores_and_preserves_input_order():
    retrieve_top_matches = get_retrieve_top_matches()
    exercises = [
        exercise_named("Chin Up"),
        exercise_named("Neutral Grip Pull Up"),
        exercise_named("Body Row"),
    ]
    exercise_embeddings = {
        "Chin Up": [1.0, 0.0],
        "Neutral Grip Pull Up": [1.0, 0.0],
        "Body Row": [0.8, 0.2],
    }

    first_results = retrieve_top_matches(
        [1.0, 0.0], exercises, exercise_embeddings, limit=3
    )
    second_results = retrieve_top_matches(
        [1.0, 0.0], exercises, exercise_embeddings, limit=3
    )

    assert result_names(first_results) == [
        "Chin Up",
        "Neutral Grip Pull Up",
        "Body Row",
    ]
    assert result_names(second_results) == result_names(first_results)
    assert result_scores(second_results) == pytest.approx(result_scores(first_results))


def test_retrieve_top_matches_raises_a_clear_error_when_an_exercise_embedding_is_missing():
    retrieve_top_matches = get_retrieve_top_matches()
    exercises = [
        exercise_named("Pull Up"),
        exercise_named("Body Row"),
    ]
    exercise_embeddings = {
        "Pull Up": [1.0, 0.0],
    }

    with pytest.raises(KeyError, match="Body Row|embedding|Embedding"):
        retrieve_top_matches([1.0, 0.0], exercises, exercise_embeddings, limit=2)


@pytest.mark.parametrize(
    ("query_embedding", "exercise_embedding"),
    [
        ([0.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [0.0, 0.0]),
        ([1.0, 0.0], [1.0, 0.0, 0.0]),
    ],
)
def test_retrieve_top_matches_raises_clearly_for_invalid_vectors(
    query_embedding, exercise_embedding
):
    retrieve_top_matches = get_retrieve_top_matches()
    exercises = [exercise_named("Pull Up")]
    exercise_embeddings = {
        "Pull Up": exercise_embedding,
    }

    with pytest.raises(ValueError, match="zero|length"):
        retrieve_top_matches(query_embedding, exercises, exercise_embeddings, limit=1)


def test_retrieve_top_matches_does_not_mutate_inputs():
    retrieve_top_matches = get_retrieve_top_matches()
    query_embedding = [1.0, 0.0]
    exercises = [
        exercise_named("Pull Up"),
        exercise_named("Push Up"),
    ]
    exercise_embeddings = {
        "Pull Up": [1.0, 0.0],
        "Push Up": [0.0, 1.0],
    }
    original_query_embedding = list(query_embedding)
    original_exercises = deepcopy(exercises)
    original_exercise_embeddings = deepcopy(exercise_embeddings)

    retrieve_top_matches(query_embedding, exercises, exercise_embeddings, limit=2)

    assert query_embedding == original_query_embedding
    assert exercises == original_exercises
    assert exercise_embeddings == original_exercise_embeddings


def test_retrieve_top_matches_is_pure_and_does_not_touch_files_or_network(monkeypatch):
    retrieve_top_matches = get_retrieve_top_matches()
    exercises = [exercise_named("Pull Up")]
    exercise_embeddings = {
        "Pull Up": [1.0, 0.0],
    }

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    results = retrieve_top_matches([1.0, 0.0], exercises, exercise_embeddings, limit=1)

    assert result_names(results) == ["Pull Up"]
