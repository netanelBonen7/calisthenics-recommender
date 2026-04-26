from copy import deepcopy
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


def embedded_exercise_named(name: str, embedding: list[float]):
    EmbeddedExercise = get_embedded_exercise_model()
    return EmbeddedExercise(exercise=exercise_named(name), embedding=embedding)


class LenExplodingIterable(Iterable):
    def __init__(self, values):
        self._values = tuple(values)

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __len__(self) -> int:
        raise AssertionError("len() should not be used")


def result_names(results) -> list[str]:
    return [result.exercise.name for result in results]


def result_scores(results) -> list[float]:
    return [result.score for result in results]


def test_retrieve_top_matches_accepts_expected_arguments():
    retrieve_top_matches = get_retrieve_top_matches()

    assert list(inspect.signature(retrieve_top_matches).parameters) == [
        "query_embedding",
        "embedded_exercises",
        "limit",
    ]


def test_retrieve_top_matches_returns_exercises_sorted_by_cosine_similarity_descending():
    retrieve_top_matches = get_retrieve_top_matches()
    embedded_exercises = [
        embedded_exercise_named("Pull Up", [1.0, 0.0]),
        embedded_exercise_named("Body Row", [0.8, 0.2]),
        embedded_exercise_named("Push Up", [0.0, 1.0]),
    ]

    results = retrieve_top_matches(
        [1.0, 0.0], (exercise for exercise in embedded_exercises), limit=3
    )

    assert result_names(results) == ["Pull Up", "Body Row", "Push Up"]
    assert results[0].exercise == embedded_exercises[0].exercise
    assert results[0].score == pytest.approx(1.0)
    assert results[-1].score == pytest.approx(0.0)
    assert result_scores(results) == sorted(result_scores(results), reverse=True)


def test_retrieve_top_matches_respects_the_requested_limit():
    retrieve_top_matches = get_retrieve_top_matches()
    embedded_exercises = [
        embedded_exercise_named("Pull Up", [1.0, 0.0]),
        embedded_exercise_named("Body Row", [0.8, 0.2]),
        embedded_exercise_named("Push Up", [0.0, 1.0]),
    ]

    results = retrieve_top_matches(
        [1.0, 0.0], (exercise for exercise in embedded_exercises), limit=2
    )

    assert result_names(results) == ["Pull Up", "Body Row"]


def test_retrieve_top_matches_returns_all_matches_when_limit_exceeds_candidate_count():
    retrieve_top_matches = get_retrieve_top_matches()
    embedded_exercises = [
        embedded_exercise_named("Pull Up", [1.0, 0.0]),
        embedded_exercise_named("Body Row", [0.8, 0.2]),
    ]

    results = retrieve_top_matches(
        [1.0, 0.0], (exercise for exercise in embedded_exercises), limit=10
    )

    assert result_names(results) == ["Pull Up", "Body Row"]


def test_retrieve_top_matches_is_deterministic_for_equal_scores_and_preserves_input_order():
    retrieve_top_matches = get_retrieve_top_matches()
    embedded_exercises = [
        embedded_exercise_named("Chin Up", [1.0, 0.0]),
        embedded_exercise_named("Neutral Grip Pull Up", [1.0, 0.0]),
        embedded_exercise_named("Body Row", [0.8, 0.2]),
    ]

    first_results = retrieve_top_matches(
        [1.0, 0.0], (exercise for exercise in embedded_exercises), limit=3
    )
    second_results = retrieve_top_matches(
        [1.0, 0.0], (exercise for exercise in embedded_exercises), limit=3
    )

    assert result_names(first_results) == [
        "Chin Up",
        "Neutral Grip Pull Up",
        "Body Row",
    ]
    assert result_names(second_results) == result_names(first_results)
    assert result_scores(second_results) == pytest.approx(result_scores(first_results))


def test_retrieve_top_matches_supports_duplicate_exercise_names_without_collisions():
    retrieve_top_matches = get_retrieve_top_matches()
    embedded_exercises = [
        embedded_exercise_named("Pull Up", [1.0, 0.0]),
        embedded_exercise_named("Pull Up", [0.0, 1.0]),
    ]

    results = retrieve_top_matches(
        [1.0, 0.0], (exercise for exercise in embedded_exercises), limit=2
    )

    assert result_names(results) == ["Pull Up", "Pull Up"]
    assert result_scores(results) == pytest.approx([1.0, 0.0])


def test_retrieve_top_matches_accepts_a_one_pass_generator():
    retrieve_top_matches = get_retrieve_top_matches()
    embedded_exercises = (
        embedded_exercise_named("Pull Up", [1.0, 0.0]),
        embedded_exercise_named("Body Row", [0.8, 0.2]),
        embedded_exercise_named("Push Up", [0.0, 1.0]),
    )

    results = retrieve_top_matches(
        [1.0, 0.0], (exercise for exercise in embedded_exercises), limit=2
    )

    assert result_names(results) == ["Pull Up", "Body Row"]


def test_retrieve_top_matches_does_not_require_len_on_embedded_exercises():
    retrieve_top_matches = get_retrieve_top_matches()
    embedded_exercises = LenExplodingIterable(
        [
            embedded_exercise_named("Pull Up", [1.0, 0.0]),
            embedded_exercise_named("Body Row", [0.8, 0.2]),
        ]
    )

    results = retrieve_top_matches([1.0, 0.0], embedded_exercises, limit=2)

    assert result_names(results) == ["Pull Up", "Body Row"]


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
    embedded_exercises = (embedded_exercise_named("Pull Up", exercise_embedding),)

    with pytest.raises(ValueError, match="zero|length"):
        retrieve_top_matches(
            query_embedding,
            (exercise for exercise in embedded_exercises),
            limit=1,
        )


def test_retrieve_top_matches_does_not_mutate_inputs():
    retrieve_top_matches = get_retrieve_top_matches()
    query_embedding = [1.0, 0.0]
    embedded_exercises = [
        embedded_exercise_named("Pull Up", [1.0, 0.0]),
        embedded_exercise_named("Push Up", [0.0, 1.0]),
    ]
    original_query_embedding = list(query_embedding)
    original_embedded_exercises = deepcopy(embedded_exercises)

    retrieve_top_matches(
        query_embedding, (exercise for exercise in embedded_exercises), limit=2
    )

    assert query_embedding == original_query_embedding
    assert embedded_exercises == original_embedded_exercises


def test_retrieve_top_matches_is_pure_and_does_not_touch_files_or_network(monkeypatch):
    retrieve_top_matches = get_retrieve_top_matches()
    embedded_exercises = [embedded_exercise_named("Pull Up", [1.0, 0.0])]

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    results = retrieve_top_matches(
        [1.0, 0.0], (exercise for exercise in embedded_exercises), limit=1
    )

    assert result_names(results) == ["Pull Up"]
