from importlib import import_module
import inspect

import pytest


def get_cosine_similarity():
    module = import_module("calisthenics_recommender.application.similarity")
    return getattr(module, "cosine_similarity")


def test_cosine_similarity_accepts_expected_arguments():
    cosine_similarity = get_cosine_similarity()

    assert list(inspect.signature(cosine_similarity).parameters) == [
        "vector_a",
        "vector_b",
    ]


def test_cosine_similarity_returns_1_for_identical_vectors():
    cosine_similarity = get_cosine_similarity()

    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_returns_0_for_orthogonal_vectors():
    cosine_similarity = get_cosine_similarity()

    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_returns_expected_value_for_similar_vectors():
    cosine_similarity = get_cosine_similarity()

    assert cosine_similarity([1.0, 2.0], [2.0, 1.0]) == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("vector_a", "vector_b"),
    [
        ([0.0, 0.0], [1.0, 1.0]),
        ([1.0, 1.0], [0.0, 0.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ],
)
def test_cosine_similarity_raises_value_error_for_zero_vectors(vector_a, vector_b):
    cosine_similarity = get_cosine_similarity()

    with pytest.raises(ValueError):
        cosine_similarity(vector_a, vector_b)


def test_cosine_similarity_raises_value_error_for_mismatched_vector_lengths():
    cosine_similarity = get_cosine_similarity()

    with pytest.raises(ValueError):
        cosine_similarity([1.0, 2.0], [1.0])


def test_cosine_similarity_does_not_mutate_input_vectors():
    cosine_similarity = get_cosine_similarity()
    vector_a = [1.0, 2.0]
    vector_b = [2.0, 1.0]

    cosine_similarity(vector_a, vector_b)

    assert vector_a == [1.0, 2.0]
    assert vector_b == [2.0, 1.0]
