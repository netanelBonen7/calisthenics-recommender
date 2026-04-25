from math import sqrt
from typing import Sequence


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same length")

    dot_product = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
    norm_a = sqrt(sum(value * value for value in vector_a))
    norm_b = sqrt(sum(value * value for value in vector_b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("cosine similarity is undefined for zero vectors")

    return dot_product / (norm_a * norm_b)
