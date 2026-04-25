from dataclasses import dataclass
from typing import Mapping, Sequence

from calisthenics_recommender.application.similarity import cosine_similarity
from calisthenics_recommender.domain.exercise import Exercise


@dataclass(frozen=True)
class RetrievalResult:
    exercise: Exercise
    score: float


def retrieve_top_matches(
    query_embedding: Sequence[float],
    exercises: Sequence[Exercise],
    exercise_embeddings: Mapping[str, Sequence[float]],
    limit: int,
) -> list[RetrievalResult]:
    scored_results: list[RetrievalResult] = []

    for exercise in exercises:
        try:
            exercise_embedding = exercise_embeddings[exercise.name]
        except KeyError as error:
            raise KeyError(f"missing embedding for exercise '{exercise.name}'") from error

        score = cosine_similarity(query_embedding, exercise_embedding)
        scored_results.append(RetrievalResult(exercise=exercise, score=score))

    ranked_results = sorted(scored_results, key=lambda result: result.score, reverse=True)
    return ranked_results[:limit]
