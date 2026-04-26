from dataclasses import dataclass
from typing import Sequence

from calisthenics_recommender.application.similarity import cosine_similarity
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise


@dataclass(frozen=True)
class RetrievalResult:
    exercise: Exercise
    score: float


def retrieve_top_matches(
    query_embedding: Sequence[float],
    embedded_exercises: Sequence[EmbeddedExercise],
    limit: int,
) -> list[RetrievalResult]:
    scored_results: list[RetrievalResult] = []

    for embedded_exercise in embedded_exercises:
        score = cosine_similarity(query_embedding, embedded_exercise.embedding)
        scored_results.append(
            RetrievalResult(exercise=embedded_exercise.exercise, score=score)
        )

    ranked_results = sorted(scored_results, key=lambda result: result.score, reverse=True)
    return ranked_results[:limit]
