from dataclasses import dataclass
import heapq
from typing import Iterable, Sequence

from calisthenics_recommender.application.similarity import cosine_similarity
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise


@dataclass(frozen=True)
class RetrievalResult:
    exercise: Exercise
    score: float


def retrieve_top_matches(
    query_embedding: Sequence[float],
    embedded_exercises: Iterable[EmbeddedExercise],
    limit: int,
) -> list[RetrievalResult]:
    if limit <= 0:
        return []

    heap: list[tuple[float, int, int, RetrievalResult]] = []

    for encounter_index, embedded_exercise in enumerate(embedded_exercises):
        score = cosine_similarity(query_embedding, embedded_exercise.embedding)
        retrieval_result = RetrievalResult(
            exercise=embedded_exercise.exercise, score=score
        )
        heap_entry = (score, -encounter_index, encounter_index, retrieval_result)

        if len(heap) < limit:
            heapq.heappush(heap, heap_entry)
            continue

        if heap_entry[:3] > heap[0][:3]:
            heapq.heapreplace(heap, heap_entry)

    ranked_results = sorted(heap, key=lambda entry: (-entry[0], entry[2]))
    return [entry[3] for entry in ranked_results]
