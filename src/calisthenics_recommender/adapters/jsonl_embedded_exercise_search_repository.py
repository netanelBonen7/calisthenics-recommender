from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator, Sequence

from calisthenics_recommender.application.filters import exercise_matches_equipment
from calisthenics_recommender.application.similarity import cosine_similarity
from calisthenics_recommender.domain.embedded_exercise_search_result import (
    EmbeddedExerciseSearchResult,
)
from calisthenics_recommender.ports.embedded_exercise_repository import (
    EmbeddedExerciseRepository,
)


class JsonlEmbeddedExerciseSearchRepository:
    def __init__(self, embedded_exercise_repository: EmbeddedExerciseRepository) -> None:
        self._embedded_exercise_repository = embedded_exercise_repository

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        available_equipment: Sequence[str],
        limit: int,
    ) -> Iterable[EmbeddedExerciseSearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        return self._iter_search_results(
            query_embedding=query_embedding,
            available_equipment=list(available_equipment),
            limit=limit,
        )

    def _iter_search_results(
        self,
        *,
        query_embedding: Sequence[float],
        available_equipment: list[str],
        limit: int,
    ) -> Iterator[EmbeddedExerciseSearchResult]:
        heap: list[tuple[float, int, int, EmbeddedExerciseSearchResult]] = []

        for encounter_index, embedded_exercise in enumerate(
            self._embedded_exercise_repository.iter_embedded_exercises()
        ):
            if not exercise_matches_equipment(
                embedded_exercise.exercise,
                available_equipment,
            ):
                continue

            similarity = cosine_similarity(query_embedding, embedded_exercise.embedding)
            search_result = EmbeddedExerciseSearchResult(
                exercise=embedded_exercise.exercise,
                similarity=similarity,
            )
            heap_entry = (similarity, -encounter_index, encounter_index, search_result)

            if len(heap) < limit:
                heapq.heappush(heap, heap_entry)
                continue

            if heap_entry[:3] > heap[0][:3]:
                heapq.heapreplace(heap, heap_entry)

        for _, _, _, search_result in sorted(heap, key=lambda entry: (-entry[0], entry[2])):
            yield search_result
