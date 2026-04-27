from typing import Iterable, Protocol, Sequence, runtime_checkable

from calisthenics_recommender.domain.embedded_exercise_search_result import (
    EmbeddedExerciseSearchResult,
)


@runtime_checkable
class EmbeddedExerciseSearchRepository(Protocol):
    def search(
        self,
        *,
        query_embedding: Sequence[float],
        available_equipment: Sequence[str],
        limit: int,
    ) -> Iterable[EmbeddedExerciseSearchResult]:
        ...
