from typing import Protocol, runtime_checkable

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
)
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise


@runtime_checkable
class EmbeddedExerciseCacheUpdater(Protocol):
    def upsert_embedded_exercise(
        self,
        embedded_exercise: EmbeddedExercise,
        metadata: EmbeddedExerciseCacheMetadata,
    ) -> None:
        ...

    def delete_embedded_exercise(self, exercise_id: str) -> None:
        ...
