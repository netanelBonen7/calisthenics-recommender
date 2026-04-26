from typing import Protocol, runtime_checkable

from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise


@runtime_checkable
class EmbeddedExerciseRepository(Protocol):
    def list_embedded_exercises(self) -> list[EmbeddedExercise]:
        ...
