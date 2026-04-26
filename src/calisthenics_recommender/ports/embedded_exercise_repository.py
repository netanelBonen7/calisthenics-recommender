from typing import Iterable, Protocol, runtime_checkable

from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise


@runtime_checkable
class EmbeddedExerciseRepository(Protocol):
    def iter_embedded_exercises(self) -> Iterable[EmbeddedExercise]:
        ...
