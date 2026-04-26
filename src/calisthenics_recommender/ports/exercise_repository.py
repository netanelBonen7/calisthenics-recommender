from typing import Iterable, Protocol, runtime_checkable

from calisthenics_recommender.domain.exercise import Exercise


@runtime_checkable
class ExerciseRepository(Protocol):
    def iter_exercises(self) -> Iterable[Exercise]:
        ...
