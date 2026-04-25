from typing import Protocol, runtime_checkable

from calisthenics_recommender.domain.exercise import Exercise


@runtime_checkable
class ExerciseRepository(Protocol):
    def list_exercises(self) -> list[Exercise]:
        ...
