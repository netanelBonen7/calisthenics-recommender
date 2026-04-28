from typing import Protocol, runtime_checkable

from calisthenics_recommender.domain.exercise import Exercise


@runtime_checkable
class ExerciseLookupRepository(Protocol):
    def get_by_exercise_id(self, exercise_id: str) -> Exercise | None:
        ...
