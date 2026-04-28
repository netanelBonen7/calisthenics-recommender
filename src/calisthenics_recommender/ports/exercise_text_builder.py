from __future__ import annotations

from typing import Protocol, runtime_checkable

from calisthenics_recommender.domain.exercise import Exercise


@runtime_checkable
class ExerciseTextBuilder(Protocol):
    def build(self, exercise: Exercise) -> str:
        ...
