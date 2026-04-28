from dataclasses import dataclass
from typing import Literal

from calisthenics_recommender.domain.types import ExerciseId


PendingEmbeddingOperation = Literal["upsert", "delete"]


@dataclass(frozen=True)
class PendingEmbeddingUpdate:
    exercise_id: ExerciseId
    operation: PendingEmbeddingOperation
    version: int

    def __post_init__(self) -> None:
        if self.operation not in ("upsert", "delete"):
            raise ValueError("operation must be 'upsert' or 'delete'")
        if isinstance(self.version, bool) or self.version <= 0:
            raise ValueError("version must be greater than 0")
