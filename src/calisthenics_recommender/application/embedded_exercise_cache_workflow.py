from collections.abc import Iterable
from typing import Protocol

from calisthenics_recommender.application.embedded_exercise_builder import (
    build_embedded_exercises,
)
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider
from calisthenics_recommender.ports.exercise_repository import ExerciseRepository


class EmbeddedExerciseCacheWriter(Protocol):
    def write_embedded_exercises(
        self,
        embedded_exercises: Iterable[EmbeddedExercise],
        metadata: object,
    ) -> None:
        ...


def build_embedded_exercise_cache(
    exercise_repository: ExerciseRepository,
    embedding_provider: EmbeddingProvider,
    cache_writer: EmbeddedExerciseCacheWriter,
    metadata: object,
) -> None:
    embedded_exercises = build_embedded_exercises(
        exercises=exercise_repository.iter_exercises(),
        embedding_provider=embedding_provider,
    )
    cache_writer.write_embedded_exercises(
        embedded_exercises=embedded_exercises,
        metadata=metadata,
    )
