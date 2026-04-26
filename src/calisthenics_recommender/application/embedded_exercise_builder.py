from collections.abc import Iterable, Iterator

from calisthenics_recommender.application.exercise_text_builder import (
    build_exercise_text,
)
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider


def build_embedded_exercises(
    exercises: Iterable[Exercise],
    embedding_provider: EmbeddingProvider,
) -> Iterator[EmbeddedExercise]:
    for exercise in exercises:
        exercise_text = build_exercise_text(exercise)
        embedding = embedding_provider.embed(exercise_text)
        yield EmbeddedExercise(exercise=exercise, embedding=embedding)
