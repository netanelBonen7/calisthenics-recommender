from collections.abc import Iterable, Iterator

from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider
from calisthenics_recommender.ports.exercise_text_builder import (
    ExerciseTextBuilder,
)


def build_embedded_exercises(
    exercises: Iterable[Exercise],
    embedding_provider: EmbeddingProvider,
    exercise_text_builder: ExerciseTextBuilder,
) -> Iterator[EmbeddedExercise]:
    for exercise in exercises:
        exercise_text = exercise_text_builder.build(exercise)
        embedding = embedding_provider.embed(exercise_text)
        yield EmbeddedExercise(exercise=exercise, embedding=embedding)
