from pydantic import BaseModel, ConfigDict

from calisthenics_recommender.domain.exercise import Exercise


class EmbeddedExercise(BaseModel):
    model_config = ConfigDict(frozen=True)

    exercise: Exercise
    embedding: tuple[float, ...]
