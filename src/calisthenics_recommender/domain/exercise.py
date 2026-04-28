from pydantic import BaseModel

from calisthenics_recommender.domain.types import ExerciseId, NonEmptyString


class Exercise(BaseModel):
    exercise_id: ExerciseId
    name: NonEmptyString
    description: NonEmptyString
    muscle_groups: list[str]
    families: list[str]
    materials: list[str]
    categories: list[str]
