from pydantic import BaseModel

from calisthenics_recommender.domain.types import NonEmptyString


class Exercise(BaseModel):
    name: NonEmptyString
    description: NonEmptyString
    muscle_groups: list[str]
    families: list[str]
    materials: list[str]
    categories: list[str]
