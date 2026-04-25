from pydantic import BaseModel, Field

from calisthenics_recommender.domain.types import NonEmptyString


class CategoryFamily(BaseModel):
    categories: list[str]
    families: list[str]


class Recommendation(BaseModel):
    exercise_name: NonEmptyString
    match_score: float = Field(ge=0, le=100)
    reason: NonEmptyString
    required_equipment: list[str]
    category_family: CategoryFamily
