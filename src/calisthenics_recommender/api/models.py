from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from calisthenics_recommender.domain.types import NonEmptyString


class RecommendRequest(BaseModel):
    target_family: NonEmptyString
    goal: NonEmptyString
    current_level: NonEmptyString
    available_equipment: list[str]
    limit: int = Field(default=5, gt=0)


class CategoryFamilyResponse(BaseModel):
    categories: list[str]
    families: list[str]


class RecommendationResponse(BaseModel):
    exercise_name: str
    match_score: int
    reason: str
    required_equipment: list[str]
    category_family: CategoryFamilyResponse


class RecommendResponse(BaseModel):
    recommendations: list[RecommendationResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]
