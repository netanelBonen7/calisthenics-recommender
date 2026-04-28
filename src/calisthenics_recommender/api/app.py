from __future__ import annotations

from fastapi import FastAPI

from calisthenics_recommender.api.models import (
    CategoryFamilyResponse,
    HealthResponse,
    RecommendationResponse,
    RecommendRequest,
    RecommendResponse,
)
from calisthenics_recommender.application.recommend_exercises import (
    recommend_exercises,
)
from calisthenics_recommender.domain.recommendation import Recommendation
from calisthenics_recommender.domain.user_request import UserRequest
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider
from calisthenics_recommender.ports.embedded_exercise_search_repository import (
    EmbeddedExerciseSearchRepository,
)
from calisthenics_recommender.ports.query_text_builder import QueryTextBuilder


def create_app(
    *,
    embedded_exercise_search_repository: EmbeddedExerciseSearchRepository,
    embedding_provider: EmbeddingProvider,
    query_text_builder: QueryTextBuilder,
) -> FastAPI:
    app = FastAPI()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(request: RecommendRequest) -> RecommendResponse:
        user_request = UserRequest(
            target_family=request.target_family,
            goal=request.goal,
            current_level=request.current_level,
            available_equipment=list(request.available_equipment),
        )
        recommendations = recommend_exercises(
            user_request=user_request,
            embedded_exercise_search_repository=embedded_exercise_search_repository,
            embedding_provider=embedding_provider,
            query_text_builder=query_text_builder,
            limit=request.limit,
        )
        return RecommendResponse(
            recommendations=[
                _to_recommendation_response(recommendation)
                for recommendation in recommendations
            ]
        )

    return app


def _to_recommendation_response(
    recommendation: Recommendation,
) -> RecommendationResponse:
    return RecommendationResponse(
        exercise_name=recommendation.exercise_name,
        match_score=int(recommendation.match_score),
        reason=recommendation.reason,
        required_equipment=list(recommendation.required_equipment),
        category_family=CategoryFamilyResponse(
            categories=list(recommendation.category_family.categories),
            families=list(recommendation.category_family.families),
        ),
    )
