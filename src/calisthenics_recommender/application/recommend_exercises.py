import logging
from collections.abc import Iterable

from calisthenics_recommender.application.explanation_builder import (
    build_explanation,
)
from calisthenics_recommender.application.query_builder import build_query_text
from calisthenics_recommender.domain.embedded_exercise_search_result import (
    EmbeddedExerciseSearchResult,
)
from calisthenics_recommender.domain.recommendation import (
    CategoryFamily,
    Recommendation,
)
from calisthenics_recommender.domain.user_request import UserRequest
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider
from calisthenics_recommender.ports.embedded_exercise_search_repository import (
    EmbeddedExerciseSearchRepository,
)


logger = logging.getLogger(__name__)


def recommend_exercises(
    user_request: UserRequest,
    embedded_exercise_search_repository: EmbeddedExerciseSearchRepository,
    embedding_provider: EmbeddingProvider,
    limit: int,
) -> list[Recommendation]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    query_embedding = embedding_provider.embed(build_query_text(user_request))
    search_results = embedded_exercise_search_repository.search(
        query_embedding=query_embedding,
        available_equipment=user_request.available_equipment,
        limit=limit,
    )
    recommendations = build_recommendations(user_request, search_results)
    logger.info(
        "Search returned %d recommendations",
        len(recommendations),
    )
    return recommendations


def build_recommendation(
    user_request: UserRequest, search_result: EmbeddedExerciseSearchResult
) -> Recommendation:
    exercise = search_result.exercise

    return Recommendation(
        exercise_name=exercise.name,
        match_score=_convert_score_to_match_score(search_result.similarity),
        reason=build_explanation(user_request, exercise),
        required_equipment=list(exercise.materials),
        category_family=CategoryFamily(
            categories=list(exercise.categories),
            families=list(exercise.families),
        ),
    )


def build_recommendations(
    user_request: UserRequest, search_results: Iterable[EmbeddedExerciseSearchResult]
) -> list[Recommendation]:
    return [
        build_recommendation(user_request, search_result)
        for search_result in search_results
    ]


def _convert_score_to_match_score(score: float) -> int:
    return max(0, min(100, int(round(score * 100))))
