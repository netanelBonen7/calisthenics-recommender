import logging
from typing import Sequence

from calisthenics_recommender.application.explanation_builder import (
    build_explanation,
)
from calisthenics_recommender.application.filters import exercise_matches_equipment
from calisthenics_recommender.application.query_builder import build_query_text
from calisthenics_recommender.application.retriever import RetrievalResult
from calisthenics_recommender.application.retriever import retrieve_top_matches
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.recommendation import (
    CategoryFamily,
    Recommendation,
)
from calisthenics_recommender.domain.user_request import UserRequest
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider
from calisthenics_recommender.ports.embedded_exercise_repository import (
    EmbeddedExerciseRepository,
)


logger = logging.getLogger(__name__)


def recommend_exercises(
    user_request: UserRequest,
    embedded_exercise_repository: EmbeddedExerciseRepository,
    embedding_provider: EmbeddingProvider,
    limit: int,
) -> list[Recommendation]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    embedded_exercises = list(
        embedded_exercise_repository.list_embedded_exercises()
    )
    logger.info("Loaded %d embedded exercises from repository", len(embedded_exercises))

    filtered_embedded_exercises = _filter_embedded_exercises_by_equipment(
        embedded_exercises, user_request.available_equipment
    )
    logger.info(
        "Filtered embedded exercises down to %d candidates",
        len(filtered_embedded_exercises),
    )

    if not filtered_embedded_exercises:
        logger.info("Returning %d recommendations", 0)
        return []

    query_embedding = embedding_provider.embed(build_query_text(user_request))
    retrieval_results = retrieve_top_matches(
        query_embedding=query_embedding,
        embedded_exercises=filtered_embedded_exercises,
        limit=limit,
    )
    recommendations = build_recommendations(user_request, retrieval_results)
    logger.info("Returning %d recommendations", len(recommendations))
    return recommendations


def _filter_embedded_exercises_by_equipment(
    embedded_exercises: Sequence[EmbeddedExercise], available_equipment: list[str]
) -> list[EmbeddedExercise]:
    return [
        embedded_exercise
        for embedded_exercise in embedded_exercises
        if exercise_matches_equipment(
            embedded_exercise.exercise, available_equipment
        )
    ]


def build_recommendation(
    user_request: UserRequest, retrieval_result: RetrievalResult
) -> Recommendation:
    exercise = retrieval_result.exercise

    return Recommendation(
        exercise_name=exercise.name,
        match_score=_convert_score_to_match_score(retrieval_result.score),
        reason=build_explanation(user_request, exercise),
        required_equipment=list(exercise.materials),
        category_family=CategoryFamily(
            categories=list(exercise.categories),
            families=list(exercise.families),
        ),
    )


def build_recommendations(
    user_request: UserRequest, retrieval_results: Sequence[RetrievalResult]
) -> list[Recommendation]:
    return [
        build_recommendation(user_request, retrieval_result)
        for retrieval_result in retrieval_results
    ]


def _convert_score_to_match_score(score: float) -> int:
    return max(0, min(100, int(round(score * 100))))
