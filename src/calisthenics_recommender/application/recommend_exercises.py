from typing import Sequence

from calisthenics_recommender.application.explanation_builder import (
    build_explanation,
)
from calisthenics_recommender.application.retriever import RetrievalResult
from calisthenics_recommender.domain.recommendation import (
    CategoryFamily,
    Recommendation,
)
from calisthenics_recommender.domain.user_request import UserRequest


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
