from itertools import chain
import logging
from typing import Iterable, Iterator, Sequence

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


class _FilteredEmbeddedExerciseStream(Iterator[EmbeddedExercise]):
    def __init__(
        self,
        embedded_exercises: Iterable[EmbeddedExercise],
        available_equipment: list[str],
    ) -> None:
        self._embedded_exercises = iter(embedded_exercises)
        self._available_equipment = available_equipment
        self.scanned_count = 0
        self.matched_count = 0

    def __iter__(self) -> "_FilteredEmbeddedExerciseStream":
        return self

    def __next__(self) -> EmbeddedExercise:
        for embedded_exercise in self._embedded_exercises:
            self.scanned_count += 1
            if exercise_matches_equipment(
                embedded_exercise.exercise, self._available_equipment
            ):
                self.matched_count += 1
                return embedded_exercise

        raise StopIteration


def recommend_exercises(
    user_request: UserRequest,
    embedded_exercise_repository: EmbeddedExerciseRepository,
    embedding_provider: EmbeddingProvider,
    limit: int,
) -> list[Recommendation]:
    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    filtered_embedded_exercises = _FilteredEmbeddedExerciseStream(
        embedded_exercise_repository.iter_embedded_exercises(),
        user_request.available_equipment,
    )
    first_candidate = next(filtered_embedded_exercises, None)
    if first_candidate is None:
        logger.info(
            "Scanned %d embedded exercises; %d matched equipment; returning %d recommendations",
            filtered_embedded_exercises.scanned_count,
            filtered_embedded_exercises.matched_count,
            0,
        )
        return []

    query_embedding = embedding_provider.embed(build_query_text(user_request))
    retrieval_results = retrieve_top_matches(
        query_embedding=query_embedding,
        embedded_exercises=chain((first_candidate,), filtered_embedded_exercises),
        limit=limit,
    )
    recommendations = build_recommendations(user_request, retrieval_results)
    logger.info(
        "Scanned %d embedded exercises; %d matched equipment; returning %d recommendations",
        filtered_embedded_exercises.scanned_count,
        filtered_embedded_exercises.matched_count,
        len(recommendations),
    )
    return recommendations


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
