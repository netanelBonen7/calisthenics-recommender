from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.adapters.local_deterministic_embedding_provider import (
    LocalDeterministicEmbeddingProvider,
)
from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    LocalEmbeddedExerciseRepository,
    read_embedded_exercise_cache_metadata,
)
from calisthenics_recommender.application.recommend_exercises import (
    recommend_exercises,
)
from calisthenics_recommender.domain.recommendation import Recommendation
from calisthenics_recommender.domain.user_request import UserRequest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read an existing local embedded exercise cache and print "
            "human-readable exercise recommendations for a user request."
        )
    )
    parser.add_argument("--cache-path", required=True, type=Path)
    parser.add_argument("--target-family", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--current-level", required=True)
    parser.add_argument("--available-equipment", required=True, action="append")
    parser.add_argument("--limit", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)

    metadata = read_embedded_exercise_cache_metadata(args.cache_path)
    embedding_provider = LocalDeterministicEmbeddingProvider(
        dimension=metadata.embedding_dimension
    )
    embedded_exercise_repository = LocalEmbeddedExerciseRepository(args.cache_path)
    user_request = UserRequest(
        target_family=args.target_family,
        goal=args.goal,
        current_level=args.current_level,
        available_equipment=list(args.available_equipment),
    )
    recommendations = recommend_exercises(
        user_request=user_request,
        embedded_exercise_repository=embedded_exercise_repository,
        embedding_provider=embedding_provider,
        limit=args.limit,
    )

    _print_recommendations(recommendations)
    return 0


def _print_recommendations(recommendations: list[Recommendation]) -> None:
    if not recommendations:
        print("No recommendations found.")
        return

    for rank, recommendation in enumerate(recommendations, start=1):
        print(f"{rank}. Exercise: {recommendation.exercise_name}")
        print(f"   Match score: {recommendation.match_score}")
        print(f"   Reason: {recommendation.reason}")
        print(
            "   Required equipment: "
            f"{', '.join(recommendation.required_equipment)}"
        )
        print(
            "   Categories: "
            f"{', '.join(recommendation.category_family.categories)}"
        )
        print(
            "   Families: "
            f"{', '.join(recommendation.category_family.families)}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
