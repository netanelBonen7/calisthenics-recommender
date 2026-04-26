from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.adapters.local_deterministic_embedding_provider import (
    LocalDeterministicEmbeddingProvider,
)
from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
    LocalEmbeddedExerciseRepository,
    read_embedded_exercise_cache_metadata,
)
from calisthenics_recommender.adapters.sentence_transformer_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
)
from calisthenics_recommender.application.recommend_exercises import (
    recommend_exercises,
)
from calisthenics_recommender.domain.recommendation import Recommendation
from calisthenics_recommender.domain.user_request import UserRequest
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read an existing local embedded exercise cache and print "
            "human-readable exercise recommendations for a user request."
        )
    )
    parser.add_argument("--cache-path", required=True, type=Path)
    parser.add_argument(
        "--embedding-provider",
        choices=("local-deterministic", "sentence-transformer"),
        default="local-deterministic",
    )
    parser.add_argument("--embedding-model")
    parser.add_argument("--target-family", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--current-level", required=True)
    parser.add_argument("--available-equipment", required=True, action="append")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--limit", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)

    metadata = read_embedded_exercise_cache_metadata(args.cache_path)
    embedding_provider = _build_embedding_provider(
        embedding_provider_name=args.embedding_provider,
        embedding_model=args.embedding_model,
        metadata=metadata,
        query_prefix=args.query_prefix,
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


def _build_embedding_provider(
    *,
    embedding_provider_name: str,
    embedding_model: str | None,
    metadata: EmbeddedExerciseCacheMetadata,
    query_prefix: str,
) -> EmbeddingProvider:
    if embedding_provider_name == "sentence-transformer":
        model_name = embedding_model or metadata.embedding_model
        embedding_provider = SentenceTransformerEmbeddingProvider(
            model_name=model_name,
            text_prefix=query_prefix,
        )
        embedding_dimension = embedding_provider.get_embedding_dimension()
        if embedding_dimension != metadata.embedding_dimension:
            raise ValueError(
                "Sentence-transformer embedding dimension does not match cache metadata"
            )
        return embedding_provider

    return LocalDeterministicEmbeddingProvider(
        dimension=metadata.embedding_dimension
    )


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
