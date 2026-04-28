from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.application.recommend_exercises import (
    recommend_exercises,
)
from calisthenics_recommender.config import (
    EmbeddedCacheConfig,
    EmbeddingConfig,
    QueryBuilderConfig,
    RecommenderConfig,
    load_recommender_config,
)
from calisthenics_recommender.domain.recommendation import Recommendation
from calisthenics_recommender.domain.user_request import UserRequest
from calisthenics_recommender.wiring import (
    build_embedded_exercise_search_repository,
    build_query_text_builder,
    build_query_embedding_provider,
    read_embedded_cache_metadata,
)


def build_argument_parser(
    config: RecommenderConfig | None = None,
) -> argparse.ArgumentParser:
    embedded_cache_config = None if config is None else config.embedded_cache
    embedding_config = None if config is None else config.embedding

    parser = argparse.ArgumentParser(
        description=(
            "Read an existing local embedded exercise cache and print "
            "human-readable exercise recommendations for a user request."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--cache-path",
        required=embedded_cache_config is None,
        type=Path,
        default=(
            None if embedded_cache_config is None else embedded_cache_config.path
        ),
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("local-deterministic", "sentence-transformer"),
        default=_default_embedding_provider(embedding_config),
    )
    parser.add_argument(
        "--embedding-model",
        default=None if embedding_config is None else embedding_config.model,
    )
    parser.add_argument("--target-family", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--current-level", required=True)
    parser.add_argument("--available-equipment", required=True, action="append")
    parser.add_argument(
        "--query-builder-strategy",
        choices=("v1",),
        default=None,
    )
    parser.add_argument(
        "--query-prefix",
        default="" if embedding_config is None else embedding_config.query_prefix,
    )
    parser.add_argument("--limit", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else None
    config = _load_optional_config(argv_list)
    args = build_argument_parser(config).parse_args(argv_list)

    embedded_cache_config = _resolve_embedded_cache_config(args, config)
    metadata = read_embedded_cache_metadata(embedded_cache_config)
    embedding_provider = build_query_embedding_provider(
        embedding_config=_resolve_embedding_config(args),
        metadata=metadata,
    )
    query_text_builder = build_query_text_builder(
        _resolve_query_builder_config(args, config)
    )
    user_request = UserRequest(
        target_family=args.target_family,
        goal=args.goal,
        current_level=args.current_level,
        available_equipment=list(args.available_equipment),
    )
    recommendations = recommend_exercises(
        user_request=user_request,
        embedded_exercise_search_repository=build_embedded_exercise_search_repository(
            embedded_cache_config
        ),
        embedding_provider=embedding_provider,
        query_text_builder=query_text_builder,
        limit=args.limit,
    )

    _print_recommendations(recommendations)
    return 0


def _default_embedding_provider(embedding_config: EmbeddingConfig | None) -> str:
    if embedding_config is None:
        return "local-deterministic"
    return embedding_config.provider


def _load_optional_config(argv: Sequence[str] | None) -> RecommenderConfig | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    args, _ = parser.parse_known_args(argv)
    if args.config is None:
        return None
    return load_recommender_config(args.config)


def _resolve_embedded_cache_config(
    args: argparse.Namespace,
    config: RecommenderConfig | None,
) -> EmbeddedCacheConfig:
    if args.cache_path is None:
        raise ValueError("--cache-path is required")

    backend = "jsonl"
    if config is not None and config.embedded_cache is not None:
        backend = config.embedded_cache.backend

    return EmbeddedCacheConfig(
        backend=backend,
        path=args.cache_path,
    )


def _resolve_embedding_config(args: argparse.Namespace) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=args.embedding_provider,
        model=args.embedding_model,
        query_prefix=args.query_prefix,
    )


def _resolve_query_builder_config(
    args: argparse.Namespace,
    config: RecommenderConfig | None,
) -> QueryBuilderConfig:
    if args.query_builder_strategy is not None:
        return QueryBuilderConfig(strategy=args.query_builder_strategy)
    if config is not None:
        return config.query_builder
    return QueryBuilderConfig()


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
