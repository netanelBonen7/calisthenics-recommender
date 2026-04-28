from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.config import (
    EmbeddedCacheConfig,
    EmbeddingConfig,
    ExerciseTextBuilderConfig,
    QueryBuilderConfig,
    RawExercisesConfig,
    RecommenderConfig,
    load_recommender_config,
)
from calisthenics_recommender.domain.user_request import UserRequest
from calisthenics_recommender.ports.exercise_text_builder import (
    ExerciseTextBuilder,
)
from calisthenics_recommender.ports.exercise_repository import ExerciseRepository
from calisthenics_recommender.ports.query_text_builder import QueryTextBuilder
from calisthenics_recommender.wiring import (
    build_embedded_exercise_search_repository,
    build_exercise_repository,
    build_exercise_text_builder,
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
            "Inspect query text, exercise text, and retrieval candidates using "
            "the current recommendation builders and an existing local cache."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--exercise-name", action="append")
    parser.add_argument(
        "--cache-path",
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
    parser.add_argument("--target-family")
    parser.add_argument("--goal")
    parser.add_argument("--current-level")
    parser.add_argument("--available-equipment", action="append")
    parser.add_argument(
        "--query-builder-strategy",
        choices=("v1",),
        default=None,
    )
    parser.add_argument(
        "--exercise-text-builder-strategy",
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
    _validate_args(args, config)
    user_request = (
        _build_user_request(args) if _has_complete_user_request_args(args) else None
    )
    query_text_builder = build_query_text_builder(
        _resolve_query_builder_config(args, config)
    )
    exercise_text_builder = build_exercise_text_builder(
        _resolve_exercise_text_builder_config(args, config)
    )

    if user_request is not None:
        _print_query_text(user_request, query_text_builder)

    if args.exercise_name:
        _print_exercise_texts(
            exercise_repository=_build_exercise_text_repository(args, config),
            exercise_text_builder=exercise_text_builder,
            requested_names=list(args.exercise_name),
        )

    if args.cache_path is not None and user_request is not None:
        _print_top_candidates(
            embedded_cache_config=_resolve_embedded_cache_config(args, config),
            embedding_config=_resolve_embedding_config(args),
            query_text_builder=query_text_builder,
            exercise_text_builder=exercise_text_builder,
            user_request=user_request,
            limit=args.limit,
        )

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


def _validate_args(
    args: argparse.Namespace,
    config: RecommenderConfig | None,
) -> None:
    has_complete_request = _has_complete_user_request_args(args)
    has_any_request_fields = _has_any_user_request_args(args)
    has_raw_source = args.input_csv is not None or (
        config is not None and config.raw_exercises is not None
    )

    if args.exercise_name and not has_raw_source:
        raise ValueError("--exercise-name requires --input-csv or configured raw_exercises")

    if args.cache_path is not None and not has_complete_request:
        raise ValueError(
            "--cache-path requires --target-family, --goal, --current-level, "
            "and at least one --available-equipment"
        )

    if has_any_request_fields and not has_complete_request:
        raise ValueError(
            "Query inspection requires --target-family, --goal, --current-level, "
            "and at least one --available-equipment"
        )

    if args.cache_path is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")

    if (
        args.cache_path is None
        and not has_complete_request
        and not args.exercise_name
    ):
        raise ValueError("No inspection requested")


def _has_complete_user_request_args(args: argparse.Namespace) -> bool:
    return bool(
        args.target_family
        and args.goal
        and args.current_level
        and args.available_equipment
    )


def _has_any_user_request_args(args: argparse.Namespace) -> bool:
    return bool(
        args.target_family
        or args.goal
        or args.current_level
        or args.available_equipment
    )


def _build_user_request(args: argparse.Namespace) -> UserRequest:
    return UserRequest(
        target_family=args.target_family,
        goal=args.goal,
        current_level=args.current_level,
        available_equipment=list(args.available_equipment),
    )


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


def _resolve_exercise_text_builder_config(
    args: argparse.Namespace,
    config: RecommenderConfig | None,
) -> ExerciseTextBuilderConfig:
    if args.exercise_text_builder_strategy is not None:
        return ExerciseTextBuilderConfig(strategy=args.exercise_text_builder_strategy)
    if config is not None:
        return config.exercise_text_builder
    return ExerciseTextBuilderConfig()


def _build_exercise_text_repository(
    args: argparse.Namespace,
    config: RecommenderConfig | None,
) -> ExerciseRepository:
    if args.input_csv is not None:
        return build_exercise_repository(
            RawExercisesConfig(backend="csv", path=args.input_csv)
        )
    if config is None or config.raw_exercises is None:
        raise ValueError("--exercise-name requires --input-csv or configured raw_exercises")
    return build_exercise_repository(config.raw_exercises)


def _print_query_text(
    user_request: UserRequest,
    query_text_builder: QueryTextBuilder,
) -> None:
    print("=== QUERY TEXT ===")
    print(query_text_builder.build(user_request))


def _print_exercise_texts(
    *,
    exercise_repository: ExerciseRepository,
    exercise_text_builder: ExerciseTextBuilder,
    requested_names: list[str],
) -> None:
    matches_by_name = {name: [] for name in requested_names}

    for exercise in exercise_repository.iter_exercises():
        if exercise.name in matches_by_name:
            matches_by_name[exercise.name].append(exercise)

    print("=== EXERCISE TEXTS ===")
    for requested_name in requested_names:
        matches = matches_by_name[requested_name]
        if not matches:
            print(f"--- {requested_name} ---")
            print("Exercise not found.")
            continue

        multiple_matches = len(matches) > 1
        for index, exercise in enumerate(matches, start=1):
            header = (
                f"--- {requested_name} (match {index}) ---"
                if multiple_matches
                else f"--- {requested_name} ---"
            )
            print(header)
            print(exercise_text_builder.build(exercise))


def _print_top_candidates(
    *,
    embedded_cache_config: EmbeddedCacheConfig,
    embedding_config: EmbeddingConfig,
    query_text_builder: QueryTextBuilder,
    exercise_text_builder: ExerciseTextBuilder,
    user_request: UserRequest,
    limit: int,
) -> None:
    metadata = read_embedded_cache_metadata(embedded_cache_config)
    embedding_provider = build_query_embedding_provider(
        embedding_config=embedding_config,
        metadata=metadata,
    )
    query_text = query_text_builder.build(user_request)
    query_embedding = embedding_provider.embed(query_text)
    search_repository = build_embedded_exercise_search_repository(
        embedded_cache_config
    )
    search_results = search_repository.search(
        query_embedding=query_embedding,
        available_equipment=user_request.available_equipment,
        limit=limit,
    )

    print("=== TOP CANDIDATES ===")
    search_results = list(search_results)
    if not search_results:
        print("No candidates found.")
        return

    for rank, search_result in enumerate(search_results, start=1):
        exercise = search_result.exercise
        print(f"{rank}. Exercise: {exercise.name}")
        print(f"   Score: {search_result.similarity:.6f}")
        print(f"   Families: {', '.join(exercise.families)}")
        print(f"   Categories: {', '.join(exercise.categories)}")
        print(f"   Required equipment: {', '.join(exercise.materials)}")
        print("   Exercise text:")
        for line in exercise_text_builder.build(exercise).splitlines():
            print(f"   {line}")


if __name__ == "__main__":
    raise SystemExit(main())
