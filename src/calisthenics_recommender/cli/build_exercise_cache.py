from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.application.embedded_exercise_cache_workflow import (
    build_embedded_exercise_cache,
)
from calisthenics_recommender.config import (
    EmbeddedCacheConfig,
    EmbeddingConfig,
    ExerciseTextBuilderConfig,
    RawExercisesConfig,
    RecommenderConfig,
    load_recommender_config,
)
from calisthenics_recommender.wiring import (
    build_cache_embedding_provider_and_metadata,
    build_embedded_exercise_cache_writer,
    build_exercise_text_builder,
    build_exercise_repository,
)


def build_argument_parser(
    config: RecommenderConfig | None = None,
) -> argparse.ArgumentParser:
    raw_exercises_config = None if config is None else config.raw_exercises
    embedded_cache_config = None if config is None else config.embedded_cache
    embedding_config = None if config is None else config.embedding

    parser = argparse.ArgumentParser(
        description=(
            "Build a local embedded exercise cache from a CSV file or SQLite "
            "database using either deterministic local embeddings or a "
            "sentence-transformer model."
        )
    )
    parser.add_argument("--config", type=Path)
    input_group = parser.add_mutually_exclusive_group(
        required=raw_exercises_config is None
    )
    input_group.add_argument("--input-csv", type=Path)
    input_group.add_argument("--input-db", type=Path)
    parser.add_argument(
        "--output-cache",
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
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=None if embedding_config is None else embedding_config.dimension,
    )
    parser.add_argument(
        "--text-prefix",
        default="" if embedding_config is None else embedding_config.text_prefix,
    )
    parser.add_argument(
        "--exercise-text-builder-strategy",
        choices=("v1",),
        default=None,
    )
    parser.add_argument(
        "--text-builder-version",
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else None
    config = _load_optional_config(argv_list)
    args = build_argument_parser(config).parse_args(argv_list)

    raw_exercises_config = _resolve_raw_exercises_config(args, config)
    embedded_cache_config = _resolve_embedded_cache_config(args, config)
    embedding_config = _resolve_embedding_config(args, config)
    exercise_text_builder_config = _resolve_exercise_text_builder_config(args, config)

    embedding_provider, metadata = build_cache_embedding_provider_and_metadata(
        embedding_config,
        exercise_text_builder_config,
    )
    exercise_repository = build_exercise_repository(raw_exercises_config)
    cache_writer = build_embedded_exercise_cache_writer(embedded_cache_config)
    exercise_text_builder = build_exercise_text_builder(exercise_text_builder_config)

    build_embedded_exercise_cache(
        exercise_repository=exercise_repository,
        embedding_provider=embedding_provider,
        exercise_text_builder=exercise_text_builder,
        cache_writer=cache_writer,
        metadata=metadata,
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


def _resolve_raw_exercises_config(
    args: argparse.Namespace,
    config: RecommenderConfig | None,
) -> RawExercisesConfig:
    if args.input_db is not None:
        return RawExercisesConfig(backend="sqlite", path=args.input_db)
    if args.input_csv is not None:
        return RawExercisesConfig(backend="csv", path=args.input_csv)
    if config is None or config.raw_exercises is None:
        raise ValueError("Exactly one raw exercise input source is required")
    return config.raw_exercises


def _resolve_embedded_cache_config(
    args: argparse.Namespace,
    config: RecommenderConfig | None,
) -> EmbeddedCacheConfig:
    if args.output_cache is None:
        raise ValueError("--output-cache is required")

    backend = "jsonl"
    if config is not None and config.embedded_cache is not None:
        backend = config.embedded_cache.backend

    return EmbeddedCacheConfig(
        backend=backend,
        path=args.output_cache,
    )


def _resolve_embedding_config(
    args: argparse.Namespace,
    config: RecommenderConfig | None,
) -> EmbeddingConfig:
    return EmbeddingConfig(
        provider=args.embedding_provider,
        model=args.embedding_model,
        dimension=args.embedding_dimension,
        query_prefix=(
            ""
            if config is None or config.embedding is None
            else config.embedding.query_prefix
        ),
        text_prefix=args.text_prefix,
        text_builder_version=(
            None
            if config is None or config.embedding is None
            else config.embedding.text_builder_version
        ),
    )


def _resolve_exercise_text_builder_config(
    args: argparse.Namespace,
    config: RecommenderConfig | None,
) -> ExerciseTextBuilderConfig:
    if args.exercise_text_builder_strategy is not None:
        return ExerciseTextBuilderConfig(strategy=args.exercise_text_builder_strategy)
    if args.text_builder_version is not None:
        return ExerciseTextBuilderConfig(strategy=args.text_builder_version)
    if config is not None:
        return config.exercise_text_builder
    return ExerciseTextBuilderConfig()


if __name__ == "__main__":
    raise SystemExit(main())
