from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.application.process_pending_embedding_updates_workflow import (
    ProcessPendingEmbeddingUpdatesWorkflow,
)
from calisthenics_recommender.config import (
    RecommenderConfig,
    load_recommender_config,
)
from calisthenics_recommender.wiring import (
    build_cache_embedding_provider_and_metadata,
    build_embedded_exercise_cache_updater,
    build_exercise_lookup_repository,
    build_pending_embedding_update_repository,
    read_embedded_cache_metadata,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process pending SQLite embedding updates from raw exercises into "
            "a SQLite embedded exercise cache."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit", type=_positive_int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    config = load_recommender_config(args.config)
    _validate_required_config(config)

    raw_exercises_config = config.raw_exercises
    embedded_cache_config = config.embedded_cache
    embedding_config = config.embedding
    if (
        raw_exercises_config is None
        or embedded_cache_config is None
        or embedding_config is None
    ):
        raise AssertionError("config should have been validated")

    pending_update_repository = build_pending_embedding_update_repository(
        raw_exercises_config
    )
    exercise_repository = build_exercise_lookup_repository(raw_exercises_config)
    cache_updater = build_embedded_exercise_cache_updater(embedded_cache_config)
    embedding_provider, expected_metadata = build_cache_embedding_provider_and_metadata(
        embedding_config
    )
    actual_metadata = read_embedded_cache_metadata(embedded_cache_config)

    result = ProcessPendingEmbeddingUpdatesWorkflow(
        pending_update_repository=pending_update_repository,
        exercise_repository=exercise_repository,
        embedding_provider=embedding_provider,
        cache_updater=cache_updater,
        expected_metadata=expected_metadata,
        actual_metadata=actual_metadata,
    ).process(limit=args.limit)

    print(
        "Processed pending embedding updates: "
        f"seen={result.seen_count}, "
        f"processed={result.processed_count}, "
        f"failed={result.failed_count}, "
        f"remaining={result.remaining_count}"
    )
    return 1 if result.failed_count else 0


def _validate_required_config(config: RecommenderConfig) -> None:
    if config.raw_exercises is None:
        raise ValueError("[raw_exercises] config is required")
    if config.raw_exercises.backend != "sqlite":
        raise ValueError("Pending embedding update processing requires SQLite raw exercises")
    if config.embedded_cache is None:
        raise ValueError("[embedded_cache] config is required")
    if config.embedded_cache.backend != "sqlite":
        raise ValueError(
            "Pending embedding update processing requires a SQLite embedded cache"
        )
    if config.embedding is None:
        raise ValueError("[embedding] config is required")


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed_value


if __name__ == "__main__":
    raise SystemExit(main())
