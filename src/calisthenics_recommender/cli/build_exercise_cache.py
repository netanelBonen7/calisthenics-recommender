from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.adapters.csv_exercise_repository import (
    CsvExerciseRepository,
)
from calisthenics_recommender.adapters.local_deterministic_embedding_provider import (
    LocalDeterministicEmbeddingProvider,
)
from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
    LocalEmbeddedExerciseCache,
)
from calisthenics_recommender.adapters.sentence_transformer_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
)
from calisthenics_recommender.application.embedded_exercise_cache_workflow import (
    build_embedded_exercise_cache,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local embedded exercise cache from a CSV file using "
            "either deterministic local embeddings or a sentence-transformer model."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-cache", required=True, type=Path)
    parser.add_argument(
        "--embedding-provider",
        choices=("local-deterministic", "sentence-transformer"),
        default="local-deterministic",
    )
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument("--text-prefix", default="")
    parser.add_argument("--text-builder-version", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    embedding_provider, metadata = _build_embedding_provider_and_metadata(args)
    exercise_repository = CsvExerciseRepository(args.input_csv)
    cache_writer = LocalEmbeddedExerciseCache(args.output_cache)

    build_embedded_exercise_cache(
        exercise_repository=exercise_repository,
        embedding_provider=embedding_provider,
        cache_writer=cache_writer,
        metadata=metadata,
    )
    return 0


def _build_embedding_provider_and_metadata(
    args: argparse.Namespace,
) -> tuple[object, EmbeddedExerciseCacheMetadata]:
    if args.embedding_provider == "sentence-transformer":
        model_name = (
            args.embedding_model
            or SentenceTransformerEmbeddingProvider.DEFAULT_MODEL_NAME
        )
        embedding_provider = SentenceTransformerEmbeddingProvider(
            model_name=model_name,
            text_prefix=args.text_prefix,
        )
        metadata = EmbeddedExerciseCacheMetadata(
            embedding_model=model_name,
            embedding_dimension=embedding_provider.get_embedding_dimension(),
            text_builder_version=args.text_builder_version,
        )
        return embedding_provider, metadata

    if args.embedding_model is None:
        raise ValueError("--embedding-model is required for local-deterministic mode")
    if args.embedding_dimension is None:
        raise ValueError(
            "--embedding-dimension is required for local-deterministic mode"
        )

    metadata = EmbeddedExerciseCacheMetadata(
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        text_builder_version=args.text_builder_version,
    )
    embedding_provider = LocalDeterministicEmbeddingProvider(
        dimension=metadata.embedding_dimension
    )
    return embedding_provider, metadata


if __name__ == "__main__":
    raise SystemExit(main())
