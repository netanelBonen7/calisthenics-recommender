from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.adapters.csv_exercise_repository import (
    CsvExerciseRepository,
)
from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
    LocalEmbeddedExerciseCache,
)
from calisthenics_recommender.application.embedded_exercise_cache_workflow import (
    build_embedded_exercise_cache,
)


class _LocalDeterministicEmbeddingProvider:
    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        return [
            _deterministic_component(text=text, index=index)
            for index in range(self._dimension)
        ]


def _deterministic_component(text: str, index: int) -> float:
    digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer % 1_000_000 + 1) / 1_000_001


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local embedded exercise cache from a CSV file using "
            "deterministic fake embeddings for local development."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-cache", required=True, type=Path)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-dimension", required=True, type=int)
    parser.add_argument("--text-builder-version", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)

    metadata = EmbeddedExerciseCacheMetadata(
        embedding_model=args.embedding_model,
        embedding_dimension=args.embedding_dimension,
        text_builder_version=args.text_builder_version,
    )
    embedding_provider = _LocalDeterministicEmbeddingProvider(
        dimension=metadata.embedding_dimension
    )
    exercise_repository = CsvExerciseRepository(args.input_csv)
    cache_writer = LocalEmbeddedExerciseCache(args.output_cache)

    build_embedded_exercise_cache(
        exercise_repository=exercise_repository,
        embedding_provider=embedding_provider,
        cache_writer=cache_writer,
        metadata=metadata,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
