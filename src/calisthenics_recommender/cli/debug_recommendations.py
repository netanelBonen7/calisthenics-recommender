from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from calisthenics_recommender.adapters.csv_exercise_repository import (
    CsvExerciseRepository,
)
from calisthenics_recommender.adapters.jsonl_embedded_exercise_search_repository import (
    JsonlEmbeddedExerciseSearchRepository,
)
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
from calisthenics_recommender.application.exercise_text_builder import (
    build_exercise_text,
)
from calisthenics_recommender.application.query_builder import build_query_text
from calisthenics_recommender.domain.user_request import UserRequest
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect query text, exercise text, and retrieval candidates using "
            "the current recommendation builders and an existing local cache."
        )
    )
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--exercise-name", action="append")
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument(
        "--embedding-provider",
        choices=("local-deterministic", "sentence-transformer"),
        default="local-deterministic",
    )
    parser.add_argument("--embedding-model")
    parser.add_argument("--target-family")
    parser.add_argument("--goal")
    parser.add_argument("--current-level")
    parser.add_argument("--available-equipment", action="append")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--limit", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(list(argv) if argv is not None else None)
    _validate_args(args)
    user_request = (
        _build_user_request(args) if _has_complete_user_request_args(args) else None
    )

    if user_request is not None:
        _print_query_text(user_request)

    if args.exercise_name:
        _print_exercise_texts(
            input_csv=args.input_csv,
            requested_names=list(args.exercise_name),
        )

    if args.cache_path is not None and user_request is not None:
        _print_top_candidates(
            cache_path=args.cache_path,
            embedding_provider_name=args.embedding_provider,
            embedding_model=args.embedding_model,
            query_prefix=args.query_prefix,
            user_request=user_request,
            limit=args.limit,
        )

    return 0


def _validate_args(args: argparse.Namespace) -> None:
    has_complete_request = _has_complete_user_request_args(args)
    has_any_request_fields = _has_any_user_request_args(args)

    if args.exercise_name and args.input_csv is None:
        raise ValueError("--exercise-name requires --input-csv")

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
        args.input_csv is None
        and args.cache_path is None
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


def _print_query_text(user_request: UserRequest) -> None:
    print("=== QUERY TEXT ===")
    print(build_query_text(user_request))


def _print_exercise_texts(*, input_csv: Path, requested_names: list[str]) -> None:
    matches_by_name = {name: [] for name in requested_names}
    repository = CsvExerciseRepository(input_csv)

    for exercise in repository.iter_exercises():
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
            print(build_exercise_text(exercise))


def _print_top_candidates(
    *,
    cache_path: Path,
    embedding_provider_name: str,
    embedding_model: str | None,
    query_prefix: str,
    user_request: UserRequest,
    limit: int,
) -> None:
    metadata = read_embedded_exercise_cache_metadata(cache_path)
    embedding_provider = _build_embedding_provider(
        embedding_provider_name=embedding_provider_name,
        embedding_model=embedding_model,
        metadata=metadata,
        query_prefix=query_prefix,
    )
    query_text = build_query_text(user_request)
    query_embedding = embedding_provider.embed(query_text)
    search_repository = JsonlEmbeddedExerciseSearchRepository(
        LocalEmbeddedExerciseRepository(cache_path)
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
        for line in build_exercise_text(exercise).splitlines():
            print(f"   {line}")


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


if __name__ == "__main__":
    raise SystemExit(main())
