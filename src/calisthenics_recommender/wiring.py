from __future__ import annotations

from typing import Any

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
    LocalEmbeddedExerciseCache,
    LocalEmbeddedExerciseRepository,
    read_embedded_exercise_cache_metadata,
)
from calisthenics_recommender.adapters.sentence_transformer_embedding_provider import (
    SentenceTransformerEmbeddingProvider,
)
from calisthenics_recommender.adapters.sqlite_embedded_exercise_cache import (
    SQLiteEmbeddedExerciseCache,
    read_sqlite_embedded_exercise_cache_metadata,
)
from calisthenics_recommender.adapters.sqlite_embedded_exercise_search_repository import (
    SQLiteEmbeddedExerciseSearchRepository,
)
from calisthenics_recommender.adapters.sqlite_exercise_repository import (
    SQLiteExerciseRepository,
)
from calisthenics_recommender.config import (
    EmbeddedCacheConfig,
    EmbeddingConfig,
    RawExercisesConfig,
)
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider
from calisthenics_recommender.ports.embedded_exercise_search_repository import (
    EmbeddedExerciseSearchRepository,
)
from calisthenics_recommender.ports.exercise_repository import ExerciseRepository


def build_exercise_repository(
    raw_exercises_config: RawExercisesConfig,
) -> ExerciseRepository:
    if raw_exercises_config.backend == "sqlite":
        return SQLiteExerciseRepository(raw_exercises_config.path)
    return CsvExerciseRepository(raw_exercises_config.path)


def build_embedded_exercise_cache_writer(
    embedded_cache_config: EmbeddedCacheConfig,
) -> Any:
    if embedded_cache_config.backend == "sqlite":
        return SQLiteEmbeddedExerciseCache(embedded_cache_config.path)
    return LocalEmbeddedExerciseCache(embedded_cache_config.path)


def build_embedded_exercise_search_repository(
    embedded_cache_config: EmbeddedCacheConfig,
) -> EmbeddedExerciseSearchRepository:
    if embedded_cache_config.backend == "sqlite":
        return SQLiteEmbeddedExerciseSearchRepository(embedded_cache_config.path)
    return JsonlEmbeddedExerciseSearchRepository(
        LocalEmbeddedExerciseRepository(embedded_cache_config.path)
    )


def read_embedded_cache_metadata(
    embedded_cache_config: EmbeddedCacheConfig,
) -> EmbeddedExerciseCacheMetadata:
    cache_path = embedded_cache_config.path
    try:
        if embedded_cache_config.backend == "sqlite":
            return read_sqlite_embedded_exercise_cache_metadata(cache_path)
        return read_embedded_exercise_cache_metadata(cache_path)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError(
            f"Invalid embedded exercise cache metadata for {cache_path}: {error}"
        ) from error


def build_query_embedding_provider(
    *,
    embedding_config: EmbeddingConfig,
    metadata: EmbeddedExerciseCacheMetadata,
) -> EmbeddingProvider:
    if embedding_config.provider == "sentence-transformer":
        model_name = embedding_config.model or metadata.embedding_model
        embedding_provider = SentenceTransformerEmbeddingProvider(
            model_name=model_name,
            text_prefix=embedding_config.query_prefix,
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


def build_cache_embedding_provider_and_metadata(
    embedding_config: EmbeddingConfig,
) -> tuple[object, EmbeddedExerciseCacheMetadata]:
    text_builder_version = _require_text_builder_version(embedding_config)

    if embedding_config.provider == "sentence-transformer":
        model_name = (
            embedding_config.model
            or SentenceTransformerEmbeddingProvider.DEFAULT_MODEL_NAME
        )
        embedding_provider = SentenceTransformerEmbeddingProvider(
            model_name=model_name,
            text_prefix=embedding_config.text_prefix,
        )
        metadata = EmbeddedExerciseCacheMetadata(
            embedding_model=model_name,
            embedding_dimension=embedding_provider.get_embedding_dimension(),
            text_builder_version=text_builder_version,
        )
        return embedding_provider, metadata

    model_name = _require_embedding_model(embedding_config)
    embedding_dimension = _require_embedding_dimension(embedding_config)
    metadata = EmbeddedExerciseCacheMetadata(
        embedding_model=model_name,
        embedding_dimension=embedding_dimension,
        text_builder_version=text_builder_version,
    )
    embedding_provider = LocalDeterministicEmbeddingProvider(
        dimension=embedding_dimension
    )
    return embedding_provider, metadata


def _require_embedding_model(embedding_config: EmbeddingConfig) -> str:
    if embedding_config.model is None:
        raise ValueError("--embedding-model is required for local-deterministic mode")
    return embedding_config.model


def _require_embedding_dimension(embedding_config: EmbeddingConfig) -> int:
    if embedding_config.dimension is None:
        raise ValueError(
            "--embedding-dimension is required for local-deterministic mode"
        )
    return embedding_config.dimension


def _require_text_builder_version(embedding_config: EmbeddingConfig) -> str:
    if embedding_config.text_builder_version is None:
        raise ValueError("--text-builder-version is required")
    return embedding_config.text_builder_version
