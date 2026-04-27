from __future__ import annotations

import os
from typing import Mapping

from fastapi import FastAPI

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
from calisthenics_recommender.adapters.sqlite_embedded_exercise_cache import (
    read_sqlite_embedded_exercise_cache_metadata,
)
from calisthenics_recommender.adapters.sqlite_embedded_exercise_search_repository import (
    SQLiteEmbeddedExerciseSearchRepository,
)
from calisthenics_recommender.api.app import create_app
from calisthenics_recommender.config import (
    ApiRuntimeConfig,
    load_api_runtime_config,
)
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider
from calisthenics_recommender.ports.embedded_exercise_search_repository import (
    EmbeddedExerciseSearchRepository,
)


_CONFIG_PATH_ENV_VAR = "CALISTHENICS_RECOMMENDER_CONFIG_PATH"


def read_runtime_config_from_env(
    environ: Mapping[str, str] | None = None,
) -> ApiRuntimeConfig:
    env = os.environ if environ is None else environ
    config_path = _require_env_var(env, _CONFIG_PATH_ENV_VAR)
    return load_api_runtime_config(config_path)


def create_configured_app_from_env(
    environ: Mapping[str, str] | None = None,
) -> FastAPI:
    config = read_runtime_config_from_env(environ)
    metadata = _read_cache_metadata(config)
    embedded_exercise_search_repository = (
        _build_embedded_exercise_search_repository(config)
    )
    embedding_provider = _build_embedding_provider(
        config=config,
        metadata=metadata,
    )
    return create_app(
        embedded_exercise_search_repository=embedded_exercise_search_repository,
        embedding_provider=embedding_provider,
    )


def _build_embedded_exercise_search_repository(
    config: ApiRuntimeConfig,
) -> EmbeddedExerciseSearchRepository:
    if config.embedded_cache.backend == "sqlite":
        return SQLiteEmbeddedExerciseSearchRepository(config.embedded_cache.path)

    return JsonlEmbeddedExerciseSearchRepository(
        LocalEmbeddedExerciseRepository(config.embedded_cache.path)
    )


def _build_embedding_provider(
    *,
    config: ApiRuntimeConfig,
    metadata: EmbeddedExerciseCacheMetadata,
) -> EmbeddingProvider:
    if config.embedding.provider == "sentence-transformer":
        model_name = config.embedding.model or metadata.embedding_model
        embedding_provider = SentenceTransformerEmbeddingProvider(
            model_name=model_name,
            text_prefix=config.embedding.query_prefix,
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


def _read_cache_metadata(config: ApiRuntimeConfig) -> EmbeddedExerciseCacheMetadata:
    cache_path = config.embedded_cache.path
    try:
        if config.embedded_cache.backend == "sqlite":
            return read_sqlite_embedded_exercise_cache_metadata(cache_path)
        return read_embedded_exercise_cache_metadata(cache_path)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ValueError(
            f"Invalid embedded exercise cache metadata for {cache_path}: {error}"
        ) from error


def _require_env_var(environ: Mapping[str, str], name: str) -> str:
    value = _read_optional_env_var(environ, name)
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _read_optional_env_var(
    environ: Mapping[str, str],
    name: str,
) -> str | None:
    value = environ.get(name)
    if value is None:
        return None

    normalized_value = value.strip()
    if not normalized_value:
        return None

    return normalized_value
