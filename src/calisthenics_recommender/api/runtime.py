from __future__ import annotations

import os
from typing import Mapping

from fastapi import FastAPI

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
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
from calisthenics_recommender.ports.query_text_builder import QueryTextBuilder
from calisthenics_recommender.wiring import (
    build_embedded_exercise_search_repository,
    build_query_text_builder,
    build_query_embedding_provider,
    read_embedded_cache_metadata,
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
        query_text_builder=_build_query_text_builder(config),
    )


def _build_embedded_exercise_search_repository(
    config: ApiRuntimeConfig,
) -> EmbeddedExerciseSearchRepository:
    return build_embedded_exercise_search_repository(
        config.embedded_cache
    )


def _build_embedding_provider(
    *,
    config: ApiRuntimeConfig,
    metadata: EmbeddedExerciseCacheMetadata,
) -> EmbeddingProvider:
    return build_query_embedding_provider(
        embedding_config=config.embedding,
        metadata=metadata,
    )


def _build_query_text_builder(config: ApiRuntimeConfig) -> QueryTextBuilder:
    return build_query_text_builder(config.query_builder)


def _read_cache_metadata(config: ApiRuntimeConfig) -> EmbeddedExerciseCacheMetadata:
    return read_embedded_cache_metadata(config.embedded_cache)


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
