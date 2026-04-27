from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Mapping

from fastapi import FastAPI

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
from calisthenics_recommender.api.app import create_app
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider
from calisthenics_recommender.ports.embedded_exercise_repository import (
    EmbeddedExerciseRepository,
)


_CACHE_PATH_ENV_VAR = "CALISTHENICS_RECOMMENDER_CACHE_PATH"
_EMBEDDING_PROVIDER_ENV_VAR = "CALISTHENICS_RECOMMENDER_EMBEDDING_PROVIDER"
_EMBEDDING_MODEL_ENV_VAR = "CALISTHENICS_RECOMMENDER_EMBEDDING_MODEL"
_QUERY_PREFIX_ENV_VAR = "CALISTHENICS_RECOMMENDER_QUERY_PREFIX"
_SUPPORTED_EMBEDDING_PROVIDERS = (
    "local-deterministic",
    "sentence-transformer",
)


@dataclass(frozen=True)
class ApiRuntimeConfig:
    cache_path: Path
    embedding_provider: Literal["local-deterministic", "sentence-transformer"]
    embedding_model: str | None = None
    query_prefix: str = ""


def read_runtime_config_from_env(
    environ: Mapping[str, str] | None = None,
) -> ApiRuntimeConfig:
    env = os.environ if environ is None else environ

    cache_path = Path(_require_env_var(env, _CACHE_PATH_ENV_VAR))
    embedding_provider = _require_env_var(env, _EMBEDDING_PROVIDER_ENV_VAR)
    if embedding_provider not in _SUPPORTED_EMBEDDING_PROVIDERS:
        supported_values = ", ".join(_SUPPORTED_EMBEDDING_PROVIDERS)
        raise ValueError(
            f"{_EMBEDDING_PROVIDER_ENV_VAR} must be one of: {supported_values}"
        )

    embedding_model = _read_optional_env_var(env, _EMBEDDING_MODEL_ENV_VAR)
    query_prefix = env.get(_QUERY_PREFIX_ENV_VAR, "")
    return ApiRuntimeConfig(
        cache_path=cache_path,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        query_prefix=query_prefix,
    )


def create_configured_app_from_env(
    environ: Mapping[str, str] | None = None,
) -> FastAPI:
    config = read_runtime_config_from_env(environ)
    metadata = _read_cache_metadata(config.cache_path)
    embedded_exercise_repository = _build_embedded_exercise_repository(config)
    embedding_provider = _build_embedding_provider(
        config=config,
        metadata=metadata,
    )
    return create_app(
        embedded_exercise_repository=embedded_exercise_repository,
        embedding_provider=embedding_provider,
    )


def _build_embedded_exercise_repository(
    config: ApiRuntimeConfig,
) -> EmbeddedExerciseRepository:
    return LocalEmbeddedExerciseRepository(config.cache_path)


def _build_embedding_provider(
    *,
    config: ApiRuntimeConfig,
    metadata: EmbeddedExerciseCacheMetadata,
) -> EmbeddingProvider:
    if config.embedding_provider == "sentence-transformer":
        model_name = config.embedding_model or metadata.embedding_model
        embedding_provider = SentenceTransformerEmbeddingProvider(
            model_name=model_name,
            text_prefix=config.query_prefix,
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


def _read_cache_metadata(cache_path: Path) -> EmbeddedExerciseCacheMetadata:
    try:
        return read_embedded_exercise_cache_metadata(cache_path)
    except FileNotFoundError as error:
        raise ValueError(
            f"Embedded exercise cache file does not exist: {cache_path}"
        ) from error
    except OSError as error:
        raise ValueError(
            f"Unable to read embedded exercise cache file: {cache_path}"
        ) from error
    except ValueError as error:
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
