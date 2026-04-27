from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Literal


_SUPPORTED_RAW_EXERCISES_BACKENDS = ("csv", "sqlite")
_SUPPORTED_EMBEDDED_CACHE_BACKENDS = ("jsonl", "sqlite")
_SUPPORTED_EMBEDDING_PROVIDERS = (
    "local-deterministic",
    "sentence-transformer",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RawExercisesConfig:
    backend: Literal["csv", "sqlite"]
    path: Path


@dataclass(frozen=True)
class EmbeddedCacheConfig:
    backend: Literal["jsonl", "sqlite"]
    path: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: Literal["local-deterministic", "sentence-transformer"]
    model: str | None = None
    dimension: int | None = None
    query_prefix: str = ""
    text_prefix: str = ""
    text_builder_version: str | None = None


@dataclass(frozen=True)
class RecommenderConfig:
    raw_exercises: RawExercisesConfig | None = None
    embedded_cache: EmbeddedCacheConfig | None = None
    embedding: EmbeddingConfig | None = None


@dataclass(frozen=True)
class ApiRuntimeConfig:
    embedded_cache: EmbeddedCacheConfig
    embedding: EmbeddingConfig


def load_recommender_config(config_path: Path | str) -> RecommenderConfig:
    normalized_config_path = Path(config_path)
    config_data = _load_config_data(normalized_config_path)

    return RecommenderConfig(
        raw_exercises=_read_optional_raw_exercises_config(
            config_data,
            config_path=normalized_config_path,
        ),
        embedded_cache=_read_optional_embedded_cache_config(
            config_data,
            config_path=normalized_config_path,
        ),
        embedding=_read_optional_embedding_config(
            config_data,
            config_path=normalized_config_path,
        ),
    )


def load_api_runtime_config(config_path: Path | str) -> ApiRuntimeConfig:
    normalized_config_path = Path(config_path)
    config = load_recommender_config(normalized_config_path)

    if config.embedded_cache is None:
        raise ConfigError(
            f"Invalid config at {normalized_config_path}: [embedded_cache] section is required"
        )
    if config.embedding is None:
        raise ConfigError(
            f"Invalid config at {normalized_config_path}: [embedding] section is required"
        )

    return ApiRuntimeConfig(
        embedded_cache=config.embedded_cache,
        embedding=config.embedding,
    )


def _load_config_data(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    try:
        config_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"Invalid TOML config at {config_path}: {error}") from error
    except OSError as error:
        raise ConfigError(f"Unable to read config file: {config_path}") from error

    if not isinstance(config_data, dict):
        raise ConfigError(f"Invalid config at {config_path}: expected top-level TOML table")
    return config_data


def _read_optional_raw_exercises_config(
    config_data: dict[str, Any],
    *,
    config_path: Path,
) -> RawExercisesConfig | None:
    section = _read_optional_table(
        config_data,
        section_name="raw_exercises",
        config_path=config_path,
    )
    if section is None:
        return None

    backend = _require_literal_string(
        section,
        field_name="[raw_exercises].backend",
        config_path=config_path,
        supported_values=_SUPPORTED_RAW_EXERCISES_BACKENDS,
    )
    path_field_name = (
        "[raw_exercises].csv_path"
        if backend == "csv"
        else "[raw_exercises].sqlite_path"
    )
    return RawExercisesConfig(
        backend=backend,
        path=_require_path(
            section,
            field_name=path_field_name,
            config_path=config_path,
        ),
    )


def _read_optional_embedded_cache_config(
    config_data: dict[str, Any],
    *,
    config_path: Path,
) -> EmbeddedCacheConfig | None:
    section = _read_optional_table(
        config_data,
        section_name="embedded_cache",
        config_path=config_path,
    )
    if section is None:
        return None

    return EmbeddedCacheConfig(
        backend=_require_literal_string(
            section,
            field_name="[embedded_cache].backend",
            config_path=config_path,
            supported_values=_SUPPORTED_EMBEDDED_CACHE_BACKENDS,
        ),
        path=_require_path(
            section,
            field_name="[embedded_cache].path",
            config_path=config_path,
        ),
    )


def _read_optional_embedding_config(
    config_data: dict[str, Any],
    *,
    config_path: Path,
) -> EmbeddingConfig | None:
    section = _read_optional_table(
        config_data,
        section_name="embedding",
        config_path=config_path,
    )
    if section is None:
        return None

    query_prefix = _read_optional_string(
        section,
        field_name="[embedding].query_prefix",
        config_path=config_path,
        preserve_whitespace=True,
    )
    text_prefix = _read_optional_string(
        section,
        field_name="[embedding].text_prefix",
        config_path=config_path,
        preserve_whitespace=True,
    )

    return EmbeddingConfig(
        provider=_require_literal_string(
            section,
            field_name="[embedding].provider",
            config_path=config_path,
            supported_values=_SUPPORTED_EMBEDDING_PROVIDERS,
        ),
        model=_read_optional_string(
            section,
            field_name="[embedding].model",
            config_path=config_path,
        ),
        dimension=_read_optional_positive_int(
            section,
            field_name="[embedding].dimension",
            config_path=config_path,
        ),
        query_prefix="" if query_prefix is None else query_prefix,
        text_prefix="" if text_prefix is None else text_prefix,
        text_builder_version=_read_optional_non_empty_string(
            section,
            field_name="[embedding].text_builder_version",
            config_path=config_path,
        ),
    )


def _read_optional_table(
    config_data: dict[str, Any],
    *,
    section_name: str,
    config_path: Path,
) -> dict[str, Any] | None:
    section = config_data.get(section_name)
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ConfigError(
            f"Invalid config at {config_path}: [{section_name}] section must be a table"
        )
    return section


def _require_literal_string(
    section: dict[str, Any],
    *,
    field_name: str,
    config_path: Path,
    supported_values: tuple[str, ...],
) -> str:
    value = _read_optional_string(
        section,
        field_name=field_name,
        config_path=config_path,
    )
    if value is None:
        raise ConfigError(f"Invalid config at {config_path}: {field_name} is required")
    if value not in supported_values:
        supported_values_text = ", ".join(supported_values)
        raise ConfigError(
            f"Invalid config at {config_path}: {field_name} must be one of: {supported_values_text}"
        )
    return value


def _require_path(
    section: dict[str, Any],
    *,
    field_name: str,
    config_path: Path,
) -> Path:
    value = _read_optional_string(
        section,
        field_name=field_name,
        config_path=config_path,
    )
    if value is None:
        raise ConfigError(f"Invalid config at {config_path}: {field_name} is required")

    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (config_path.parent / raw_path).resolve()


def _read_optional_non_empty_string(
    section: dict[str, Any],
    *,
    field_name: str,
    config_path: Path,
) -> str | None:
    key = field_name.rsplit(".", maxsplit=1)[-1]
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(
            f"Invalid config at {config_path}: {field_name} must be a non-empty string"
        )

    normalized_value = value.strip()
    if not normalized_value:
        raise ConfigError(
            f"Invalid config at {config_path}: {field_name} must be a non-empty string"
        )
    return normalized_value


def _read_optional_string(
    section: dict[str, Any],
    *,
    field_name: str,
    config_path: Path,
    preserve_whitespace: bool = False,
) -> str | None:
    key = field_name.rsplit(".", maxsplit=1)[-1]
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(
            f"Invalid config at {config_path}: {field_name} must be a non-empty string"
        )

    normalized_value = value.strip()
    if not normalized_value:
        return None

    if preserve_whitespace:
        return value

    return normalized_value


def _read_optional_positive_int(
    section: dict[str, Any],
    *,
    field_name: str,
    config_path: Path,
) -> int | None:
    key = field_name.rsplit(".", maxsplit=1)[-1]
    value = section.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(
            f"Invalid config at {config_path}: {field_name} must be a positive integer"
        )
    return value
