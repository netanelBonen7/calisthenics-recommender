from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any, Literal


_SUPPORTED_EMBEDDED_CACHE_BACKENDS = ("jsonl", "sqlite")
_SUPPORTED_EMBEDDING_PROVIDERS = (
    "local-deterministic",
    "sentence-transformer",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class EmbeddedCacheConfig:
    backend: Literal["jsonl", "sqlite"]
    path: Path


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: Literal["local-deterministic", "sentence-transformer"]
    model: str | None = None
    query_prefix: str = ""


@dataclass(frozen=True)
class ApiRuntimeConfig:
    embedded_cache: EmbeddedCacheConfig
    embedding: EmbeddingConfig


def load_api_runtime_config(config_path: Path | str) -> ApiRuntimeConfig:
    normalized_config_path = Path(config_path)
    if not normalized_config_path.exists():
        raise ConfigError(f"Config file does not exist: {normalized_config_path}")

    try:
        config_data = tomllib.loads(
            normalized_config_path.read_text(encoding="utf-8")
        )
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(
            f"Invalid TOML config at {normalized_config_path}: {error}"
        ) from error
    except OSError as error:
        raise ConfigError(
            f"Unable to read config file: {normalized_config_path}"
        ) from error

    embedded_cache_section = _require_table(
        config_data,
        section_name="embedded_cache",
        config_path=normalized_config_path,
    )
    embedding_section = _require_table(
        config_data,
        section_name="embedding",
        config_path=normalized_config_path,
    )

    embedded_cache_backend = _require_literal_string(
        embedded_cache_section,
        field_name="[embedded_cache].backend",
        config_path=normalized_config_path,
        supported_values=_SUPPORTED_EMBEDDED_CACHE_BACKENDS,
    )
    embedded_cache_path = _require_path(
        embedded_cache_section,
        field_name="[embedded_cache].path",
        config_path=normalized_config_path,
    )
    embedding_provider = _require_literal_string(
        embedding_section,
        field_name="[embedding].provider",
        config_path=normalized_config_path,
        supported_values=_SUPPORTED_EMBEDDING_PROVIDERS,
    )
    embedding_model = _read_optional_string(
        embedding_section,
        field_name="[embedding].model",
        config_path=normalized_config_path,
    )
    query_prefix = _read_optional_string(
        embedding_section,
        field_name="[embedding].query_prefix",
        config_path=normalized_config_path,
        preserve_whitespace=True,
    )

    return ApiRuntimeConfig(
        embedded_cache=EmbeddedCacheConfig(
            backend=embedded_cache_backend,
            path=embedded_cache_path,
        ),
        embedding=EmbeddingConfig(
            provider=embedding_provider,
            model=embedding_model,
            query_prefix="" if query_prefix is None else query_prefix,
        ),
    )


def _require_table(
    config_data: Any,
    *,
    section_name: str,
    config_path: Path,
) -> dict[str, Any]:
    if not isinstance(config_data, dict):
        raise ConfigError(f"Invalid config at {config_path}: [{section_name}] section is required")

    section = config_data.get(section_name)
    if not isinstance(section, dict):
        raise ConfigError(f"Invalid config at {config_path}: [{section_name}] section is required")

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
