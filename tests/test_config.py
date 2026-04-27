from __future__ import annotations

from pathlib import Path

import pytest

from calisthenics_recommender.config import ConfigError, load_api_runtime_config


def write_config(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def test_load_api_runtime_config_loads_valid_jsonl_config(tmp_path):
    config_path = tmp_path / "runtime.toml"
    write_config(
        config_path,
        (
            "[embedded_cache]\n"
            'backend = "jsonl"\n'
            'path = "data/cache/embedded_exercises.jsonl"\n'
            "\n"
            "[embedding]\n"
            'provider = "local-deterministic"\n'
            'model = "fake-hash-v1"\n'
            'query_prefix = "query: "\n'
        ),
    )

    config = load_api_runtime_config(config_path)

    assert config.embedded_cache.backend == "jsonl"
    assert config.embedded_cache.path == (
        tmp_path / "data" / "cache" / "embedded_exercises.jsonl"
    )
    assert config.embedding.provider == "local-deterministic"
    assert config.embedding.model == "fake-hash-v1"
    assert config.embedding.query_prefix == "query: "


def test_load_api_runtime_config_loads_valid_sqlite_config(tmp_path):
    config_path = tmp_path / "runtime.toml"
    write_config(
        config_path,
        (
            "[embedded_cache]\n"
            'backend = "sqlite"\n'
            'path = "data/cache/embedded_exercises.sqlite"\n'
            "\n"
            "[embedding]\n"
            'provider = "sentence-transformer"\n'
            'model = "custom/local-model"\n'
        ),
    )

    config = load_api_runtime_config(config_path)

    assert config.embedded_cache.backend == "sqlite"
    assert config.embedding.provider == "sentence-transformer"
    assert config.embedding.model == "custom/local-model"


def test_load_api_runtime_config_resolves_relative_paths_against_config_file(tmp_path):
    config_path = tmp_path / "config" / "runtime.toml"
    write_config(
        config_path,
        (
            "[embedded_cache]\n"
            'backend = "jsonl"\n'
            'path = "../data/cache/embedded_exercises.jsonl"\n'
            "\n"
            "[embedding]\n"
            'provider = "local-deterministic"\n'
        ),
    )

    config = load_api_runtime_config(config_path)

    assert config.embedded_cache.path == (
        tmp_path / "config" / ".." / "data" / "cache" / "embedded_exercises.jsonl"
    ).resolve()


def test_load_api_runtime_config_rejects_missing_config_file(tmp_path):
    config_path = tmp_path / "missing.toml"

    with pytest.raises(ConfigError, match=r"Config file does not exist: .*missing\.toml"):
        load_api_runtime_config(config_path)


def test_load_api_runtime_config_rejects_malformed_toml(tmp_path):
    config_path = tmp_path / "runtime.toml"
    write_config(
        config_path,
        "[embedded_cache\nbackend = \"jsonl\"\n",
    )

    with pytest.raises(ConfigError, match=r"Invalid TOML config at .*runtime\.toml:"):
        load_api_runtime_config(config_path)


@pytest.mark.parametrize(
    ("contents", "expected_message"),
    [
        (
            (
                "[embedding]\n"
                'provider = "local-deterministic"\n'
            ),
            r"Invalid config at .*: \[embedded_cache\] section is required",
        ),
        (
            (
                "[embedded_cache]\n"
                'backend = "jsonl"\n'
                'path = "embedded_exercises.jsonl"\n'
            ),
            r"Invalid config at .*: \[embedding\] section is required",
        ),
        (
            (
                "[embedded_cache]\n"
                'path = "embedded_exercises.jsonl"\n'
                "\n"
                "[embedding]\n"
                'provider = "local-deterministic"\n'
            ),
            r"Invalid config at .*: \[embedded_cache\]\.backend is required",
        ),
        (
            (
                "[embedded_cache]\n"
                'backend = "jsonl"\n'
                "\n"
                "[embedding]\n"
                'provider = "local-deterministic"\n'
            ),
            r"Invalid config at .*: \[embedded_cache\]\.path is required",
        ),
        (
            (
                "[embedded_cache]\n"
                'backend = "jsonl"\n'
                'path = "embedded_exercises.jsonl"\n'
                "\n"
                "[embedding]\n"
            ),
            r"Invalid config at .*: \[embedding\]\.provider is required",
        ),
    ],
)
def test_load_api_runtime_config_rejects_missing_required_sections_and_fields(
    tmp_path, contents, expected_message
):
    config_path = tmp_path / "runtime.toml"
    write_config(config_path, contents)

    with pytest.raises(ConfigError, match=expected_message):
        load_api_runtime_config(config_path)


def test_load_api_runtime_config_rejects_invalid_backend(tmp_path):
    config_path = tmp_path / "runtime.toml"
    write_config(
        config_path,
        (
            "[embedded_cache]\n"
            'backend = "unsupported"\n'
            'path = "embedded_exercises.jsonl"\n'
            "\n"
            "[embedding]\n"
            'provider = "local-deterministic"\n'
        ),
    )

    with pytest.raises(
        ConfigError,
        match=r"Invalid config at .*: \[embedded_cache\]\.backend must be one of: jsonl, sqlite",
    ):
        load_api_runtime_config(config_path)


def test_load_api_runtime_config_rejects_invalid_provider(tmp_path):
    config_path = tmp_path / "runtime.toml"
    write_config(
        config_path,
        (
            "[embedded_cache]\n"
            'backend = "jsonl"\n'
            'path = "embedded_exercises.jsonl"\n'
            "\n"
            "[embedding]\n"
            'provider = "unsupported"\n'
        ),
    )

    with pytest.raises(
        ConfigError,
        match=r"Invalid config at .*: \[embedding\]\.provider must be one of: local-deterministic, sentence-transformer",
    ):
        load_api_runtime_config(config_path)


def test_load_api_runtime_config_defaults_query_prefix_to_empty_string(tmp_path):
    config_path = tmp_path / "runtime.toml"
    write_config(
        config_path,
        (
            "[embedded_cache]\n"
            'backend = "jsonl"\n'
            'path = "embedded_exercises.jsonl"\n'
            "\n"
            "[embedding]\n"
            'provider = "local-deterministic"\n'
        ),
    )

    config = load_api_runtime_config(config_path)

    assert config.embedding.query_prefix == ""
