from importlib import import_module
from pathlib import Path
import socket

import pytest


def get_embedding_provider_protocol():
    module = import_module("calisthenics_recommender.ports.embedding_provider")
    return getattr(module, "EmbeddingProvider")


def get_fake_embedding_provider():
    module = import_module("calisthenics_recommender.adapters.fake_embedding_provider")
    return getattr(module, "FakeEmbeddingProvider")


def test_fake_embedding_provider_implements_embedding_provider_protocol():
    EmbeddingProvider = get_embedding_provider_protocol()
    FakeEmbeddingProvider = get_fake_embedding_provider()

    provider = FakeEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)


@pytest.mark.parametrize(
    ("text", "expected_vector"),
    [
        ("pull-up query", [1.0, 0.0, 0.0]),
        ("pull-up exercise", [0.9, 0.1, 0.0]),
        ("push-up exercise", [0.0, 1.0, 0.0]),
        ("core exercise", [0.0, 0.1, 0.9]),
    ],
)
def test_fake_embedding_provider_returns_expected_deterministic_vectors(
    text, expected_vector
):
    FakeEmbeddingProvider = get_fake_embedding_provider()
    provider = FakeEmbeddingProvider()

    assert provider.embed(text) == expected_vector
    assert provider.embed(text) == expected_vector


def test_fake_embedding_provider_raises_a_clear_error_for_unknown_text():
    FakeEmbeddingProvider = get_fake_embedding_provider()
    provider = FakeEmbeddingProvider()

    with pytest.raises(KeyError, match="unknown|Unknown"):
        provider.embed("handstand query")


def test_fake_embedding_provider_is_pure_and_does_not_touch_files_or_network(
    monkeypatch,
):
    FakeEmbeddingProvider = get_fake_embedding_provider()
    provider = FakeEmbeddingProvider()

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    assert provider.embed("pull-up query") == [1.0, 0.0, 0.0]
