from importlib import import_module
import inspect
from typing import get_type_hints


def get_embedding_provider_protocol():
    module = import_module("calisthenics_recommender.ports.embedding_provider")
    return getattr(module, "EmbeddingProvider")


def test_embedding_provider_is_a_runtime_checkable_protocol_with_embed():
    EmbeddingProvider = get_embedding_provider_protocol()

    assert getattr(EmbeddingProvider, "_is_protocol", False) is True
    assert getattr(EmbeddingProvider, "_is_runtime_protocol", False) is True
    assert list(inspect.signature(EmbeddingProvider.embed).parameters) == ["self", "text"]
    assert get_type_hints(EmbeddingProvider.embed) == {
        "text": str,
        "return": list[float],
    }


def test_embedding_provider_can_be_implemented_by_a_small_fake_class():
    EmbeddingProvider = get_embedding_provider_protocol()

    class SmallFakeEmbeddingProvider:
        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0] if text == "pull-up query" else [0.0, 1.0, 0.0]

    provider = SmallFakeEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)
    assert provider.embed("pull-up query") == [1.0, 0.0, 0.0]
