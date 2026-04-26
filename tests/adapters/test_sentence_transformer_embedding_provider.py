from importlib import import_module
import sys
from types import SimpleNamespace


class FakeArray:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class RecordingModel:
    def __init__(self, output) -> None:
        self._output = output
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, normalize_embeddings: bool = True):
        self.calls.append((text, normalize_embeddings))
        return self._output


def get_embedding_provider_protocol():
    module = import_module("calisthenics_recommender.ports.embedding_provider")
    return getattr(module, "EmbeddingProvider")


def get_sentence_transformer_embedding_provider():
    module = import_module(
        "calisthenics_recommender.adapters.sentence_transformer_embedding_provider"
    )
    return getattr(module, "SentenceTransformerEmbeddingProvider")


def test_sentence_transformer_embedding_provider_implements_embedding_provider_protocol():
    EmbeddingProvider = get_embedding_provider_protocol()
    SentenceTransformerEmbeddingProvider = get_sentence_transformer_embedding_provider()

    provider = SentenceTransformerEmbeddingProvider(model=RecordingModel([1.0, 2.0]))

    assert isinstance(provider, EmbeddingProvider)


def test_sentence_transformer_embedding_provider_prefixes_text_and_returns_list_of_floats():
    SentenceTransformerEmbeddingProvider = get_sentence_transformer_embedding_provider()
    model = RecordingModel(FakeArray([1, 2.5, 3]))
    provider = SentenceTransformerEmbeddingProvider(
        model=model,
        text_prefix="query: ",
        normalize_embeddings=True,
    )

    embedding = provider.embed("pull-up strength")

    assert embedding == [1.0, 2.5, 3.0]
    assert model.calls == [("query: pull-up strength", True)]


def test_sentence_transformer_embedding_provider_lazy_loads_custom_model_name(
    monkeypatch,
):
    SentenceTransformerEmbeddingProvider = get_sentence_transformer_embedding_provider()
    loader_calls: list[str] = []
    encode_calls: list[tuple[str, bool]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            loader_calls.append(model_name)

        def encode(self, text: str, normalize_embeddings: bool = True):
            encode_calls.append((text, normalize_embeddings))
            return [0.1, 0.2]

        def get_sentence_embedding_dimension(self) -> int:
            return 2

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    provider = SentenceTransformerEmbeddingProvider(
        model_name="custom/local-model",
        text_prefix="doc: ",
        normalize_embeddings=False,
    )

    assert loader_calls == []

    embedding = provider.embed("Body Row")

    assert embedding == [0.1, 0.2]
    assert loader_calls == ["custom/local-model"]
    assert encode_calls == [("doc: Body Row", False)]
    assert provider.get_embedding_dimension() == 2
