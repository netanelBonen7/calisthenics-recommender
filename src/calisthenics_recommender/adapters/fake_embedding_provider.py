from collections.abc import Mapping


DEFAULT_EMBEDDINGS: dict[str, list[float]] = {
    "pull-up query": [1.0, 0.0, 0.0],
    "pull-up exercise": [0.9, 0.1, 0.0],
    "push-up exercise": [0.0, 1.0, 0.0],
    "core exercise": [0.0, 0.1, 0.9],
}


class FakeEmbeddingProvider:
    def __init__(self, embeddings: Mapping[str, list[float]] | None = None) -> None:
        source = DEFAULT_EMBEDDINGS if embeddings is None else embeddings
        self._embeddings = {text: list(vector) for text, vector in source.items()}

    def embed(self, text: str) -> list[float]:
        try:
            return list(self._embeddings[text])
        except KeyError as error:
            raise KeyError(f"Unknown text: {text}") from error
