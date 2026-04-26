from __future__ import annotations

from typing import Any


class SentenceTransformerEmbeddingProvider:
    DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        text_prefix: str = "",
        normalize_embeddings: bool = True,
        model: object | None = None,
    ) -> None:
        self._model_name = model_name
        self._text_prefix = text_prefix
        self._normalize_embeddings = normalize_embeddings
        self._model = model

    def embed(self, text: str) -> list[float]:
        prefixed_text = f"{self._text_prefix}{text}"
        raw_embedding = self._get_model().encode(
            prefixed_text,
            normalize_embeddings=self._normalize_embeddings,
        )
        return _coerce_embedding_to_float_list(raw_embedding)

    def get_embedding_dimension(self) -> int:
        dimension = self._get_model().get_sentence_embedding_dimension()
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("sentence transformer embedding dimension must be a positive integer")
        return dimension

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)

        return self._model


def _coerce_embedding_to_float_list(raw_embedding: object) -> list[float]:
    normalized_embedding = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else raw_embedding
    if not isinstance(normalized_embedding, list | tuple):
        raise ValueError("embedding output must be array-like")

    values: list[float] = []
    for value in normalized_embedding:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("embedding output must contain only numeric values")
        values.append(float(value))

    return values
