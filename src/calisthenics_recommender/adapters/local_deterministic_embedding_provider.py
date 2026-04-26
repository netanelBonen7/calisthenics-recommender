from __future__ import annotations

import hashlib


class LocalDeterministicEmbeddingProvider:
    """Local fake embedding provider for development-only cache and demo flows."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        return [
            _deterministic_component(text=text, index=index)
            for index in range(self._dimension)
        ]


def _deterministic_component(text: str, index: int) -> float:
    digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer % 1_000_000 + 1) / 1_000_001
