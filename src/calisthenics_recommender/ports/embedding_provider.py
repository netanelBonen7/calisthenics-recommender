from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        ...
