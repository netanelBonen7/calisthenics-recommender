from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from calisthenics_recommender.domain.pending_embedding_update import (
    PendingEmbeddingUpdate,
)


@runtime_checkable
class PendingEmbeddingUpdateRepository(Protocol):
    def iter_pending_updates(
        self, limit: int | None = None
    ) -> Iterable[PendingEmbeddingUpdate]:
        ...

    def mark_processed(self, update: PendingEmbeddingUpdate) -> bool:
        ...

    def record_failure(
        self, update: PendingEmbeddingUpdate, error_message: str
    ) -> bool:
        ...

    def count_pending_updates(self) -> int:
        ...
