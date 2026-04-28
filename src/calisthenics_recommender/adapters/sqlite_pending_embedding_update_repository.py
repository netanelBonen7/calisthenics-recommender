from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
import sqlite3

from calisthenics_recommender.domain.pending_embedding_update import (
    PendingEmbeddingUpdate,
)


class SQLitePendingEmbeddingUpdateRepository:
    def __init__(self, sqlite_path: Path | str) -> None:
        self._sqlite_path = Path(sqlite_path)

    def iter_pending_updates(
        self, limit: int | None = None
    ) -> Iterable[PendingEmbeddingUpdate]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than 0")
        return self._iter_pending_updates(limit)

    def _iter_pending_updates(
        self, limit: int | None = None
    ) -> Iterator[PendingEmbeddingUpdate]:
        query = """
            SELECT exercise_id, operation, version
            FROM pending_embedding_updates
            ORDER BY updated_at, exercise_id
        """
        parameters: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            parameters = (limit,)

        with sqlite3.connect(self._sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters)
            for row in rows:
                yield PendingEmbeddingUpdate(
                    exercise_id=row["exercise_id"],
                    operation=row["operation"],
                    version=row["version"],
                )

    def mark_processed(self, update: PendingEmbeddingUpdate) -> bool:
        with sqlite3.connect(self._sqlite_path) as connection:
            cursor = connection.execute(
                """
                DELETE FROM pending_embedding_updates
                WHERE exercise_id = ? AND version = ?
                """,
                (update.exercise_id, update.version),
            )
            return cursor.rowcount > 0

    def record_failure(
        self,
        update: PendingEmbeddingUpdate,
        error_message: str,
    ) -> bool:
        with sqlite3.connect(self._sqlite_path) as connection:
            cursor = connection.execute(
                """
                UPDATE pending_embedding_updates
                SET
                    attempt_count = attempt_count + 1,
                    last_attempted_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    last_error = ?
                WHERE exercise_id = ? AND version = ?
                """,
                (error_message, update.exercise_id, update.version),
            )
            return cursor.rowcount > 0

    def count_pending_updates(self) -> int:
        with sqlite3.connect(self._sqlite_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM pending_embedding_updates"
            ).fetchone()
        return int(row[0])
