from __future__ import annotations

from dataclasses import dataclass

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
)
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.ports.embedded_exercise_cache_updater import (
    EmbeddedExerciseCacheUpdater,
)
from calisthenics_recommender.ports.embedding_provider import EmbeddingProvider
from calisthenics_recommender.ports.exercise_text_builder import (
    ExerciseTextBuilder,
)
from calisthenics_recommender.ports.exercise_lookup_repository import (
    ExerciseLookupRepository,
)
from calisthenics_recommender.ports.pending_embedding_update_repository import (
    PendingEmbeddingUpdateRepository,
)


@dataclass(frozen=True)
class ProcessPendingEmbeddingUpdatesResult:
    seen_count: int
    processed_count: int
    failed_count: int
    remaining_count: int


class ProcessPendingEmbeddingUpdatesWorkflow:
    def __init__(
        self,
        *,
        pending_update_repository: PendingEmbeddingUpdateRepository,
        exercise_repository: ExerciseLookupRepository,
        embedding_provider: EmbeddingProvider,
        exercise_text_builder: ExerciseTextBuilder,
        cache_updater: EmbeddedExerciseCacheUpdater,
        expected_metadata: EmbeddedExerciseCacheMetadata,
        actual_metadata: EmbeddedExerciseCacheMetadata,
    ) -> None:
        self._pending_update_repository = pending_update_repository
        self._exercise_repository = exercise_repository
        self._embedding_provider = embedding_provider
        self._exercise_text_builder = exercise_text_builder
        self._cache_updater = cache_updater
        self._expected_metadata = expected_metadata
        self._actual_metadata = actual_metadata

    def process(
        self, limit: int | None = None
    ) -> ProcessPendingEmbeddingUpdatesResult:
        validate_embedded_cache_metadata(
            expected_metadata=self._expected_metadata,
            actual_metadata=self._actual_metadata,
        )

        seen_count = 0
        processed_count = 0
        failed_count = 0

        for update in self._pending_update_repository.iter_pending_updates(limit):
            seen_count += 1
            try:
                if update.operation == "upsert":
                    exercise = self._exercise_repository.get_by_exercise_id(
                        update.exercise_id
                    )
                    if exercise is None:
                        self._cache_updater.delete_embedded_exercise(update.exercise_id)
                    else:
                        exercise_text = self._exercise_text_builder.build(exercise)
                        embedding = self._embedding_provider.embed(exercise_text)
                        self._cache_updater.upsert_embedded_exercise(
                            EmbeddedExercise(exercise=exercise, embedding=embedding),
                            self._expected_metadata,
                        )
                else:
                    self._cache_updater.delete_embedded_exercise(update.exercise_id)

                self._pending_update_repository.mark_processed(update)
                processed_count += 1
            except Exception as error:
                failed_count += 1
                self._pending_update_repository.record_failure(update, str(error))

        return ProcessPendingEmbeddingUpdatesResult(
            seen_count=seen_count,
            processed_count=processed_count,
            failed_count=failed_count,
            remaining_count=self._pending_update_repository.count_pending_updates(),
        )


def validate_embedded_cache_metadata(
    *,
    expected_metadata: EmbeddedExerciseCacheMetadata,
    actual_metadata: EmbeddedExerciseCacheMetadata,
) -> None:
    if not isinstance(expected_metadata, EmbeddedExerciseCacheMetadata):
        raise ValueError("expected_metadata must be an EmbeddedExerciseCacheMetadata")
    if not isinstance(actual_metadata, EmbeddedExerciseCacheMetadata):
        raise ValueError("actual_metadata must be an EmbeddedExerciseCacheMetadata")

    mismatches: list[str] = []
    if actual_metadata.embedding_model != expected_metadata.embedding_model:
        mismatches.append("embedding_model")
    if actual_metadata.embedding_dimension != expected_metadata.embedding_dimension:
        mismatches.append("embedding_dimension")
    if actual_metadata.text_builder_version != expected_metadata.text_builder_version:
        mismatches.append("text_builder_version")

    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise ValueError(
            "Embedded cache metadata is incompatible with the current embedding "
            f"config ({mismatch_text}); run a full cache rebuild"
        )
