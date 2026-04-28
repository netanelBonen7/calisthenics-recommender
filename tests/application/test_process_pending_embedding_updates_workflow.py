import pytest

from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
)
from calisthenics_recommender.application.exercise_text_builder import (
    V1ExerciseTextBuilder,
    build_exercise_text,
)
from calisthenics_recommender.application.process_pending_embedding_updates_workflow import (
    ProcessPendingEmbeddingUpdatesWorkflow,
)
from calisthenics_recommender.domain.exercise import Exercise
from calisthenics_recommender.domain.pending_embedding_update import (
    PendingEmbeddingUpdate,
)


def make_exercise(**overrides):
    payload = {
        "exercise_id": "pull-up",
        "name": "Pull Up",
        "description": "A strict vertical pulling movement.",
        "muscle_groups": ["Back", "Biceps"],
        "families": ["Pull-up"],
        "materials": ["Bar"],
        "categories": ["Upper Body Pull"],
    }
    payload.update(overrides)
    return Exercise(**payload)


def make_metadata(**overrides):
    payload = {
        "embedding_model": "test-model",
        "embedding_dimension": 3,
        "text_builder_version": "v1",
    }
    payload.update(overrides)
    return EmbeddedExerciseCacheMetadata(**payload)


class InMemoryPendingRepository:
    def __init__(self, updates):
        self.updates = list(updates)
        self.failures = []
        self.iter_called = False

    def iter_pending_updates(self, limit=None):
        self.iter_called = True
        updates = self.updates if limit is None else self.updates[:limit]
        return iter(list(updates))

    def mark_processed(self, update):
        for index, pending_update in enumerate(self.updates):
            if (
                pending_update.exercise_id == update.exercise_id
                and pending_update.version == update.version
            ):
                del self.updates[index]
                return True
        return False

    def record_failure(self, update, error_message):
        self.failures.append((update, error_message))
        return True

    def count_pending_updates(self):
        return len(self.updates)


class InMemoryExerciseLookupRepository:
    def __init__(self, exercises):
        self._exercises = {
            exercise.exercise_id: exercise for exercise in exercises
        }

    def get_by_exercise_id(self, exercise_id):
        return self._exercises.get(exercise_id)


class RecordingEmbeddingProvider:
    def __init__(self, embeddings=None, error=None):
        self._embeddings = {} if embeddings is None else dict(embeddings)
        self._error = error
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return self._embeddings[text]


class RecordingExerciseTextBuilder:
    def __init__(self, text: str):
        self._text = text
        self.calls = []

    def build(self, exercise):
        self.calls.append(exercise)
        return self._text


class RecordingCacheUpdater:
    def __init__(self):
        self.upserts = []
        self.deletes = []

    def upsert_embedded_exercise(self, embedded_exercise, metadata):
        self.upserts.append((embedded_exercise, metadata))

    def delete_embedded_exercise(self, exercise_id):
        self.deletes.append(exercise_id)


def build_workflow(
    pending_repository,
    exercise_repository,
    embedding_provider,
    cache_updater,
    *,
    expected_metadata=None,
    actual_metadata=None,
):
    metadata = make_metadata() if expected_metadata is None else expected_metadata
    return ProcessPendingEmbeddingUpdatesWorkflow(
        pending_update_repository=pending_repository,
        exercise_repository=exercise_repository,
        embedding_provider=embedding_provider,
        exercise_text_builder=V1ExerciseTextBuilder(),
        cache_updater=cache_updater,
        expected_metadata=metadata,
        actual_metadata=metadata if actual_metadata is None else actual_metadata,
    )


def test_process_pending_upsert_builds_embedding_and_updates_cache():
    exercise = make_exercise()
    exercise_text = build_exercise_text(exercise)
    pending_repository = InMemoryPendingRepository(
        [PendingEmbeddingUpdate("pull-up", "upsert", 1)]
    )
    embedding_provider = RecordingEmbeddingProvider(
        embeddings={exercise_text: [1.0, 0.0, 0.0]}
    )
    cache_updater = RecordingCacheUpdater()

    result = build_workflow(
        pending_repository,
        InMemoryExerciseLookupRepository([exercise]),
        embedding_provider,
        cache_updater,
    ).process()

    assert result.processed_count == 1
    assert result.failed_count == 0
    assert result.remaining_count == 0
    assert embedding_provider.calls == [exercise_text]
    assert cache_updater.upserts[0][0].exercise == exercise
    assert cache_updater.upserts[0][0].embedding == (1.0, 0.0, 0.0)


def test_process_pending_upsert_uses_the_injected_exercise_text_builder():
    exercise = make_exercise()
    custom_text = "custom exercise text"
    pending_repository = InMemoryPendingRepository(
        [PendingEmbeddingUpdate("pull-up", "upsert", 1)]
    )
    exercise_text_builder = RecordingExerciseTextBuilder(custom_text)
    embedding_provider = RecordingEmbeddingProvider(
        embeddings={custom_text: [1.0, 0.0, 0.0]}
    )
    cache_updater = RecordingCacheUpdater()

    ProcessPendingEmbeddingUpdatesWorkflow(
        pending_update_repository=pending_repository,
        exercise_repository=InMemoryExerciseLookupRepository([exercise]),
        embedding_provider=embedding_provider,
        exercise_text_builder=exercise_text_builder,
        cache_updater=cache_updater,
        expected_metadata=make_metadata(),
        actual_metadata=make_metadata(),
    ).process()

    assert exercise_text_builder.calls == [exercise]
    assert embedding_provider.calls == [custom_text]


def test_process_pending_delete_removes_cache_entry():
    pending_repository = InMemoryPendingRepository(
        [PendingEmbeddingUpdate("pull-up", "delete", 1)]
    )
    cache_updater = RecordingCacheUpdater()

    result = build_workflow(
        pending_repository,
        InMemoryExerciseLookupRepository([]),
        RecordingEmbeddingProvider(),
        cache_updater,
    ).process()

    assert result.processed_count == 1
    assert cache_updater.deletes == ["pull-up"]
    assert cache_updater.upserts == []


def test_process_pending_upsert_for_missing_raw_exercise_deletes_cache_entry():
    pending_repository = InMemoryPendingRepository(
        [PendingEmbeddingUpdate("missing-exercise", "upsert", 1)]
    )
    cache_updater = RecordingCacheUpdater()

    build_workflow(
        pending_repository,
        InMemoryExerciseLookupRepository([]),
        RecordingEmbeddingProvider(),
        cache_updater,
    ).process()

    assert cache_updater.deletes == ["missing-exercise"]


def test_process_pending_failure_records_error_and_keeps_pending_update():
    exercise = make_exercise()
    pending_update = PendingEmbeddingUpdate("pull-up", "upsert", 1)
    pending_repository = InMemoryPendingRepository([pending_update])

    result = build_workflow(
        pending_repository,
        InMemoryExerciseLookupRepository([exercise]),
        RecordingEmbeddingProvider(error=RuntimeError("embedding failed")),
        RecordingCacheUpdater(),
    ).process()

    assert result.processed_count == 0
    assert result.failed_count == 1
    assert result.remaining_count == 1
    assert pending_repository.failures == [(pending_update, "embedding failed")]


def test_metadata_mismatch_fails_before_reading_pending_updates():
    pending_repository = InMemoryPendingRepository(
        [PendingEmbeddingUpdate("pull-up", "upsert", 1)]
    )

    with pytest.raises(ValueError, match=r"embedding_model|full cache rebuild"):
        build_workflow(
            pending_repository,
            InMemoryExerciseLookupRepository([]),
            RecordingEmbeddingProvider(),
            RecordingCacheUpdater(),
            actual_metadata=make_metadata(embedding_model="other-model"),
        ).process()

    assert pending_repository.iter_called is False
