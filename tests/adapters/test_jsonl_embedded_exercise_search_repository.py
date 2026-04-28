from importlib import import_module
from pathlib import Path
import socket
from typing import Iterable, Iterator

import pytest


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_model():
    module = import_module("calisthenics_recommender.domain.embedded_exercise")
    return getattr(module, "EmbeddedExercise")


def get_embedded_exercise_search_result_model():
    module = import_module(
        "calisthenics_recommender.domain.embedded_exercise_search_result"
    )
    return getattr(module, "EmbeddedExerciseSearchResult")


def get_embedded_exercise_search_repository_protocol():
    module = import_module(
        "calisthenics_recommender.ports.embedded_exercise_search_repository"
    )
    return getattr(module, "EmbeddedExerciseSearchRepository")


def get_jsonl_embedded_exercise_search_repository():
    module = import_module(
        "calisthenics_recommender.adapters.jsonl_embedded_exercise_search_repository"
    )
    return getattr(module, "JsonlEmbeddedExerciseSearchRepository")


def get_local_embedded_exercise_cache():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "LocalEmbeddedExerciseCache")


def get_local_embedded_exercise_repository():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "LocalEmbeddedExerciseRepository")


def get_cache_metadata_model():
    module = import_module(
        "calisthenics_recommender.adapters.local_embedded_exercise_cache"
    )
    return getattr(module, "EmbeddedExerciseCacheMetadata")


def get_retrieve_top_matches():
    module = import_module("calisthenics_recommender.application.retriever")
    return getattr(module, "retrieve_top_matches")


def get_exercise_matches_equipment():
    module = import_module("calisthenics_recommender.application.filters")
    return getattr(module, "exercise_matches_equipment")


def exercise_id_for(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def exercise_named(
    name: str,
    *,
    materials: list[str],
    embedding_family: str = "Pull-up",
    exercise_id: str | None = None,
):
    Exercise = get_exercise_model()
    return Exercise(
        exercise_id=exercise_id_for(name) if exercise_id is None else exercise_id,
        name=name,
        description=f"{name} description.",
        muscle_groups=["Back"],
        families=[embedding_family],
        materials=materials,
        categories=["Upper Body Pull"],
    )


def embedded_exercise_named(
    name: str,
    embedding: list[float],
    *,
    materials: list[str] | None = None,
    exercise_id: str | None = None,
):
    EmbeddedExercise = get_embedded_exercise_model()
    return EmbeddedExercise(
        exercise=exercise_named(
            name,
            materials=["Bar"] if materials is None else materials,
            exercise_id=exercise_id,
        ),
        embedding=embedding,
    )


def result_names(results) -> list[str]:
    return [result.exercise.name for result in results]


def result_similarities(results) -> list[float]:
    return [result.similarity for result in results]


class OnePassEmbeddedExerciseRepository:
    def __init__(self, embedded_exercises):
        self._embedded_exercises = tuple(embedded_exercises)
        self.calls = 0

    def iter_embedded_exercises(self):
        if self.calls != 0:
            raise AssertionError("iter_embedded_exercises() should only be called once")
        self.calls += 1
        return (exercise for exercise in self._embedded_exercises)


class ExplodingEmbeddedExerciseRepository:
    def __init__(self):
        self.calls = 0

    def iter_embedded_exercises(self):
        self.calls += 1
        raise AssertionError("repository should not be scanned")


class LenExplodingIterable(Iterable):
    def __init__(self, values):
        self._values = tuple(values)

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __len__(self) -> int:
        raise AssertionError("len() should not be used")

    def __length_hint__(self) -> int:
        raise AssertionError("__length_hint__() should not be used")


class LenExplodingEmbeddedExerciseRepository:
    def __init__(self, embedded_exercises):
        self._embedded_exercises = LenExplodingIterable(embedded_exercises)

    def iter_embedded_exercises(self):
        return self._embedded_exercises


def build_repository(embedded_exercises):
    JsonlEmbeddedExerciseSearchRepository = get_jsonl_embedded_exercise_search_repository()
    return JsonlEmbeddedExerciseSearchRepository(
        OnePassEmbeddedExerciseRepository(embedded_exercises)
    )


def build_metadata():
    EmbeddedExerciseCacheMetadata = get_cache_metadata_model()
    return EmbeddedExerciseCacheMetadata(
        embedding_model="test-model",
        embedding_dimension=2,
        text_builder_version="v1",
    )


def test_jsonl_embedded_exercise_search_repository_implements_search_protocol():
    EmbeddedExerciseSearchRepository = get_embedded_exercise_search_repository_protocol()
    JsonlEmbeddedExerciseSearchRepository = get_jsonl_embedded_exercise_search_repository()
    embedded_repository = OnePassEmbeddedExerciseRepository(
        [embedded_exercise_named("Pull Up", [1.0, 0.0])]
    )

    repository = JsonlEmbeddedExerciseSearchRepository(embedded_repository)
    results = repository.search(
        query_embedding=[1.0, 0.0],
        available_equipment=["Bar"],
        limit=1,
    )

    assert isinstance(repository, EmbeddedExerciseSearchRepository)
    assert not isinstance(results, list)
    assert result_names(list(results)) == ["Pull Up"]


def test_search_returns_search_results_with_raw_cosine_similarity():
    EmbeddedExerciseSearchResult = get_embedded_exercise_search_result_model()
    repository = build_repository(
        [
            embedded_exercise_named("Pull Up", [1.0, 0.0]),
            embedded_exercise_named("Body Row", [0.8, 0.2]),
        ]
    )

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=2,
        )
    )

    assert all(isinstance(result, EmbeddedExerciseSearchResult) for result in results)
    assert result_names(results) == ["Pull Up", "Body Row"]
    assert result_similarities(results) == pytest.approx([1.0, 0.9701425])


def test_search_applies_current_equipment_filtering_semantics():
    repository = build_repository(
        [
            embedded_exercise_named("Pull Up", [1.0, 0.0], materials=[" Bar "]),
            embedded_exercise_named("Ring Row", [0.99, 0.01], materials=["Rings"]),
            embedded_exercise_named(
                "Transition",
                [0.98, 0.02],
                materials=["Bar", "Rings"],
            ),
            embedded_exercise_named("Hollow Body Hold", [0.0, 1.0], materials=[]),
        ]
    )

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["bar", " Rings "],
            limit=4,
        )
    )

    assert result_names(results) == [
        "Pull Up",
        "Ring Row",
        "Transition",
        "Hollow Body Hold",
    ]


def test_search_excludes_exercises_missing_required_equipment():
    repository = build_repository(
        [
            embedded_exercise_named("Pull Up", [1.0, 0.0], materials=["Bar"]),
            embedded_exercise_named("Ring Row", [0.99, 0.01], materials=["Rings"]),
            embedded_exercise_named(
                "Transition",
                [0.98, 0.02],
                materials=["Bar", "Rings"],
            ),
            embedded_exercise_named("Hollow Body Hold", [0.0, 1.0], materials=[]),
        ]
    )

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=4,
        )
    )

    assert result_names(results) == ["Pull Up", "Hollow Body Hold"]


def test_search_matches_current_filter_then_retrieve_top_matches_behavior():
    retrieve_top_matches = get_retrieve_top_matches()
    exercise_matches_equipment = get_exercise_matches_equipment()
    embedded_exercises = [
        embedded_exercise_named("Low Match", [0.0, 1.0], materials=["Bar"]),
        embedded_exercise_named("Best Match", [1.0, 0.0], materials=["Bar"]),
        embedded_exercise_named("Wrong Equipment", [0.99, 0.01], materials=["Rings"]),
        embedded_exercise_named("Second Match", [0.8, 0.2], materials=["Bar"]),
        embedded_exercise_named("Third Match", [0.6, 0.4], materials=[]),
    ]
    repository = build_repository(embedded_exercises)
    query_embedding = [1.0, 0.0]
    available_equipment = ["Bar"]

    results = list(
        repository.search(
            query_embedding=query_embedding,
            available_equipment=available_equipment,
            limit=3,
        )
    )
    expected = retrieve_top_matches(
        query_embedding=query_embedding,
        embedded_exercises=(
            embedded_exercise
            for embedded_exercise in embedded_exercises
            if exercise_matches_equipment(
                embedded_exercise.exercise,
                available_equipment,
            )
        ),
        limit=3,
    )

    assert result_names(results) == [result.exercise.name for result in expected]
    assert result_similarities(results) == pytest.approx(
        [result.score for result in expected]
    )


def test_search_preserves_input_order_for_equal_scores():
    repository = build_repository(
        [
            embedded_exercise_named("Chin Up", [1.0, 0.0]),
            embedded_exercise_named("Neutral Grip Pull Up", [1.0, 0.0]),
            embedded_exercise_named("Body Row", [0.8, 0.2]),
        ]
    )

    first_results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=3,
        )
    )
    second_results = list(
        build_repository(
            [
                embedded_exercise_named("Chin Up", [1.0, 0.0]),
                embedded_exercise_named("Neutral Grip Pull Up", [1.0, 0.0]),
                embedded_exercise_named("Body Row", [0.8, 0.2]),
            ]
        ).search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=3,
        )
    )

    assert result_names(first_results) == [
        "Chin Up",
        "Neutral Grip Pull Up",
        "Body Row",
    ]
    assert result_names(second_results) == result_names(first_results)


def test_search_supports_duplicate_exercise_names_without_collisions():
    repository = build_repository(
        [
            embedded_exercise_named("Pull Up", [1.0, 0.0]),
            embedded_exercise_named("Pull Up", [0.0, 1.0]),
        ]
    )

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=2,
        )
    )

    assert result_names(results) == ["Pull Up", "Pull Up"]
    assert result_similarities(results) == pytest.approx([1.0, 0.0])


def test_search_works_with_local_jsonl_cache_repository(tmp_path):
    LocalEmbeddedExerciseCache = get_local_embedded_exercise_cache()
    LocalEmbeddedExerciseRepository = get_local_embedded_exercise_repository()
    JsonlEmbeddedExerciseSearchRepository = get_jsonl_embedded_exercise_search_repository()
    cache_path = tmp_path / "embedded_exercises.jsonl"
    embedded_exercises = [
        embedded_exercise_named("Pull Up", [1.0, 0.0], materials=["Bar"]),
        embedded_exercise_named("Ring Row", [0.8, 0.2], materials=["Rings"]),
    ]
    LocalEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        embedded_exercises,
        build_metadata(),
    )
    repository = JsonlEmbeddedExerciseSearchRepository(
        LocalEmbeddedExerciseRepository(cache_path)
    )

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=2,
        )
    )

    assert result_names(results) == ["Pull Up"]


def test_search_works_with_one_pass_embedded_exercise_iterables():
    embedded_repository = OnePassEmbeddedExerciseRepository(
        [
            embedded_exercise_named("Pull Up", [1.0, 0.0]),
            embedded_exercise_named("Body Row", [0.8, 0.2]),
        ]
    )
    JsonlEmbeddedExerciseSearchRepository = get_jsonl_embedded_exercise_search_repository()
    repository = JsonlEmbeddedExerciseSearchRepository(embedded_repository)

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=2,
        )
    )

    assert embedded_repository.calls == 1
    assert result_names(results) == ["Pull Up", "Body Row"]


def test_search_does_not_require_len_or_length_hint_on_embedded_exercises():
    JsonlEmbeddedExerciseSearchRepository = get_jsonl_embedded_exercise_search_repository()
    repository = JsonlEmbeddedExerciseSearchRepository(
        LenExplodingEmbeddedExerciseRepository(
            [
                embedded_exercise_named("Pull Up", [1.0, 0.0]),
                embedded_exercise_named("Body Row", [0.8, 0.2]),
            ]
        )
    )

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=2,
        )
    )

    assert result_names(results) == ["Pull Up", "Body Row"]


@pytest.mark.parametrize("limit", [0, -1])
def test_search_raises_for_invalid_limit_before_scanning_repository(limit):
    JsonlEmbeddedExerciseSearchRepository = get_jsonl_embedded_exercise_search_repository()
    embedded_repository = ExplodingEmbeddedExerciseRepository()
    repository = JsonlEmbeddedExerciseSearchRepository(embedded_repository)

    with pytest.raises(ValueError, match="limit"):
        list(
            repository.search(
                query_embedding=[1.0, 0.0],
                available_equipment=["Bar"],
                limit=limit,
            )
        )

    assert embedded_repository.calls == 0


def test_search_does_not_touch_network_or_sentence_transformer_modules(monkeypatch):
    repository = build_repository([embedded_exercise_named("Pull Up", [1.0, 0.0])])

    def fail(*args, **kwargs):
        raise AssertionError("unexpected external side effect")

    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    monkeypatch.setattr(Path, "read_text", fail)

    results = list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=1,
        )
    )

    assert result_names(results) == ["Pull Up"]
