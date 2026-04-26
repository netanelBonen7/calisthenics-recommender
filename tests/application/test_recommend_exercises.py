from copy import deepcopy
from importlib import import_module
import inspect
from typing import Iterable, Iterator
from pathlib import Path
import socket

import pytest

from calisthenics_recommender.application.query_builder import build_query_text


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_model():
    module = import_module("calisthenics_recommender.domain.embedded_exercise")
    return getattr(module, "EmbeddedExercise")


def get_user_request_model():
    module = import_module("calisthenics_recommender.domain.user_request")
    return getattr(module, "UserRequest")


def get_recommend_exercises():
    module = import_module("calisthenics_recommender.application.recommend_exercises")
    return getattr(module, "recommend_exercises")


def exercise_named(
    name: str,
    *,
    description: str,
    families: list[str],
    materials: list[str],
    categories: list[str],
):
    Exercise = get_exercise_model()
    return Exercise(
        name=name,
        description=description,
        muscle_groups=["Back"],
        families=families,
        materials=materials,
        categories=categories,
    )


def valid_user_request(available_equipment: list[str] | None = None):
    UserRequest = get_user_request_model()
    return UserRequest(
        target_family="Pull-up",
        goal="I want stronger pulling strength for harder variations.",
        current_level="I can do 5 strict pull-ups with slow last reps.",
        available_equipment=["Bar"] if available_equipment is None else available_equipment,
    )


def embedded_exercise(
    name: str,
    *,
    description: str,
    families: list[str],
    materials: list[str],
    categories: list[str],
    embedding: list[float],
):
    EmbeddedExercise = get_embedded_exercise_model()
    exercise = exercise_named(
        name,
        description=description,
        families=families,
        materials=materials,
        categories=categories,
    )
    return EmbeddedExercise(exercise=exercise, embedding=embedding)


class InMemoryEmbeddedExerciseRepository:
    def __init__(self, embedded_exercises):
        self._embedded_exercises = tuple(embedded_exercises)
        self.calls = 0

    def iter_embedded_exercises(self):
        self.calls += 1
        return iter(self._embedded_exercises)

    def snapshot(self):
        return list(self._embedded_exercises)


class OnePassEmbeddedExerciseRepository:
    def __init__(self, embedded_exercises):
        self._embedded_exercises = tuple(embedded_exercises)
        self.calls = 0

    def iter_embedded_exercises(self):
        if self.calls != 0:
            raise AssertionError("iter_embedded_exercises() should only be called once")
        self.calls += 1
        return (exercise for exercise in self._embedded_exercises)


class LenExplodingIterable(Iterable):
    def __init__(self, values):
        self._values = tuple(values)

    def __iter__(self) -> Iterator:
        return iter(self._values)

    def __len__(self) -> int:
        raise AssertionError("len() should not be used")

    def __length_hint__(self) -> int:
        raise AssertionError("__length_hint__() should not be used")


class StreamingProbeIterable:
    def __init__(self, embedded_exercises, embedding_provider):
        self._embedded_exercises = tuple(embedded_exercises)
        self._embedding_provider = embedding_provider

    def __iter__(self):
        for index, embedded_exercise in enumerate(self._embedded_exercises):
            if index >= 2 and not self._embedding_provider.calls:
                raise AssertionError(
                    "embedded exercises were fully materialized before query embedding"
                )
            yield embedded_exercise

    def __len__(self) -> int:
        raise AssertionError("len() should not be used")

    def __length_hint__(self) -> int:
        raise AssertionError("__length_hint__() should not be used")


class StreamingProbeEmbeddedExerciseRepository:
    def __init__(self, embedded_exercises, embedding_provider):
        self._embedded_exercises = embedded_exercises
        self._embedding_provider = embedding_provider
        self.calls = 0

    def iter_embedded_exercises(self):
        self.calls += 1
        return iter(
            StreamingProbeIterable(
                self._embedded_exercises, self._embedding_provider
            )
        )


class LenExplodingEmbeddedExerciseRepository:
    def __init__(self, embedded_exercises):
        self._embedded_exercises = embedded_exercises
        self.calls = 0

    def iter_embedded_exercises(self):
        self.calls += 1
        return LenExplodingIterable(self._embedded_exercises)


class RecordingEmbeddingProvider:
    def __init__(self, embeddings):
        self._embeddings = {text: list(vector) for text, vector in embeddings.items()}
        self.calls: list[str] = []

    @property
    def embeddings(self):
        return self._embeddings

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        try:
            return list(self._embeddings[text])
        except KeyError as error:
            raise KeyError(f"Unknown text: {text}") from error


def test_recommend_exercises_accepts_expected_arguments():
    recommend_exercises = get_recommend_exercises()

    assert list(inspect.signature(recommend_exercises).parameters) == [
        "user_request",
        "embedded_exercise_repository",
        "embedding_provider",
        "limit",
    ]


def test_recommend_exercises_returns_ranked_recommendations_with_filtering_and_limit():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    pull_up = embedded_exercise(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        embedding=[0.95, 0.05],
    )
    body_row = embedded_exercise(
        "Body Row",
        description="A horizontal pull that builds pulling volume.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        embedding=[0.80, 0.20],
    )
    hollow_body_hold = embedded_exercise(
        "Hollow Body Hold",
        description="A core position drill for body tension.",
        families=["Core"],
        materials=[],
        categories=["Core"],
        embedding=[0.0, 1.0],
    )
    ring_pull_up = embedded_exercise(
        "Ring Pull Up",
        description="A pulling variation that requires rings.",
        families=["Pull-up"],
        materials=["Rings"],
        categories=["Upper Body Pull"],
        embedding=[1.0, 0.0],
    )
    repository = InMemoryEmbeddedExerciseRepository(
        [pull_up, body_row, hollow_body_hold, ring_pull_up]
    )
    embeddings = {build_query_text(user_request): [1.0, 0.0]}
    embedding_provider = RecordingEmbeddingProvider(embeddings)

    recommendations = recommend_exercises(
        user_request,
        repository,
        embedding_provider,
        limit=2,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up",
        "Body Row",
    ]
    assert [recommendation.match_score for recommendation in recommendations] == [100, 97]
    assert all(
        recommendation.exercise_name != "Ring Pull Up"
        for recommendation in recommendations
    )
    assert embedding_provider.calls == [build_query_text(user_request)]


def test_recommend_exercises_embeds_only_the_query_and_never_runtime_exercise_text(
    monkeypatch,
):
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    repository = InMemoryEmbeddedExerciseRepository(
        [
            embedded_exercise(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.95, 0.05],
            )
        ]
    )
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )

    def fail(*args, **kwargs):
        raise AssertionError("build_exercise_text should not be called at runtime")

    monkeypatch.setattr(
        "calisthenics_recommender.application.exercise_text_builder.build_exercise_text",
        fail,
    )

    recommendations = recommend_exercises(
        user_request,
        repository,
        embedding_provider,
        limit=1,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up"
    ]
    assert embedding_provider.calls == [build_query_text(user_request)]


def test_recommend_exercises_returns_the_full_recommendation_shape_in_the_happy_path():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    pull_up = embedded_exercise(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        embedding=[0.95, 0.05],
    )
    body_row = embedded_exercise(
        "Body Row",
        description="A horizontal pull that builds pulling volume.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        embedding=[0.80, 0.20],
    )
    repository = InMemoryEmbeddedExerciseRepository([pull_up, body_row])
    embedding_provider = RecordingEmbeddingProvider({build_query_text(user_request): [1.0, 0.0]})

    first_recommendation = recommend_exercises(
        user_request,
        repository,
        embedding_provider,
        limit=2,
    )[0]

    assert first_recommendation.exercise_name == "Pull Up"
    assert first_recommendation.match_score == 100
    assert "Pull-up target family" in first_recommendation.reason
    assert "Pull-up families" in first_recommendation.reason
    assert "Upper Body Pull categories" in first_recommendation.reason
    assert "requires Bar" in first_recommendation.reason
    assert first_recommendation.required_equipment == ["Bar"]
    assert first_recommendation.category_family.categories == ["Upper Body Pull"]
    assert first_recommendation.category_family.families == ["Pull-up"]


def test_recommend_exercises_returns_an_empty_list_when_repository_is_empty():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    repository = InMemoryEmbeddedExerciseRepository([])
    embedding_provider = RecordingEmbeddingProvider({})

    recommendations = recommend_exercises(
        user_request,
        repository,
        embedding_provider,
        limit=3,
    )

    assert recommendations == []
    assert repository.calls == 1
    assert embedding_provider.calls == []


def test_recommend_exercises_returns_an_empty_list_when_no_exercises_match_equipment():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request(available_equipment=["Parallettes"])
    repository = InMemoryEmbeddedExerciseRepository(
        [
            embedded_exercise(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.95, 0.05],
            ),
            embedded_exercise(
                "Ring Dip",
                description="A dip variation that requires rings.",
                families=["Dip"],
                materials=["Rings"],
                categories=["Upper Body Push"],
                embedding=[0.0, 1.0],
            ),
        ]
    )
    embedding_provider = RecordingEmbeddingProvider({})

    recommendations = recommend_exercises(
        user_request,
        repository,
        embedding_provider,
        limit=3,
    )

    assert recommendations == []
    assert embedding_provider.calls == []


def test_recommend_exercises_supports_a_one_pass_generator_repository():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    repository = OnePassEmbeddedExerciseRepository(
        [
            embedded_exercise(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.95, 0.05],
            ),
            embedded_exercise(
                "Body Row",
                description="A horizontal pull that builds pulling volume.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.80, 0.20],
            ),
        ]
    )
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )

    recommendations = recommend_exercises(
        user_request,
        repository,
        embedding_provider,
        limit=2,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up",
        "Body Row",
    ]
    assert repository.calls == 1


def test_recommend_exercises_does_not_require_len_on_repository_output():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    repository = LenExplodingEmbeddedExerciseRepository(
        [
            embedded_exercise(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.95, 0.05],
            ),
            embedded_exercise(
                "Body Row",
                description="A horizontal pull that builds pulling volume.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.80, 0.20],
            ),
        ]
    )
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )

    recommendations = recommend_exercises(
        user_request,
        repository,
        embedding_provider,
        limit=2,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up",
        "Body Row",
    ]


def test_recommend_exercises_embeds_after_first_matching_candidate_without_full_materialization():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )
    repository = StreamingProbeEmbeddedExerciseRepository(
        [
            embedded_exercise(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.95, 0.05],
            ),
            embedded_exercise(
                "Body Row",
                description="A horizontal pull that builds pulling volume.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.80, 0.20],
            ),
            embedded_exercise(
                "Push Up",
                description="A horizontal pushing movement.",
                families=["Push-up"],
                materials=[],
                categories=["Upper Body Push"],
                embedding=[0.0, 1.0],
            ),
        ],
        embedding_provider,
    )

    recommendations = recommend_exercises(
        user_request,
        repository,
        embedding_provider,
        limit=2,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up",
        "Body Row",
    ]
    assert embedding_provider.calls == [build_query_text(user_request)]


@pytest.mark.parametrize("limit", [0, -1])
def test_recommend_exercises_raises_for_invalid_limits_without_embedding_work(limit):
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    repository = InMemoryEmbeddedExerciseRepository(
        [
            embedded_exercise(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                embedding=[0.95, 0.05],
            )
        ]
    )
    embedding_provider = RecordingEmbeddingProvider({build_query_text(user_request): [1.0, 0.0]})
    repository.calls = 0

    with pytest.raises(ValueError, match="limit"):
        recommend_exercises(
            user_request,
            repository,
            embedding_provider,
            limit=limit,
        )

    assert repository.calls == 0
    assert embedding_provider.calls == []


def test_recommend_exercises_raises_a_clear_error_when_the_query_embedding_is_missing():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    exercise = embedded_exercise(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        embedding=[0.95, 0.05],
    )
    repository = InMemoryEmbeddedExerciseRepository([exercise])
    embedding_provider = RecordingEmbeddingProvider({})

    with pytest.raises(KeyError, match="Unknown text"):
        recommend_exercises(
            user_request,
            repository,
            embedding_provider,
            limit=1,
        )


def test_recommend_exercises_supports_duplicate_exercise_names_when_vectors_are_precomputed():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    first_exercise = embedded_exercise(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        embedding=[0.95, 0.05],
    )
    second_exercise = embedded_exercise(
        "Pull Up",
        description="A pulling variation with the same display name.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        embedding=[0.0, 1.0],
    )
    repository = InMemoryEmbeddedExerciseRepository([first_exercise, second_exercise])
    embedding_provider = RecordingEmbeddingProvider({build_query_text(user_request): [1.0, 0.0]})

    recommendations = recommend_exercises(
        user_request,
        repository,
        embedding_provider,
        limit=2,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up",
        "Pull Up",
    ]
    assert [recommendation.match_score for recommendation in recommendations] == [100, 0]


def test_recommend_exercises_logs_only_safe_operational_counts(caplog):
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    pull_up = embedded_exercise(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        embedding=[0.95, 0.05],
    )
    body_row = embedded_exercise(
        "Body Row",
        description="A horizontal pull that builds pulling volume.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        embedding=[0.80, 0.20],
    )
    repository = InMemoryEmbeddedExerciseRepository([pull_up, body_row])
    embedding_provider = RecordingEmbeddingProvider({build_query_text(user_request): [1.0, 0.0]})

    with caplog.at_level("INFO"):
        recommend_exercises(
            user_request,
            repository,
            embedding_provider,
            limit=2,
        )

    assert caplog.messages == [
        "Scanned 2 embedded exercises; 2 matched equipment; returning 2 recommendations",
    ]
    assert user_request.goal not in caplog.text
    assert user_request.current_level not in caplog.text
    assert build_query_text(user_request) not in caplog.text
    assert "[1.0, 0.0]" not in caplog.text


def test_recommend_exercises_is_deterministic_and_does_not_mutate_inputs_or_touch_files_or_network(
    monkeypatch,
):
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    embedded_exercises = [
        embedded_exercise(
            "Pull Up",
            description="A strict vertical pulling movement on a bar.",
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
            embedding=[0.95, 0.05],
        ),
        embedded_exercise(
            "Body Row",
            description="A horizontal pull that builds pulling volume.",
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
            embedding=[0.80, 0.20],
        ),
    ]
    repository = InMemoryEmbeddedExerciseRepository(embedded_exercises)
    embedding_provider = RecordingEmbeddingProvider({build_query_text(user_request): [1.0, 0.0]})
    original_user_request = deepcopy(user_request)
    original_repository_data = deepcopy(repository.snapshot())
    original_embeddings = deepcopy(embedding_provider.embeddings)

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    first = recommend_exercises(user_request, repository, embedding_provider, limit=2)
    second = recommend_exercises(user_request, repository, embedding_provider, limit=2)

    assert first == second
    assert user_request == original_user_request
    assert repository.snapshot() == original_repository_data
    assert embedding_provider.embeddings == original_embeddings
