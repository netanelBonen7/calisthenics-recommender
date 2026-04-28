from copy import deepcopy
from importlib import import_module
import inspect
from pathlib import Path
import socket

import pytest

from calisthenics_recommender.application.query_builder import build_query_text


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_search_result_model():
    module = import_module(
        "calisthenics_recommender.domain.embedded_exercise_search_result"
    )
    return getattr(module, "EmbeddedExerciseSearchResult")


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
        exercise_id=name.lower().replace(" ", "-"),
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


def search_result(
    name: str,
    *,
    description: str,
    families: list[str],
    materials: list[str],
    categories: list[str],
    similarity: float,
):
    EmbeddedExerciseSearchResult = get_embedded_exercise_search_result_model()
    exercise = exercise_named(
        name,
        description=description,
        families=families,
        materials=materials,
        categories=categories,
    )
    return EmbeddedExerciseSearchResult(exercise=exercise, similarity=similarity)


class RecordingEmbeddedExerciseSearchRepository:
    def __init__(self, search_results):
        self._search_results = tuple(search_results)
        self.calls: list[dict[str, object]] = []

    def search(self, *, query_embedding, available_equipment, limit):
        self.calls.append(
            {
                "query_embedding": list(query_embedding),
                "available_equipment": list(available_equipment),
                "limit": limit,
            }
        )
        return iter(self._search_results[:limit])

    def snapshot(self):
        return list(self._search_results)


class LenExplodingSearchResults:
    def __init__(self, values):
        self._values = tuple(values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        raise AssertionError("len() should not be used")

    def __length_hint__(self) -> int:
        raise AssertionError("__length_hint__() should not be used")


class LenExplodingEmbeddedExerciseSearchRepository:
    def __init__(self, search_results):
        self._search_results = search_results
        self.calls: list[dict[str, object]] = []

    def search(self, *, query_embedding, available_equipment, limit):
        self.calls.append(
            {
                "query_embedding": list(query_embedding),
                "available_equipment": list(available_equipment),
                "limit": limit,
            }
        )
        return LenExplodingSearchResults(self._search_results[:limit])


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
        "embedded_exercise_search_repository",
        "embedding_provider",
        "limit",
    ]


def test_recommend_exercises_embeds_query_and_calls_search_repository_with_user_filters():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request(available_equipment=["Bar", "Rings"])
    search_repository = RecordingEmbeddedExerciseSearchRepository(
        [
            search_result(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                similarity=1.0,
            ),
            search_result(
                "Body Row",
                description="A horizontal pull that builds pulling volume.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                similarity=0.9701425,
            ),
        ]
    )
    embeddings = {build_query_text(user_request): [1.0, 0.0]}
    embedding_provider = RecordingEmbeddingProvider(embeddings)

    recommendations = recommend_exercises(
        user_request,
        search_repository,
        embedding_provider,
        limit=2,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up",
        "Body Row",
    ]
    assert embedding_provider.calls == [build_query_text(user_request)]
    assert search_repository.calls == [
        {
            "query_embedding": [1.0, 0.0],
            "available_equipment": ["Bar", "Rings"],
            "limit": 2,
        }
    ]


def test_recommend_exercises_returns_ranked_recommendations_from_search_results():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    search_repository = RecordingEmbeddedExerciseSearchRepository(
        [
            search_result(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                similarity=1.0,
            ),
            search_result(
                "Body Row",
                description="A horizontal pull that builds pulling volume.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                similarity=0.9701425,
            ),
        ]
    )
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )

    recommendations = recommend_exercises(
        user_request,
        search_repository,
        embedding_provider,
        limit=2,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up",
        "Body Row",
    ]
    assert [recommendation.match_score for recommendation in recommendations] == [
        100,
        97,
    ]


def test_recommend_exercises_returns_the_full_recommendation_shape_in_the_happy_path():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    search_repository = RecordingEmbeddedExerciseSearchRepository(
        [
            search_result(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                similarity=1.0,
            ),
            search_result(
                "Body Row",
                description="A horizontal pull that builds pulling volume.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                similarity=0.9701425,
            ),
        ]
    )
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )

    first_recommendation = recommend_exercises(
        user_request,
        search_repository,
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


def test_recommend_exercises_returns_an_empty_list_when_search_returns_no_results():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    search_repository = RecordingEmbeddedExerciseSearchRepository([])
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )

    recommendations = recommend_exercises(
        user_request,
        search_repository,
        embedding_provider,
        limit=3,
    )

    assert recommendations == []
    assert embedding_provider.calls == [build_query_text(user_request)]
    assert search_repository.calls == [
        {
            "query_embedding": [1.0, 0.0],
            "available_equipment": ["Bar"],
            "limit": 3,
        }
    ]


def test_recommend_exercises_does_not_require_len_on_search_results():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    search_repository = LenExplodingEmbeddedExerciseSearchRepository(
        [
            search_result(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                similarity=1.0,
            ),
            search_result(
                "Body Row",
                description="A horizontal pull that builds pulling volume.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                similarity=0.9701425,
            ),
        ]
    )
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )

    recommendations = recommend_exercises(
        user_request,
        search_repository,
        embedding_provider,
        limit=2,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up",
        "Body Row",
    ]


@pytest.mark.parametrize("limit", [0, -1])
def test_recommend_exercises_raises_for_invalid_limits_without_embedding_work(limit):
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    search_repository = RecordingEmbeddedExerciseSearchRepository(
        [
            search_result(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
                similarity=1.0,
            )
        ]
    )
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )

    with pytest.raises(ValueError, match="limit"):
        recommend_exercises(
            user_request,
            search_repository,
            embedding_provider,
            limit=limit,
        )

    assert search_repository.calls == []
    assert embedding_provider.calls == []


def test_recommend_exercises_raises_a_clear_error_when_the_query_embedding_is_missing():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    search_repository = RecordingEmbeddedExerciseSearchRepository([])
    embedding_provider = RecordingEmbeddingProvider({})

    with pytest.raises(KeyError, match="Unknown text"):
        recommend_exercises(
            user_request,
            search_repository,
            embedding_provider,
            limit=1,
        )


def test_recommend_exercises_supports_duplicate_exercise_names_when_search_returns_them():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    first_result = search_result(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        similarity=1.0,
    )
    second_result = search_result(
        "Pull Up",
        description="A pulling variation with the same display name.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
        similarity=0.0,
    )
    search_repository = RecordingEmbeddedExerciseSearchRepository(
        [first_result, second_result]
    )
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )

    recommendations = recommend_exercises(
        user_request,
        search_repository,
        embedding_provider,
        limit=2,
    )

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up",
        "Pull Up",
    ]
    assert [recommendation.match_score for recommendation in recommendations] == [100, 0]


def test_recommend_exercises_is_deterministic_and_does_not_mutate_inputs_or_touch_files_or_network(
    monkeypatch,
):
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    search_results = [
        search_result(
            "Pull Up",
            description="A strict vertical pulling movement on a bar.",
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
            similarity=1.0,
        ),
        search_result(
            "Body Row",
            description="A horizontal pull that builds pulling volume.",
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
            similarity=0.9701425,
        ),
    ]
    search_repository = RecordingEmbeddedExerciseSearchRepository(search_results)
    embedding_provider = RecordingEmbeddingProvider(
        {build_query_text(user_request): [1.0, 0.0]}
    )
    original_user_request = deepcopy(user_request)
    original_search_results = deepcopy(search_repository.snapshot())
    original_embeddings = deepcopy(embedding_provider.embeddings)

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    first = recommend_exercises(
        user_request, search_repository, embedding_provider, limit=2
    )
    second = recommend_exercises(
        user_request, search_repository, embedding_provider, limit=2
    )

    assert first == second
    assert user_request == original_user_request
    assert search_repository.snapshot() == original_search_results
    assert embedding_provider.embeddings == original_embeddings
