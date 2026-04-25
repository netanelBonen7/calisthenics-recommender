from copy import deepcopy
from importlib import import_module
import inspect
from pathlib import Path
import socket

import pytest

from calisthenics_recommender.application.exercise_text_builder import (
    build_exercise_text,
)
from calisthenics_recommender.application.query_builder import build_query_text


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


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


class InMemoryExerciseRepository:
    def __init__(self, exercises):
        self._exercises = list(exercises)
        self.calls = 0

    def list_exercises(self):
        self.calls += 1
        return self._exercises


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
        "exercise_repository",
        "embedding_provider",
        "limit",
    ]


def test_recommend_exercises_returns_ranked_recommendations_with_filtering_and_limit():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    pull_up = exercise_named(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    body_row = exercise_named(
        "Body Row",
        description="A horizontal pull that builds pulling volume.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    hollow_body_hold = exercise_named(
        "Hollow Body Hold",
        description="A core position drill for body tension.",
        families=["Core"],
        materials=[],
        categories=["Core"],
    )
    ring_pull_up = exercise_named(
        "Ring Pull Up",
        description="A pulling variation that requires rings.",
        families=["Pull-up"],
        materials=["Rings"],
        categories=["Upper Body Pull"],
    )
    repository = InMemoryExerciseRepository(
        [pull_up, body_row, hollow_body_hold, ring_pull_up]
    )
    embeddings = {
        build_query_text(user_request): [1.0, 0.0],
        build_exercise_text(pull_up): [0.95, 0.05],
        build_exercise_text(body_row): [0.80, 0.20],
        build_exercise_text(hollow_body_hold): [0.0, 1.0],
    }
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
    assert embedding_provider.calls == [
        build_query_text(user_request),
        build_exercise_text(pull_up),
        build_exercise_text(body_row),
        build_exercise_text(hollow_body_hold),
    ]


def test_recommend_exercises_returns_the_full_recommendation_shape_in_the_happy_path():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    pull_up = exercise_named(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    body_row = exercise_named(
        "Body Row",
        description="A horizontal pull that builds pulling volume.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    repository = InMemoryExerciseRepository([pull_up, body_row])
    embedding_provider = RecordingEmbeddingProvider(
        {
            build_query_text(user_request): [1.0, 0.0],
            build_exercise_text(pull_up): [0.95, 0.05],
            build_exercise_text(body_row): [0.80, 0.20],
        }
    )

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
    repository = InMemoryExerciseRepository([])
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
    repository = InMemoryExerciseRepository(
        [
            exercise_named(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
            ),
            exercise_named(
                "Ring Dip",
                description="A dip variation that requires rings.",
                families=["Dip"],
                materials=["Rings"],
                categories=["Upper Body Push"],
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


@pytest.mark.parametrize("limit", [0, -1])
def test_recommend_exercises_raises_for_invalid_limits_without_embedding_work(limit):
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    repository = InMemoryExerciseRepository(
        [
            exercise_named(
                "Pull Up",
                description="A strict vertical pulling movement on a bar.",
                families=["Pull-up"],
                materials=["Bar"],
                categories=["Upper Body Pull"],
            )
        ]
    )
    embedding_provider = RecordingEmbeddingProvider(
        {
            build_query_text(user_request): [1.0, 0.0],
            build_exercise_text(repository.list_exercises()[0]): [0.95, 0.05],
        }
    )
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
    exercise = exercise_named(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    repository = InMemoryExerciseRepository([exercise])
    embedding_provider = RecordingEmbeddingProvider(
        {
            build_exercise_text(exercise): [0.95, 0.05],
        }
    )

    with pytest.raises(KeyError, match="Unknown text"):
        recommend_exercises(
            user_request,
            repository,
            embedding_provider,
            limit=1,
        )


def test_recommend_exercises_raises_a_clear_error_when_an_exercise_embedding_is_missing():
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    exercise = exercise_named(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    repository = InMemoryExerciseRepository([exercise])
    embedding_provider = RecordingEmbeddingProvider(
        {
            build_query_text(user_request): [1.0, 0.0],
        }
    )

    with pytest.raises(KeyError, match="Unknown text"):
        recommend_exercises(
            user_request,
            repository,
            embedding_provider,
            limit=1,
        )


def test_recommend_exercises_logs_only_safe_operational_counts(caplog):
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    pull_up = exercise_named(
        "Pull Up",
        description="A strict vertical pulling movement on a bar.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    body_row = exercise_named(
        "Body Row",
        description="A horizontal pull that builds pulling volume.",
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    repository = InMemoryExerciseRepository([pull_up, body_row])
    embedding_provider = RecordingEmbeddingProvider(
        {
            build_query_text(user_request): [1.0, 0.0],
            build_exercise_text(pull_up): [0.95, 0.05],
            build_exercise_text(body_row): [0.80, 0.20],
        }
    )

    with caplog.at_level("INFO"):
        recommend_exercises(
            user_request,
            repository,
            embedding_provider,
            limit=2,
        )

    assert caplog.messages == [
        "Loaded 2 exercises from repository",
        "Filtered exercises down to 2 candidates",
        "Returning 2 recommendations",
    ]
    assert user_request.goal not in caplog.text
    assert user_request.current_level not in caplog.text
    assert build_query_text(user_request) not in caplog.text
    assert build_exercise_text(pull_up) not in caplog.text
    assert "[1.0, 0.0]" not in caplog.text


def test_recommend_exercises_is_deterministic_and_does_not_mutate_inputs_or_touch_files_or_network(
    monkeypatch,
):
    recommend_exercises = get_recommend_exercises()
    user_request = valid_user_request()
    exercises = [
        exercise_named(
            "Pull Up",
            description="A strict vertical pulling movement on a bar.",
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
        ),
        exercise_named(
            "Body Row",
            description="A horizontal pull that builds pulling volume.",
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
        ),
    ]
    repository = InMemoryExerciseRepository(exercises)
    embeddings = {
        build_query_text(user_request): [1.0, 0.0],
        build_exercise_text(exercises[0]): [0.95, 0.05],
        build_exercise_text(exercises[1]): [0.80, 0.20],
    }
    embedding_provider = RecordingEmbeddingProvider(embeddings)
    original_user_request = deepcopy(user_request)
    original_repository_data = deepcopy(repository.list_exercises())
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
    assert repository.list_exercises() == original_repository_data
    assert embedding_provider.embeddings == original_embeddings
