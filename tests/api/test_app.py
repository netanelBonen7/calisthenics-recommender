from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from calisthenics_recommender.adapters.fake_embedding_provider import (
    FakeEmbeddingProvider,
)
from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
    LocalEmbeddedExerciseCache,
    LocalEmbeddedExerciseRepository,
)
from calisthenics_recommender.application.query_builder import build_query_text
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise
from calisthenics_recommender.domain.user_request import UserRequest


def exercise_named(
    name: str,
    *,
    description: str,
    materials: list[str],
) -> Exercise:
    return Exercise(
        name=name,
        description=description,
        muscle_groups=["Back"],
        families=["Pull-up"],
        materials=materials,
        categories=["Upper Body Pull"],
    )


def write_embedded_exercise_cache(cache_path: Path) -> None:
    embedded_exercises = [
        EmbeddedExercise(
            exercise=exercise_named(
                "Pull Up Negative",
                description="A controlled eccentric pull-up variation.",
                materials=["Bar"],
            ),
            embedding=(1.0, 0.0, 0.0),
        ),
        EmbeddedExercise(
            exercise=exercise_named(
                "Body Row",
                description="A horizontal pulling variation with a bar.",
                materials=["Bar"],
            ),
            embedding=(0.95, 0.05, 0.0),
        ),
        EmbeddedExercise(
            exercise=exercise_named(
                "Ring Pull Up",
                description="A pull-up variation that requires rings.",
                materials=["Rings"],
            ),
            embedding=(1.0, 0.0, 0.0),
        ),
    ]
    LocalEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        embedded_exercises,
        metadata=EmbeddedExerciseCacheMetadata(
            embedding_model="fake-local",
            embedding_dimension=3,
            text_builder_version="v1",
        ),
    )


class RecordingFakeEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        super().__init__(embeddings)
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return super().embed(text)


def build_client(tmp_path: Path) -> tuple[TestClient, RecordingFakeEmbeddingProvider]:
    from calisthenics_recommender.api import create_app

    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_embedded_exercise_cache(cache_path)
    user_request = UserRequest(
        target_family="Pull-up",
        goal="I want to build pulling strength and improve pull-ups.",
        current_level="I can do a few strict pull-ups.",
        available_equipment=["Bar"],
    )
    query_text = build_query_text(user_request)
    embedding_provider = RecordingFakeEmbeddingProvider(
        {query_text: [1.0, 0.0, 0.0]}
    )
    app = create_app(
        embedded_exercise_repository=LocalEmbeddedExerciseRepository(cache_path),
        embedding_provider=embedding_provider,
    )
    return TestClient(app), embedding_provider


def valid_payload() -> dict[str, object]:
    return {
        "target_family": "Pull-up",
        "goal": "I want to build pulling strength and improve pull-ups.",
        "current_level": "I can do a few strict pull-ups.",
        "available_equipment": ["Bar"],
        "limit": 2,
    }


def test_health_endpoint_returns_ok(tmp_path):
    client, _ = build_client(tmp_path)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_returns_envelope_json_using_injected_dependencies(tmp_path):
    client, embedding_provider = build_client(tmp_path)
    payload = valid_payload()

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "recommendations": [
            {
                "exercise_name": "Pull Up Negative",
                "match_score": 100,
                "reason": (
                    "Recommended because it matched your Pull-up target family "
                    "through retrieval, belongs to the Pull-up families, falls "
                    "under the Upper Body Pull categories, and requires Bar."
                ),
                "required_equipment": ["Bar"],
                "category_family": {
                    "categories": ["Upper Body Pull"],
                    "families": ["Pull-up"],
                },
            },
            {
                "exercise_name": "Body Row",
                "match_score": 100,
                "reason": (
                    "Recommended because it matched your Pull-up target family "
                    "through retrieval, belongs to the Pull-up families, falls "
                    "under the Upper Body Pull categories, and requires Bar."
                ),
                "required_equipment": ["Bar"],
                "category_family": {
                    "categories": ["Upper Body Pull"],
                    "families": ["Pull-up"],
                },
            },
        ]
    }
    expected_query_text = build_query_text(
        UserRequest(
            target_family=payload["target_family"],
            goal=payload["goal"],
            current_level=payload["current_level"],
            available_equipment=payload["available_equipment"],
        )
    )
    assert embedding_provider.calls == [expected_query_text]


def test_recommend_endpoint_applies_existing_equipment_filtering(tmp_path):
    client, _ = build_client(tmp_path)

    response = client.post("/recommend", json=valid_payload())

    assert response.status_code == 200
    recommended_names = [
        recommendation["exercise_name"]
        for recommendation in response.json()["recommendations"]
    ]
    assert recommended_names == ["Pull Up Negative", "Body Row"]
    assert "Ring Pull Up" not in recommended_names


def test_recommend_endpoint_rejects_invalid_limit(tmp_path):
    client, _ = build_client(tmp_path)
    payload = valid_payload()
    payload["limit"] = 0

    response = client.post("/recommend", json=payload)

    assert response.status_code == 422


def test_recommend_endpoint_rejects_string_available_equipment(tmp_path):
    client, _ = build_client(tmp_path)
    payload = valid_payload()
    payload["available_equipment"] = "Bar"

    response = client.post("/recommend", json=payload)

    assert response.status_code == 422
