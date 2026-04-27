from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from calisthenics_recommender.adapters.local_deterministic_embedding_provider import (
    LocalDeterministicEmbeddingProvider,
)
from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
    LocalEmbeddedExerciseCache,
)
from calisthenics_recommender.application.exercise_text_builder import (
    build_exercise_text,
)
from calisthenics_recommender.domain.embedded_exercise import EmbeddedExercise
from calisthenics_recommender.domain.exercise import Exercise


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


def write_local_deterministic_cache(cache_path: Path) -> None:
    exercises = [
        exercise_named(
            "Pull Up Negative",
            description="A controlled eccentric pull-up variation.",
            materials=["Bar"],
        ),
        exercise_named(
            "Body Row",
            description="A horizontal pulling variation with a bar.",
            materials=["Bar"],
        ),
        exercise_named(
            "Ring Pull Up",
            description="A pull-up variation that requires rings.",
            materials=["Rings"],
        ),
    ]
    embedding_provider = LocalDeterministicEmbeddingProvider(dimension=4)
    embedded_exercises = [
        EmbeddedExercise(
            exercise=exercise,
            embedding=tuple(
                embedding_provider.embed(build_exercise_text(exercise))
            ),
        )
        for exercise in exercises
    ]
    LocalEmbeddedExerciseCache(cache_path).write_embedded_exercises(
        embedded_exercises,
        metadata=EmbeddedExerciseCacheMetadata(
            embedding_model="fake-hash-v1",
            embedding_dimension=4,
            text_builder_version="v1",
        ),
    )


def write_sentence_transformer_cache(cache_path: Path) -> None:
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
            embedding=(0.8, 0.2, 0.0),
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
            embedding_model="cache-model-name",
            embedding_dimension=3,
            text_builder_version="v1",
        ),
    )


def valid_payload() -> dict[str, object]:
    return {
        "target_family": "Pull-up",
        "goal": "I want to build pulling strength and improve pull-ups.",
        "current_level": "I can do a few strict pull-ups.",
        "available_equipment": ["Bar"],
        "limit": 2,
    }


def runtime_env(
    cache_path: Path,
    *,
    embedding_provider: str,
    embedding_model: str | None = None,
    query_prefix: str | None = None,
) -> dict[str, str]:
    environ = {
        "CALISTHENICS_RECOMMENDER_CACHE_PATH": str(cache_path),
        "CALISTHENICS_RECOMMENDER_EMBEDDING_PROVIDER": embedding_provider,
    }
    if embedding_model is not None:
        environ["CALISTHENICS_RECOMMENDER_EMBEDDING_MODEL"] = embedding_model
    if query_prefix is not None:
        environ["CALISTHENICS_RECOMMENDER_QUERY_PREFIX"] = query_prefix
    return environ


def load_runtime_module():
    return importlib.import_module("calisthenics_recommender.api.runtime")


def test_create_configured_app_from_env_supports_local_deterministic_mode(tmp_path):
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_local_deterministic_cache(cache_path)
    runtime_module = load_runtime_module()

    app = runtime_module.create_configured_app_from_env(
        runtime_env(cache_path, embedding_provider="local-deterministic")
    )
    client = TestClient(app)

    health_response = client.get("/health")
    recommend_response = client.post("/recommend", json=valid_payload())

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert recommend_response.status_code == 200
    assert recommend_response.json()["recommendations"]
    recommended_names = {
        recommendation["exercise_name"]
        for recommendation in recommend_response.json()["recommendations"]
    }
    assert recommended_names == {"Pull Up Negative", "Body Row"}
    assert "Ring Pull Up" not in recommended_names


def test_create_configured_app_from_env_supports_sentence_transformer_mode_without_real_model(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_sentence_transformer_cache(cache_path)
    runtime_module = load_runtime_module()
    init_calls: list[tuple[str, str]] = []

    class FakeSentenceTransformerEmbeddingProvider:
        def __init__(
            self,
            model_name: str = "unused",
            text_prefix: str = "",
            normalize_embeddings: bool = True,
            model=None,
        ) -> None:
            init_calls.append((model_name, text_prefix))

        def get_embedding_dimension(self) -> int:
            return 3

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    monkeypatch.setattr(
        runtime_module,
        "SentenceTransformerEmbeddingProvider",
        FakeSentenceTransformerEmbeddingProvider,
    )

    app = runtime_module.create_configured_app_from_env(
        runtime_env(
            cache_path,
            embedding_provider="sentence-transformer",
            query_prefix="query: ",
        )
    )
    client = TestClient(app)

    response = client.post("/recommend", json=valid_payload())

    assert response.status_code == 200
    assert init_calls == [("cache-model-name", "query: ")]
    assert [item["exercise_name"] for item in response.json()["recommendations"]] == [
        "Pull Up Negative",
        "Body Row",
    ]


def test_create_configured_app_from_env_rejects_missing_required_env_vars(tmp_path):
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_local_deterministic_cache(cache_path)
    runtime_module = load_runtime_module()

    with pytest.raises(
        ValueError,
        match="CALISTHENICS_RECOMMENDER_CACHE_PATH is required",
    ):
        runtime_module.create_configured_app_from_env(
            {"CALISTHENICS_RECOMMENDER_EMBEDDING_PROVIDER": "local-deterministic"}
        )

    with pytest.raises(
        ValueError,
        match="CALISTHENICS_RECOMMENDER_EMBEDDING_PROVIDER is required",
    ):
        runtime_module.create_configured_app_from_env(
            {"CALISTHENICS_RECOMMENDER_CACHE_PATH": str(cache_path)}
        )


def test_create_configured_app_from_env_rejects_invalid_embedding_provider(tmp_path):
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_local_deterministic_cache(cache_path)
    runtime_module = load_runtime_module()

    with pytest.raises(
        ValueError,
        match="CALISTHENICS_RECOMMENDER_EMBEDDING_PROVIDER must be one of",
    ):
        runtime_module.create_configured_app_from_env(
            runtime_env(cache_path, embedding_provider="unsupported")
        )


def test_create_configured_app_from_env_rejects_sentence_transformer_dimension_mismatch(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_sentence_transformer_cache(cache_path)
    runtime_module = load_runtime_module()

    class FakeSentenceTransformerEmbeddingProvider:
        def __init__(
            self,
            model_name: str = "unused",
            text_prefix: str = "",
            normalize_embeddings: bool = True,
            model=None,
        ) -> None:
            self.model_name = model_name
            self.text_prefix = text_prefix

        def get_embedding_dimension(self) -> int:
            return 2

        def embed(self, text: str) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setattr(
        runtime_module,
        "SentenceTransformerEmbeddingProvider",
        FakeSentenceTransformerEmbeddingProvider,
    )

    with pytest.raises(
        ValueError,
        match="Sentence-transformer embedding dimension does not match cache metadata",
    ):
        runtime_module.create_configured_app_from_env(
            runtime_env(cache_path, embedding_provider="sentence-transformer")
        )


def test_read_runtime_config_from_env_returns_expected_values(tmp_path):
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_local_deterministic_cache(cache_path)
    runtime_module = load_runtime_module()

    config = runtime_module.read_runtime_config_from_env(
        runtime_env(
            cache_path,
            embedding_provider="sentence-transformer",
            embedding_model="custom/local-model",
            query_prefix="query: ",
        )
    )

    assert config.cache_path == cache_path
    assert config.embedding_provider == "sentence-transformer"
    assert config.embedding_model == "custom/local-model"
    assert config.query_prefix == "query: "


def test_importing_api_main_exposes_module_level_app(tmp_path, monkeypatch):
    cache_path = tmp_path / "embedded_exercises.jsonl"
    write_local_deterministic_cache(cache_path)
    monkeypatch.setenv(
        "CALISTHENICS_RECOMMENDER_CACHE_PATH", str(cache_path)
    )
    monkeypatch.setenv(
        "CALISTHENICS_RECOMMENDER_EMBEDDING_PROVIDER",
        "local-deterministic",
    )
    sys.modules.pop("calisthenics_recommender.api.main", None)

    module = importlib.import_module("calisthenics_recommender.api.main")
    client = TestClient(module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
