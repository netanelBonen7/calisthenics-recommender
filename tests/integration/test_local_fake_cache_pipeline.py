import csv

from calisthenics_recommender.adapters.csv_exercise_repository import (
    CsvExerciseRepository,
)
from calisthenics_recommender.adapters.fake_embedding_provider import (
    FakeEmbeddingProvider,
)
from calisthenics_recommender.adapters.jsonl_embedded_exercise_search_repository import (
    JsonlEmbeddedExerciseSearchRepository,
)
from calisthenics_recommender.adapters.local_embedded_exercise_cache import (
    EmbeddedExerciseCacheMetadata,
    LocalEmbeddedExerciseCache,
    LocalEmbeddedExerciseRepository,
)
from calisthenics_recommender.application.embedded_exercise_cache_workflow import (
    build_embedded_exercise_cache,
)
from calisthenics_recommender.application.exercise_text_builder import (
    V1ExerciseTextBuilder,
    build_exercise_text,
)
from calisthenics_recommender.application.query_builder import build_query_text
from calisthenics_recommender.application.query_builder import V1QueryTextBuilder
from calisthenics_recommender.application.recommend_exercises import (
    recommend_exercises,
)
from calisthenics_recommender.domain.exercise import Exercise
from calisthenics_recommender.domain.user_request import UserRequest


class RecordingFakeEmbeddingProvider:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self._provider = FakeEmbeddingProvider(embeddings)
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._provider.embed(text)


def test_local_fake_cache_pipeline_builds_cache_and_recommends_from_it(tmp_path):
    csv_path = tmp_path / "exercises.csv"
    cache_path = tmp_path / "embedded_exercises.jsonl"
    rows = [
        {
            "exercise_id": "pull-up-negative",
            "name": "Pull Up Negative",
            "description": "A controlled eccentric pull-up variation for building strength.",
            "muscle_groups": "Back;Biceps",
            "families": "Pull-up",
            "materials": "Bar",
            "categories": "Upper Body Pull",
        },
        {
            "exercise_id": "body-row",
            "name": "Body Row",
            "description": "A horizontal pulling variation that adds volume with a bar.",
            "muscle_groups": "Back;Biceps",
            "families": "Pull-up",
            "materials": "Bar",
            "categories": "Upper Body Pull",
        },
        {
            "exercise_id": "ring-pull-up",
            "name": "Ring Pull Up",
            "description": "A pull-up variation that requires rings.",
            "muscle_groups": "Back;Biceps",
            "families": "Pull-up",
            "materials": "Rings",
            "categories": "Upper Body Pull",
        },
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "name",
                "exercise_id",
                "description",
                "muscle_groups",
                "families",
                "materials",
                "categories",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    expected_exercises = [
        Exercise(
            exercise_id="pull-up-negative",
            name="Pull Up Negative",
            description="A controlled eccentric pull-up variation for building strength.",
            muscle_groups=["Back", "Biceps"],
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
        ),
        Exercise(
            exercise_id="body-row",
            name="Body Row",
            description="A horizontal pulling variation that adds volume with a bar.",
            muscle_groups=["Back", "Biceps"],
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
        ),
        Exercise(
            exercise_id="ring-pull-up",
            name="Ring Pull Up",
            description="A pull-up variation that requires rings.",
            muscle_groups=["Back", "Biceps"],
            families=["Pull-up"],
            materials=["Rings"],
            categories=["Upper Body Pull"],
        ),
    ]
    user_request = UserRequest(
        target_family="Pull-up",
        goal="I want to build pulling strength and improve pull-ups.",
        current_level="I can do a few strict pull-ups.",
        available_equipment=["Bar"],
    )
    exercise_texts = [build_exercise_text(exercise) for exercise in expected_exercises]
    query_text = build_query_text(user_request)
    embedding_provider = RecordingFakeEmbeddingProvider(
        {
            exercise_texts[0]: [1.0, 0.0, 0.0],
            exercise_texts[1]: [0.95, 0.05, 0.0],
            exercise_texts[2]: [1.0, 0.0, 0.0],
            query_text: [1.0, 0.0, 0.0],
        }
    )

    build_embedded_exercise_cache(
        exercise_repository=CsvExerciseRepository(csv_path),
        embedding_provider=embedding_provider,
        exercise_text_builder=V1ExerciseTextBuilder(),
        cache_writer=LocalEmbeddedExerciseCache(cache_path),
        metadata=EmbeddedExerciseCacheMetadata(
            embedding_model="fake-local",
            embedding_dimension=3,
            text_builder_version="v1",
        ),
    )

    recommendations = recommend_exercises(
        user_request=user_request,
        embedded_exercise_search_repository=JsonlEmbeddedExerciseSearchRepository(
            LocalEmbeddedExerciseRepository(cache_path)
        ),
        embedding_provider=embedding_provider,
        query_text_builder=V1QueryTextBuilder(),
        limit=3,
    )

    assert recommendations
    assert embedding_provider.calls == [*exercise_texts, query_text]

    recommended_names = [recommendation.exercise_name for recommendation in recommendations]

    assert "Pull Up Negative" in recommended_names
    assert "Ring Pull Up" not in recommended_names
    assert all(
        set(recommendation.required_equipment).issubset({"Bar"})
        for recommendation in recommendations
    )

    for recommendation in recommendations:
        assert recommendation.exercise_name
        assert 0 <= recommendation.match_score <= 100
        assert recommendation.reason
        assert recommendation.required_equipment
        assert recommendation.category_family.categories
        assert recommendation.category_family.families
        assert "matched your Pull-up target family through retrieval" in recommendation.reason
        assert "Upper Body Pull categories" in recommendation.reason
        assert "requires Bar" in recommendation.reason
