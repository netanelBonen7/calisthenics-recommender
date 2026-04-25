from copy import deepcopy
from importlib import import_module
import inspect
from pathlib import Path
import socket


def get_user_request_model():
    module = import_module("calisthenics_recommender.domain.user_request")
    return getattr(module, "UserRequest")


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_retrieval_result_model():
    module = import_module("calisthenics_recommender.application.retriever")
    return getattr(module, "RetrievalResult")


def get_build_recommendation():
    module = import_module("calisthenics_recommender.application.recommend_exercises")
    return getattr(module, "build_recommendation")


def get_build_recommendations():
    module = import_module("calisthenics_recommender.application.recommend_exercises")
    return getattr(module, "build_recommendations")


def valid_user_request():
    UserRequest = get_user_request_model()
    return UserRequest(
        target_family="Pull-up",
        goal="I want to build pulling strength and unlock harder pull-up variations.",
        current_level="I can do 5 strict pull-ups, but the last reps are slow.",
        available_equipment=["Bar", "Rings"],
    )


def exercise_named(name: str, *, families: list[str], categories: list[str], materials: list[str]):
    Exercise = get_exercise_model()
    return Exercise(
        name=name,
        description=f"{name} description.",
        muscle_groups=["Back"],
        families=families,
        materials=materials,
        categories=categories,
    )


def retrieval_result_for(exercise, score: float):
    RetrievalResult = get_retrieval_result_model()
    return RetrievalResult(exercise=exercise, score=score)


def test_build_recommendation_accepts_user_request_and_retrieval_result():
    build_recommendation = get_build_recommendation()

    assert list(inspect.signature(build_recommendation).parameters) == [
        "user_request",
        "retrieval_result",
    ]


def test_build_recommendation_returns_expected_recommendation_fields():
    build_recommendation = get_build_recommendation()
    user_request = valid_user_request()
    exercise = exercise_named(
        "Pull Up Negative",
        families=["Pull-up", "Strength"],
        categories=["Upper Body Pull", "Skill Work"],
        materials=["Bar", "Rings"],
    )
    retrieval_result = retrieval_result_for(exercise, 0.87)

    recommendation = build_recommendation(user_request, retrieval_result)

    assert recommendation.exercise_name == "Pull Up Negative"
    assert recommendation.match_score == 87
    assert recommendation.reason == (
        "Recommended because it matched your Pull-up target family through retrieval, "
        "belongs to the Pull-up, Strength families, falls under the Upper Body Pull, "
        "Skill Work categories, and requires Bar, Rings."
    )
    assert recommendation.required_equipment == ["Bar", "Rings"]
    assert recommendation.category_family.categories == ["Upper Body Pull", "Skill Work"]
    assert recommendation.category_family.families == ["Pull-up", "Strength"]


def test_build_recommendation_converts_similarity_scores_to_deterministic_match_scores():
    build_recommendation = get_build_recommendation()
    user_request = valid_user_request()
    exercise = exercise_named(
        "Pull Up Negative",
        families=["Pull-up"],
        categories=["Upper Body Pull"],
        materials=["Bar"],
    )

    perfect_match = build_recommendation(
        user_request, retrieval_result_for(exercise, 1.0)
    )
    no_match = build_recommendation(user_request, retrieval_result_for(exercise, 0.0))
    rounded_match = build_recommendation(
        user_request, retrieval_result_for(exercise, 0.875)
    )

    assert perfect_match.match_score == 100
    assert no_match.match_score == 0
    assert rounded_match.match_score == 88


def test_build_recommendations_preserves_retrieval_ranking_order():
    build_recommendations = get_build_recommendations()
    user_request = valid_user_request()
    first = retrieval_result_for(
        exercise_named(
            "Pull Up Negative",
            families=["Pull-up"],
            categories=["Upper Body Pull"],
            materials=["Bar"],
        ),
        0.91,
    )
    second = retrieval_result_for(
        exercise_named(
            "Body Row",
            families=["Pull-up"],
            categories=["Upper Body Pull"],
            materials=["Bar"],
        ),
        0.83,
    )

    recommendations = build_recommendations(user_request, [first, second])

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up Negative",
        "Body Row",
    ]
    assert [recommendation.match_score for recommendation in recommendations] == [91, 83]


def test_response_construction_does_not_mutate_inputs():
    build_recommendations = get_build_recommendations()
    user_request = valid_user_request()
    retrieval_results = [
        retrieval_result_for(
            exercise_named(
                "Pull Up Negative",
                families=["Pull-up"],
                categories=["Upper Body Pull"],
                materials=["Bar"],
            ),
            0.91,
        ),
        retrieval_result_for(
            exercise_named(
                "Body Row",
                families=["Pull-up"],
                categories=["Upper Body Pull"],
                materials=["Bar"],
            ),
            0.83,
        ),
    ]
    original_user_request = deepcopy(user_request)
    original_retrieval_results = deepcopy(retrieval_results)

    build_recommendations(user_request, retrieval_results)

    assert user_request == original_user_request
    assert retrieval_results == original_retrieval_results


def test_response_construction_is_pure_and_does_not_touch_files_or_network(monkeypatch):
    build_recommendations = get_build_recommendations()
    user_request = valid_user_request()
    retrieval_results = [
        retrieval_result_for(
            exercise_named(
                "Pull Up Negative",
                families=["Pull-up"],
                categories=["Upper Body Pull"],
                materials=["Bar"],
            ),
            0.91,
        )
    ]

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    recommendations = build_recommendations(user_request, retrieval_results)

    assert [recommendation.exercise_name for recommendation in recommendations] == [
        "Pull Up Negative"
    ]
