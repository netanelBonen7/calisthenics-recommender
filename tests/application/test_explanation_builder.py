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


def get_build_explanation():
    module = import_module("calisthenics_recommender.application.explanation_builder")
    return getattr(module, "build_explanation")


def valid_user_request():
    UserRequest = get_user_request_model()
    return UserRequest(
        target_family="Pull-up",
        goal="I want to build pulling strength and unlock harder pull-up variations.",
        current_level="I can do 5 strict pull-ups, but the last reps are slow.",
        available_equipment=["Bar", "Rings"],
    )


def valid_exercise():
    Exercise = get_exercise_model()
    return Exercise(
        exercise_id="pull-up-negative",
        name="Pull Up Negative",
        description="A controlled eccentric pull-up variation for building pulling strength.",
        muscle_groups=["Back", "Biceps"],
        families=["Pull-up", "Strength"],
        materials=["Bar", "Rings"],
        categories=["Upper Body Pull", "Skill Work"],
    )


def test_build_explanation_accepts_user_request_and_exercise_and_returns_stable_text():
    build_explanation = get_build_explanation()
    user_request = valid_user_request()
    exercise = valid_exercise()

    assert list(inspect.signature(build_explanation).parameters) == [
        "user_request",
        "exercise",
    ]
    assert build_explanation(user_request, exercise) == (
        "Recommended because it matched your Pull-up target family through retrieval, "
        "belongs to the Pull-up, Strength families, falls under the Upper Body Pull, "
        "Skill Work categories, and requires Bar, Rings."
    )


def test_build_explanation_includes_target_family_family_category_and_equipment():
    build_explanation = get_build_explanation()

    explanation = build_explanation(valid_user_request(), valid_exercise())

    assert "Pull-up target family" in explanation
    assert "Pull-up, Strength families" in explanation
    assert "Upper Body Pull, Skill Work categories" in explanation
    assert "Bar, Rings" in explanation


def test_build_explanation_is_pure_and_does_not_touch_files_or_network(monkeypatch):
    build_explanation = get_build_explanation()
    user_request = valid_user_request()
    exercise = valid_exercise()

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    assert build_explanation(user_request, exercise) == (
        "Recommended because it matched your Pull-up target family through retrieval, "
        "belongs to the Pull-up, Strength families, falls under the Upper Body Pull, "
        "Skill Work categories, and requires Bar, Rings."
    )
