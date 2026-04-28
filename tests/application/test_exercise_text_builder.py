from importlib import import_module
import inspect
from pathlib import Path
import socket


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_build_exercise_text():
    module = import_module("calisthenics_recommender.application.exercise_text_builder")
    return getattr(module, "build_exercise_text")


def get_v1_exercise_text_builder():
    module = import_module("calisthenics_recommender.application.exercise_text_builder")
    return getattr(module, "V1ExerciseTextBuilder")


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


def test_build_exercise_text_accepts_only_an_exercise_and_returns_expected_text():
    build_exercise_text = get_build_exercise_text()
    exercise = valid_exercise()

    assert list(inspect.signature(build_exercise_text).parameters) == ["exercise"]
    assert build_exercise_text(exercise) == (
        "Exercise name: Pull Up Negative\n"
        "Description: A controlled eccentric pull-up variation for building pulling strength.\n"
        "Muscle groups: Back, Biceps\n"
        "Families: Pull-up, Strength\n"
        "Required equipment: Bar, Rings\n"
        "Categories: Upper Body Pull, Skill Work"
    )


def test_build_exercise_text_includes_all_required_fields_without_model_repr_noise():
    build_exercise_text = get_build_exercise_text()

    text = build_exercise_text(valid_exercise())

    assert "Pull Up Negative" in text
    assert "A controlled eccentric pull-up variation for building pulling strength." in text
    assert "Back, Biceps" in text
    assert "Pull-up, Strength" in text
    assert "Bar, Rings" in text
    assert "Upper Body Pull, Skill Work" in text
    assert "Exercise(" not in text


def test_build_exercise_text_is_pure_and_does_not_touch_files_or_network(monkeypatch):
    build_exercise_text = get_build_exercise_text()
    exercise = valid_exercise()

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    assert build_exercise_text(exercise) == (
        "Exercise name: Pull Up Negative\n"
        "Description: A controlled eccentric pull-up variation for building pulling strength.\n"
        "Muscle groups: Back, Biceps\n"
        "Families: Pull-up, Strength\n"
        "Required equipment: Bar, Rings\n"
        "Categories: Upper Body Pull, Skill Work"
    )


def test_v1_exercise_text_builder_matches_build_exercise_text_exactly():
    build_exercise_text = get_build_exercise_text()
    V1ExerciseTextBuilder = get_v1_exercise_text_builder()
    exercise = valid_exercise()

    assert V1ExerciseTextBuilder().build(exercise) == build_exercise_text(exercise)
