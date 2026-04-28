from importlib import import_module
import inspect
from pathlib import Path
import socket


def get_user_request_model():
    module = import_module("calisthenics_recommender.domain.user_request")
    return getattr(module, "UserRequest")


def get_build_query_text():
    module = import_module("calisthenics_recommender.application.query_builder")
    return getattr(module, "build_query_text")


def get_v1_query_text_builder():
    module = import_module("calisthenics_recommender.application.query_builder")
    return getattr(module, "V1QueryTextBuilder")


def valid_user_request():
    UserRequest = get_user_request_model()
    return UserRequest(
        target_family="Pull-up",
        goal="I want to build pulling strength and unlock harder pull-up variations.",
        current_level="I can do 5 strict pull-ups, but the last reps are slow.",
        available_equipment=["Bar", "Rings"],
    )


def test_build_query_text_accepts_only_a_user_request_and_returns_expected_text():
    build_query_text = get_build_query_text()
    user_request = valid_user_request()

    assert list(inspect.signature(build_query_text).parameters) == ["user_request"]
    assert build_query_text(user_request) == (
        "The user wants calisthenics exercises related to the Pull-up family.\n"
        "The user's goal is: I want to build pulling strength and unlock harder pull-up variations.\n"
        "The user's current level is: I can do 5 strict pull-ups, but the last reps are slow.\n"
        "Available equipment: Bar, Rings.\n"
        "Recommend suitable calisthenics exercises from the dataset."
    )


def test_build_query_text_includes_all_required_fields_without_model_repr_noise():
    build_query_text = get_build_query_text()

    text = build_query_text(valid_user_request())

    assert "Pull-up" in text
    assert "I want to build pulling strength and unlock harder pull-up variations." in text
    assert "I can do 5 strict pull-ups, but the last reps are slow." in text
    assert "Bar, Rings" in text
    assert "UserRequest(" not in text


def test_build_query_text_is_pure_and_does_not_touch_files_or_network(monkeypatch):
    build_query_text = get_build_query_text()
    user_request = valid_user_request()

    def fail(*args, **kwargs):
        raise AssertionError("unexpected side effect")

    monkeypatch.setattr("builtins.open", fail)
    monkeypatch.setattr(Path, "open", fail)
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    assert build_query_text(user_request) == (
        "The user wants calisthenics exercises related to the Pull-up family.\n"
        "The user's goal is: I want to build pulling strength and unlock harder pull-up variations.\n"
        "The user's current level is: I can do 5 strict pull-ups, but the last reps are slow.\n"
        "Available equipment: Bar, Rings.\n"
        "Recommend suitable calisthenics exercises from the dataset."
    )


def test_v1_query_text_builder_matches_build_query_text_exactly():
    build_query_text = get_build_query_text()
    V1QueryTextBuilder = get_v1_query_text_builder()
    user_request = valid_user_request()

    assert V1QueryTextBuilder().build(user_request) == build_query_text(user_request)
