from importlib import import_module

import pytest
from pydantic import ValidationError


def get_user_request_model():
    module = import_module("calisthenics_recommender.domain.user_request")
    return getattr(module, "UserRequest")


def valid_user_request_data():
    return {
        "target_family": "Pull-up",
        "goal": "I want to build pulling strength and unlock harder pull-up variations.",
        "current_level": "I can do 5 strict pull-ups, but the last reps are slow.",
        "available_equipment": ["Bar"],
    }


def test_user_request_can_be_created_with_valid_mvp_fields():
    UserRequest = get_user_request_model()

    user_request = UserRequest(**valid_user_request_data())

    assert user_request.model_dump() == valid_user_request_data()


@pytest.mark.parametrize(
    "missing_field",
    ["target_family", "goal", "current_level", "available_equipment"],
)
def test_user_request_missing_required_fields_raise_validation_error(missing_field):
    UserRequest = get_user_request_model()
    payload = valid_user_request_data()
    payload.pop(missing_field)

    with pytest.raises(ValidationError):
        UserRequest(**payload)


def test_user_request_list_fields_reject_plain_strings():
    UserRequest = get_user_request_model()
    payload = valid_user_request_data()
    payload["available_equipment"] = "Bar"

    with pytest.raises(ValidationError):
        UserRequest(**payload)


@pytest.mark.parametrize("field_name", ["target_family", "goal", "current_level"])
def test_user_request_rejects_empty_important_string_fields(field_name):
    UserRequest = get_user_request_model()
    payload = valid_user_request_data()
    payload[field_name] = ""

    with pytest.raises(ValidationError):
        UserRequest(**payload)
