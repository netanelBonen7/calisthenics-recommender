from importlib import import_module

import pytest
from pydantic import ValidationError


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def valid_exercise_data():
    return {
        "name": "Pull Up Negative",
        "description": "A controlled eccentric pull-up variation for building pulling strength.",
        "muscle_groups": ["Back", "Biceps"],
        "families": ["Pull-up"],
        "materials": ["Bar"],
        "categories": ["Upper Body Pull"],
    }


def test_exercise_can_be_created_with_valid_mvp_fields():
    Exercise = get_exercise_model()

    exercise = Exercise(**valid_exercise_data())

    assert exercise.model_dump() == valid_exercise_data()


@pytest.mark.parametrize(
    "missing_field",
    ["name", "description", "muscle_groups", "families", "materials", "categories"],
)
def test_exercise_missing_required_fields_raise_validation_error(missing_field):
    Exercise = get_exercise_model()
    payload = valid_exercise_data()
    payload.pop(missing_field)

    with pytest.raises(ValidationError):
        Exercise(**payload)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("muscle_groups", "Back"),
        ("families", "Pull-up"),
        ("materials", "Bar"),
        ("categories", "Upper Body Pull"),
    ],
)
def test_exercise_list_fields_reject_plain_strings(field_name, bad_value):
    Exercise = get_exercise_model()
    payload = valid_exercise_data()
    payload[field_name] = bad_value

    with pytest.raises(ValidationError):
        Exercise(**payload)


@pytest.mark.parametrize("field_name", ["name", "description"])
def test_exercise_rejects_empty_important_string_fields(field_name):
    Exercise = get_exercise_model()
    payload = valid_exercise_data()
    payload[field_name] = ""

    with pytest.raises(ValidationError):
        Exercise(**payload)
