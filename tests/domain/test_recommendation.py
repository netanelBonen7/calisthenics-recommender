from importlib import import_module

import pytest
from pydantic import ValidationError


def get_recommendation_model():
    module = import_module("calisthenics_recommender.domain.recommendation")
    return getattr(module, "Recommendation")


def valid_recommendation_data(match_score=87):
    return {
        "exercise_name": "Pull Up Negative",
        "match_score": match_score,
        "reason": "Recommended because it belongs to the Pull-up family and matches your available equipment.",
        "required_equipment": ["Bar"],
        "category_family": {
            "categories": ["Upper Body Pull"],
            "families": ["Pull-up"],
        },
    }


@pytest.mark.parametrize("match_score", [0, 87, 87.5, 100])
def test_recommendation_can_be_created_with_valid_mvp_fields(match_score):
    Recommendation = get_recommendation_model()

    recommendation = Recommendation(**valid_recommendation_data(match_score=match_score))

    assert recommendation.model_dump() == valid_recommendation_data(match_score=match_score)


@pytest.mark.parametrize(
    "missing_field",
    [
        "exercise_name",
        "match_score",
        "reason",
        "required_equipment",
        "category_family",
    ],
)
def test_recommendation_missing_required_fields_raise_validation_error(missing_field):
    Recommendation = get_recommendation_model()
    payload = valid_recommendation_data()
    payload.pop(missing_field)

    with pytest.raises(ValidationError):
        Recommendation(**payload)


def test_recommendation_requires_category_family_to_include_categories_and_families():
    Recommendation = get_recommendation_model()
    payload = valid_recommendation_data()
    payload["category_family"] = {"categories": ["Upper Body Pull"]}

    with pytest.raises(ValidationError):
        Recommendation(**payload)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("required_equipment", "Bar"),
        ("category_family", "Pull-up"),
    ],
)
def test_recommendation_list_and_object_fields_reject_plain_strings(field_name, bad_value):
    Recommendation = get_recommendation_model()
    payload = valid_recommendation_data()
    payload[field_name] = bad_value

    with pytest.raises(ValidationError):
        Recommendation(**payload)


@pytest.mark.parametrize(
    ("nested_field", "bad_value"),
    [("categories", "Upper Body Pull"), ("families", "Pull-up")],
)
def test_recommendation_category_family_nested_list_fields_reject_plain_strings(
    nested_field, bad_value
):
    Recommendation = get_recommendation_model()
    payload = valid_recommendation_data()
    payload["category_family"][nested_field] = bad_value

    with pytest.raises(ValidationError):
        Recommendation(**payload)


@pytest.mark.parametrize("field_name", ["exercise_name", "reason"])
def test_recommendation_rejects_empty_important_string_fields(field_name):
    Recommendation = get_recommendation_model()
    payload = valid_recommendation_data()
    payload[field_name] = ""

    with pytest.raises(ValidationError):
        Recommendation(**payload)


@pytest.mark.parametrize("match_score", [-1, -0.1, 100.1, 101])
def test_recommendation_rejects_match_score_outside_0_to_100(match_score):
    Recommendation = get_recommendation_model()
    payload = valid_recommendation_data(match_score=match_score)

    with pytest.raises(ValidationError):
        Recommendation(**payload)
