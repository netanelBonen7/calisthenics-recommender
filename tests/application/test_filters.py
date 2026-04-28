from importlib import import_module
import inspect


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_exercise_matches_equipment():
    module = import_module("calisthenics_recommender.application.filters")
    return getattr(module, "exercise_matches_equipment")


def get_filter_exercises_by_equipment():
    module = import_module("calisthenics_recommender.application.filters")
    return getattr(module, "filter_exercises_by_equipment")


def exercise_with_materials(name: str, materials: list[str]):
    Exercise = get_exercise_model()
    return Exercise(
        exercise_id=name.lower().replace(" ", "-"),
        name=name,
        description=f"{name} description.",
        muscle_groups=["Back"],
        families=["Pull-up"],
        materials=materials,
        categories=["Upper Body Pull"],
    )


def test_exercise_matches_equipment_accepts_expected_arguments():
    exercise_matches_equipment = get_exercise_matches_equipment()

    assert list(inspect.signature(exercise_matches_equipment).parameters) == [
        "exercise",
        "available_equipment",
    ]


def test_exercise_matches_equipment_returns_false_when_required_equipment_is_missing():
    exercise_matches_equipment = get_exercise_matches_equipment()
    exercise = exercise_with_materials("Ring Row", ["Rings"])

    assert exercise_matches_equipment(exercise, ["Bar"]) is False


def test_exercise_matches_equipment_returns_true_when_required_equipment_is_available():
    exercise_matches_equipment = get_exercise_matches_equipment()
    exercise = exercise_with_materials("Pull Up", ["Bar"])

    assert exercise_matches_equipment(exercise, ["Bar"]) is True


def test_exercise_matches_equipment_requires_all_materials_for_multi_equipment_exercises():
    exercise_matches_equipment = get_exercise_matches_equipment()
    exercise = exercise_with_materials("Muscle Up Transition", ["Bar", "Rings"])

    assert exercise_matches_equipment(exercise, ["Bar"]) is False
    assert exercise_matches_equipment(exercise, ["Bar", "Rings"]) is True


def test_exercise_matches_equipment_returns_true_when_no_materials_are_required():
    exercise_matches_equipment = get_exercise_matches_equipment()
    exercise = exercise_with_materials("Hollow Body Hold", [])

    assert exercise_matches_equipment(exercise, ["Bar"]) is True


def test_exercise_matches_equipment_is_case_insensitive_and_trims_surrounding_whitespace():
    exercise_matches_equipment = get_exercise_matches_equipment()
    exercise = exercise_with_materials("Pull Up", [" Bar ", "rings"])

    assert exercise_matches_equipment(exercise, ["bar", " Rings "]) is True


def test_filter_exercises_by_equipment_keeps_only_matching_exercises_in_original_order():
    filter_exercises_by_equipment = get_filter_exercises_by_equipment()
    exercises = [
        exercise_with_materials("Pull Up", ["Bar"]),
        exercise_with_materials("Ring Dip", ["Rings"]),
        exercise_with_materials("Hollow Body Hold", []),
    ]

    filtered = filter_exercises_by_equipment(exercises, ["Bar"])

    assert [exercise.name for exercise in filtered] == [
        "Pull Up",
        "Hollow Body Hold",
    ]
