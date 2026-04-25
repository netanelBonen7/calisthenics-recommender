from calisthenics_recommender.domain.exercise import Exercise


def _normalize_equipment_name(equipment_name: str) -> str:
    return equipment_name.strip().lower()


def exercise_matches_equipment(
    exercise: Exercise, available_equipment: list[str]
) -> bool:
    normalized_available_equipment = {
        _normalize_equipment_name(item) for item in available_equipment
    }
    normalized_required_materials = {
        _normalize_equipment_name(item) for item in exercise.materials
    }

    return normalized_required_materials.issubset(normalized_available_equipment)


def filter_exercises_by_equipment(
    exercises: list[Exercise], available_equipment: list[str]
) -> list[Exercise]:
    return [
        exercise
        for exercise in exercises
        if exercise_matches_equipment(exercise, available_equipment)
    ]
