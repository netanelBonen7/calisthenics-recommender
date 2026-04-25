from calisthenics_recommender.domain.exercise import Exercise


def build_exercise_text(exercise: Exercise) -> str:
    muscle_groups = ", ".join(exercise.muscle_groups)
    families = ", ".join(exercise.families)
    materials = ", ".join(exercise.materials)
    categories = ", ".join(exercise.categories)

    return "\n".join(
        [
            f"Exercise name: {exercise.name}",
            f"Description: {exercise.description}",
            f"Muscle groups: {muscle_groups}",
            f"Families: {families}",
            f"Required equipment: {materials}",
            f"Categories: {categories}",
        ]
    )
