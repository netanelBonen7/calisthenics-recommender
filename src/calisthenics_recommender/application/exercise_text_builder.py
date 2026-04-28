from __future__ import annotations

from calisthenics_recommender.domain.exercise import Exercise


class V1ExerciseTextBuilder:
    STRATEGY = "v1"

    def build(self, exercise: Exercise) -> str:
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


_V1_EXERCISE_TEXT_BUILDER = V1ExerciseTextBuilder()


def build_exercise_text(exercise: Exercise) -> str:
    return _V1_EXERCISE_TEXT_BUILDER.build(exercise)
