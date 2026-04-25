from calisthenics_recommender.domain.exercise import Exercise
from calisthenics_recommender.domain.user_request import UserRequest


def build_explanation(user_request: UserRequest, exercise: Exercise) -> str:
    details: list[str] = []

    if exercise.families:
        details.append(f"belongs to the {_format_items(exercise.families)} families")

    if exercise.categories:
        details.append(
            f"falls under the {_format_items(exercise.categories)} categories"
        )

    if exercise.materials:
        details.append(f"requires {_format_items(exercise.materials)}")

    prefix = (
        "Recommended because it matched your "
        f"{user_request.target_family} target family through retrieval"
    )

    if not details:
        return f"{prefix}."

    if len(details) == 1:
        return f"{prefix} and {details[0]}."

    return f"{prefix}, {', '.join(details[:-1])}, and {details[-1]}."


def _format_items(items: list[str]) -> str:
    return ", ".join(items)
