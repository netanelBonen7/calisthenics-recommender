from calisthenics_recommender.domain.user_request import UserRequest


def build_query_text(user_request: UserRequest) -> str:
    available_equipment = ", ".join(user_request.available_equipment)

    return "\n".join(
        [
            (
                "The user wants calisthenics exercises related to "
                f"the {user_request.target_family} family."
            ),
            f"The user's goal is: {user_request.goal}",
            f"The user's current level is: {user_request.current_level}",
            f"Available equipment: {available_equipment}.",
            "Recommend suitable calisthenics exercises from the dataset.",
        ]
    )
