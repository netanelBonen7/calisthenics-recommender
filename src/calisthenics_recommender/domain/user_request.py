from pydantic import BaseModel

from calisthenics_recommender.domain.types import NonEmptyString


class UserRequest(BaseModel):
    target_family: NonEmptyString
    goal: NonEmptyString
    current_level: NonEmptyString
    available_equipment: list[str]
