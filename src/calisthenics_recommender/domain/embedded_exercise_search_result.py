from math import isfinite

from pydantic import BaseModel, ConfigDict, field_validator

from calisthenics_recommender.domain.exercise import Exercise


class EmbeddedExerciseSearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    exercise: Exercise
    similarity: float

    @field_validator("similarity")
    @classmethod
    def validate_similarity(cls, similarity: float) -> float:
        if not isfinite(similarity):
            raise ValueError("similarity must be finite")
        if similarity < -1.0 or similarity > 1.0:
            raise ValueError("similarity must be between -1.0 and 1.0")
        return similarity
