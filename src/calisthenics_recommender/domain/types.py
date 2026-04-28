from typing import Annotated

from pydantic import StringConstraints


NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ExerciseId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
]
