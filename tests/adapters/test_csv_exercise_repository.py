from csv import DictWriter
from importlib import import_module
import logging

import pytest


REQUIRED_HEADERS = [
    "name",
    "description",
    "muscle_groups",
    "families",
    "materials",
    "categories",
]


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_exercise_repository_protocol():
    module = import_module("calisthenics_recommender.ports.exercise_repository")
    return getattr(module, "ExerciseRepository")


def get_csv_exercise_repository():
    module = import_module("calisthenics_recommender.adapters.csv_exercise_repository")
    return getattr(module, "CsvExerciseRepository")


def write_csv(tmp_path, rows, headers=REQUIRED_HEADERS):
    csv_path = tmp_path / "exercises.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_csv_text(tmp_path, content: str):
    csv_path = tmp_path / "exercises.csv"
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


def test_csv_exercise_repository_implements_exercise_repository_protocol(tmp_path):
    ExerciseRepository = get_exercise_repository_protocol()
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "name": "Pull Up Negative",
                "description": "A controlled eccentric pull-up variation.",
                "muscle_groups": "Back; Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            }
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    assert isinstance(repository, ExerciseRepository)


def test_csv_exercise_repository_accepts_string_csv_path(tmp_path):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "name": "Pull Up Negative",
                "description": "A controlled eccentric pull-up variation.",
                "muscle_groups": "Back; Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            }
        ],
    )

    repository = CsvExerciseRepository(str(csv_path))

    exercises = repository.list_exercises()

    assert len(exercises) == 1
    assert exercises[0].name == "Pull Up Negative"


def test_csv_exercise_repository_loads_exercises_and_parses_list_fields(tmp_path):
    Exercise = get_exercise_model()
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "name": " Pull Up Negative ",
                "description": " Controlled eccentric pulling work. ",
                "muscle_groups": " Back ; Biceps ; ",
                "families": " Pull-up ; Vertical Pull ",
                "materials": "",
                "categories": " Upper Body Pull ; Strength ",
            },
            {
                "name": "Body Row",
                "description": "Horizontal pulling variation.",
                "muscle_groups": "Back;Rear Delts",
                "families": "Row",
                "materials": "Bar; Rings",
                "categories": "Upper Body Pull",
            },
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    exercises = repository.list_exercises()

    assert exercises == [
        Exercise(
            name="Pull Up Negative",
            description="Controlled eccentric pulling work.",
            muscle_groups=["Back", "Biceps"],
            families=["Pull-up", "Vertical Pull"],
            materials=[],
            categories=["Upper Body Pull", "Strength"],
        ),
        Exercise(
            name="Body Row",
            description="Horizontal pulling variation.",
            muscle_groups=["Back", "Rear Delts"],
            families=["Row"],
            materials=["Bar", "Rings"],
            categories=["Upper Body Pull"],
        ),
    ]


def test_csv_exercise_repository_accepts_empty_materials_as_empty_list(tmp_path):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "name": "Pull Up Negative",
                "description": "Controlled eccentric pulling work.",
                "muscle_groups": "Back; Biceps",
                "families": "Pull-up",
                "materials": " ; ",
                "categories": "Upper Body Pull",
            }
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    exercises = repository.list_exercises()

    assert exercises[0].materials == []


def test_csv_exercise_repository_raises_clear_error_for_missing_required_headers(
    tmp_path,
):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        rows=[],
        headers=[
            "name",
            "description",
            "muscle_groups",
            "families",
            "materials",
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    with pytest.raises(ValueError, match="Missing required headers|categories"):
        repository.list_exercises()


def test_csv_exercise_repository_raises_clear_error_for_invalid_name_row(tmp_path):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "name": "   ",
                "description": "Controlled eccentric pulling work.",
                "muscle_groups": "Back; Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            }
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    with pytest.raises(ValueError, match="row 2|name"):
        repository.list_exercises()


def test_csv_exercise_repository_raises_clear_error_for_truncated_list_field(tmp_path):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv_text(
        tmp_path,
        "\n".join(
            [
                "name,description,muscle_groups,families,materials,categories",
                "Body Row,Horizontal pulling variation.,Back; Biceps,Row,Bar",
            ]
        ),
    )

    repository = CsvExerciseRepository(csv_path)

    with pytest.raises(ValueError, match="row 2|categories"):
        repository.list_exercises()


@pytest.mark.parametrize("field_name", ["muscle_groups", "families", "categories"])
def test_csv_exercise_repository_raises_clear_error_for_empty_required_list_fields(
    tmp_path, field_name
):
    CsvExerciseRepository = get_csv_exercise_repository()
    row = {
        "name": "Body Row",
        "description": "Horizontal pulling variation.",
        "muscle_groups": "Back; Biceps",
        "families": "Row",
        "materials": "Bar",
        "categories": "Upper Body Pull",
    }
    row[field_name] = " ; "
    csv_path = write_csv(tmp_path, [row])

    repository = CsvExerciseRepository(csv_path)

    with pytest.raises(ValueError, match=rf"row 2|{field_name}"):
        repository.list_exercises()


def test_csv_exercise_repository_returns_defensive_copies(tmp_path):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "name": "Pull Up Negative",
                "description": "A controlled eccentric pull-up variation.",
                "muscle_groups": "Back; Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            }
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    first_result = repository.list_exercises()
    first_result[0].materials.append("Rings")
    second_result = repository.list_exercises()

    assert first_result is not second_result
    assert second_result[0].materials == ["Bar"]


def test_csv_exercise_repository_logs_only_safe_operational_counts(
    tmp_path, caplog
):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "name": "Pull Up Negative",
                "description": "A controlled eccentric pull-up variation for building strength.",
                "muscle_groups": "Back; Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            }
        ],
    )
    repository = CsvExerciseRepository(csv_path)

    with caplog.at_level(logging.INFO):
        exercises = repository.list_exercises()

    assert len(exercises) == 1
    assert str(csv_path) in caplog.text
    assert "Loaded 1 exercises" in caplog.text
    assert "A controlled eccentric pull-up variation for building strength." not in caplog.text
