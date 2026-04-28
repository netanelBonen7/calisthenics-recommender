from csv import DictWriter
from importlib import import_module
import logging

import pytest


REQUIRED_HEADERS = [
    "exercise_id",
    "name",
    "description",
    "muscle_groups",
    "families",
    "materials",
    "categories",
]


def exercise_id_for(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


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
    normalized_rows = [
        {"exercise_id": exercise_id_for(row["name"]), **row}
        if "name" in row
        else row
        for row in rows
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(normalized_rows)
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

    exercises = list(repository.iter_exercises())

    assert len(exercises) == 1
    assert exercises[0].exercise_id == "pull-up-negative"
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

    exercises = list(repository.iter_exercises())

    assert exercises == [
        Exercise(
            exercise_id="pull-up-negative",
            name="Pull Up Negative",
            description="Controlled eccentric pulling work.",
            muscle_groups=["Back", "Biceps"],
            families=["Pull-up", "Vertical Pull"],
            materials=[],
            categories=["Upper Body Pull", "Strength"],
        ),
        Exercise(
            exercise_id="body-row",
            name="Body Row",
            description="Horizontal pulling variation.",
            muscle_groups=["Back", "Rear Delts"],
            families=["Row"],
            materials=["Bar", "Rings"],
            categories=["Upper Body Pull"],
        ),
    ]


def test_csv_exercise_repository_parses_json_list_cells_from_real_dataset_format(
    tmp_path,
):
    Exercise = get_exercise_model()
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "exercise_id": "360-pull",
                "name": "360° Pull",
                "description": "A dynamic explosive movement rotating around the bar.",
                "muscle_groups": '["Back", "Shoulders", "Biceps", "Core"]',
                "families": '["Pull-up"]',
                "materials": '["Bar"]',
                "categories": '["Upper Body Pull"]',
            }
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    exercises = list(repository.iter_exercises())

    assert exercises == [
        Exercise(
            exercise_id="360-pull",
            name="360° Pull",
            description="A dynamic explosive movement rotating around the bar.",
            muscle_groups=["Back", "Shoulders", "Biceps", "Core"],
            families=["Pull-up"],
            materials=["Bar"],
            categories=["Upper Body Pull"],
        )
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

    exercises = list(repository.iter_exercises())

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
        list(repository.iter_exercises())


def test_csv_exercise_repository_raises_clear_error_for_missing_exercise_id_header(
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
            "categories",
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    with pytest.raises(ValueError, match="Missing required headers|exercise_id"):
        list(repository.iter_exercises())


def test_csv_exercise_repository_raises_clear_error_for_duplicate_exercise_id(
    tmp_path,
):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "exercise_id": "pull-up-negative",
                "name": "Pull Up Negative",
                "description": "Controlled eccentric pulling work.",
                "muscle_groups": "Back; Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            },
            {
                "exercise_id": "pull-up-negative",
                "name": "Negative Pull Up",
                "description": "Same stable id by mistake.",
                "muscle_groups": "Back; Biceps",
                "families": "Pull-up",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            },
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    with pytest.raises(ValueError, match=r"Duplicate exercise_id|row 3|row 2"):
        list(repository.iter_exercises())


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
        list(repository.iter_exercises())


def test_csv_exercise_repository_raises_clear_error_for_truncated_list_field(tmp_path):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv_text(
        tmp_path,
        "\n".join(
            [
                "exercise_id,name,description,muscle_groups,families,materials,categories",
                "body-row,Body Row,Horizontal pulling variation.,Back; Biceps,Row,Bar",
            ]
        ),
    )

    repository = CsvExerciseRepository(csv_path)

    with pytest.raises(ValueError, match="row 2|categories"):
        list(repository.iter_exercises())


def test_csv_exercise_repository_raises_clear_error_for_malformed_json_list_field(
    tmp_path,
):
    CsvExerciseRepository = get_csv_exercise_repository()
    csv_path = write_csv(
        tmp_path,
        [
            {
                "name": "Body Row",
                "description": "Horizontal pulling variation.",
                "muscle_groups": '["Back", "Biceps"',
                "families": "Row",
                "materials": "Bar",
                "categories": "Upper Body Pull",
            }
        ],
    )

    repository = CsvExerciseRepository(csv_path)

    with pytest.raises(ValueError, match=r"row 2|muscle_groups"):
        list(repository.iter_exercises())


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
        list(repository.iter_exercises())


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

    first_result = list(repository.iter_exercises())
    first_result[0].materials.append("Rings")
    second_result = list(repository.iter_exercises())

    assert first_result is not second_result
    assert second_result[0].materials == ["Bar"]


def test_csv_exercise_repository_returns_an_iterable_that_can_be_materialized_with_list(
    tmp_path,
):
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
    exercises = repository.iter_exercises()

    assert not isinstance(exercises, list)
    assert not hasattr(exercises, "append")
    assert [exercise.name for exercise in exercises] == ["Pull Up Negative"]


def test_csv_exercise_repository_streams_rows_and_logs_only_safe_operational_messages(
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
        iterator = iter(repository.iter_exercises())
        first_exercise = next(iterator)

        assert first_exercise.name == "Pull Up Negative"
        assert str(csv_path) in caplog.text
        assert "Starting exercise CSV scan" in caplog.text
        assert "Finished exercise CSV scan" not in caplog.text

        remaining_exercises = list(iterator)

    assert remaining_exercises == []
    assert "Finished exercise CSV scan" in caplog.text
    assert "with 1 exercises" in caplog.text
    assert "A controlled eccentric pull-up variation for building strength." not in caplog.text


def test_csv_exercise_repository_yields_first_valid_row_before_later_row_validation_error(
    tmp_path,
):
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
            },
            {
                "name": "Body Row",
                "description": "Horizontal pulling variation.",
                "muscle_groups": "Back; Biceps",
                "families": "Row",
                "materials": "Bar",
                "categories": " ; ",
            },
        ],
    )
    repository = CsvExerciseRepository(csv_path)

    iterator = iter(repository.iter_exercises())
    first_exercise = next(iterator)

    assert first_exercise.name == "Pull Up Negative"

    with pytest.raises(ValueError, match=r"row 3|categories"):
        next(iterator)
