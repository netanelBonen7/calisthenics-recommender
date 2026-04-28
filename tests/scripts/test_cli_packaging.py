from pathlib import Path
import tomllib


def test_pyproject_declares_build_backend_and_cli_entry_points() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject_data["build-system"] == {
        "requires": ["hatchling"],
        "build-backend": "hatchling.build",
    }
    assert pyproject_data["project"]["scripts"] == {
        "build-exercise-cache": (
            "calisthenics_recommender.cli.build_exercise_cache:main"
        ),
        "demo-recommend": "calisthenics_recommender.cli.demo_recommend:main",
        "debug-recommendations": (
            "calisthenics_recommender.cli.debug_recommendations:main"
        ),
        "import-exercises-to-sqlite": (
            "calisthenics_recommender.cli.import_exercises_to_sqlite:main"
        ),
        "process-pending-embedding-updates": (
            "calisthenics_recommender.cli.process_pending_embedding_updates:main"
        ),
    }
