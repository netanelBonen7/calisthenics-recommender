from importlib import import_module
import inspect
from typing import Iterable, Sequence, get_type_hints


def get_exercise_model():
    module = import_module("calisthenics_recommender.domain.exercise")
    return getattr(module, "Exercise")


def get_embedded_exercise_search_result_model():
    module = import_module(
        "calisthenics_recommender.domain.embedded_exercise_search_result"
    )
    return getattr(module, "EmbeddedExerciseSearchResult")


def get_embedded_exercise_search_repository_protocol():
    module = import_module(
        "calisthenics_recommender.ports.embedded_exercise_search_repository"
    )
    return getattr(module, "EmbeddedExerciseSearchRepository")


def valid_search_result():
    Exercise = get_exercise_model()
    EmbeddedExerciseSearchResult = get_embedded_exercise_search_result_model()
    exercise = Exercise(
        name="Pull Up Negative",
        description="A controlled eccentric pull-up variation for building pulling strength.",
        muscle_groups=["Back", "Biceps"],
        families=["Pull-up"],
        materials=["Bar"],
        categories=["Upper Body Pull"],
    )
    return EmbeddedExerciseSearchResult(exercise=exercise, similarity=0.87)


def test_embedded_exercise_search_repository_is_a_runtime_checkable_protocol():
    EmbeddedExerciseSearchResult = get_embedded_exercise_search_result_model()
    EmbeddedExerciseSearchRepository = get_embedded_exercise_search_repository_protocol()

    signature = inspect.signature(EmbeddedExerciseSearchRepository.search)

    assert getattr(EmbeddedExerciseSearchRepository, "_is_protocol", False) is True
    assert getattr(EmbeddedExerciseSearchRepository, "_is_runtime_protocol", False) is True
    assert list(signature.parameters) == [
        "self",
        "query_embedding",
        "available_equipment",
        "limit",
    ]
    assert signature.parameters["query_embedding"].kind is inspect.Parameter.KEYWORD_ONLY
    assert (
        signature.parameters["available_equipment"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["limit"].kind is inspect.Parameter.KEYWORD_ONLY
    assert get_type_hints(EmbeddedExerciseSearchRepository.search) == {
        "query_embedding": Sequence[float],
        "available_equipment": Sequence[str],
        "limit": int,
        "return": Iterable[EmbeddedExerciseSearchResult],
    }


def test_embedded_exercise_search_repository_can_be_implemented_by_a_simple_fake_class():
    EmbeddedExerciseSearchRepository = get_embedded_exercise_search_repository_protocol()
    search_result = valid_search_result()

    class InMemoryEmbeddedExerciseSearchRepository:
        def __init__(self, search_results):
            self._search_results = tuple(search_results)

        def search(self, *, query_embedding, available_equipment, limit):
            return iter(self._search_results[:limit])

    repository = InMemoryEmbeddedExerciseSearchRepository([search_result])

    assert isinstance(repository, EmbeddedExerciseSearchRepository)
    assert list(
        repository.search(
            query_embedding=[1.0, 0.0],
            available_equipment=["Bar"],
            limit=1,
        )
    ) == [search_result]


def test_embedded_exercise_search_repository_can_return_a_one_pass_iterable():
    EmbeddedExerciseSearchRepository = get_embedded_exercise_search_repository_protocol()
    search_result = valid_search_result()

    class GeneratorEmbeddedExerciseSearchRepository:
        def search(self, *, query_embedding, available_equipment, limit):
            yield search_result

    repository = GeneratorEmbeddedExerciseSearchRepository()

    results = repository.search(
        query_embedding=[1.0, 0.0],
        available_equipment=["Bar"],
        limit=1,
    )

    assert isinstance(repository, EmbeddedExerciseSearchRepository)
    assert not isinstance(results, list)
    assert list(results) == [search_result]
