from __future__ import annotations

from typing import Protocol, runtime_checkable

from calisthenics_recommender.domain.user_request import UserRequest


@runtime_checkable
class QueryTextBuilder(Protocol):
    def build(self, user_request: UserRequest) -> str:
        ...
