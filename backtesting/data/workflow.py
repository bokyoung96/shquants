from __future__ import annotations

from typing import Iterable, Protocol


class Flow(Protocol):
    name: str

    def run(self, client, args) -> None:
        ...


class FlowRegistry:
    def __init__(self, flows: Iterable[Flow]) -> None:
        self._flows = {flow.name: flow for flow in flows}

    def get(self, name: str) -> Flow:
        try:
            return self._flows[name]
        except KeyError as exc:
            raise ValueError(f"unknown data workflow: {name}") from exc
