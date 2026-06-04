from __future__ import annotations

from typing import Iterable, Protocol


class Workflow(Protocol):
    name: str

    def run(self, client, args) -> None:
        ...


class WorkflowRegistry:
    def __init__(self, workflows: Iterable[Workflow]) -> None:
        self._workflows = {workflow.name: workflow for workflow in workflows}

    def get(self, name: str) -> Workflow:
        try:
            return self._workflows[name]
        except KeyError as exc:
            raise ValueError(f"unknown data workflow: {name}") from exc

