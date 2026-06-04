from __future__ import annotations

from typing import Iterable, Protocol, TypeVar


class Named(Protocol):
    name: str


T = TypeVar("T", bound=Named)


class NamedRegistry:
    def __init__(self, items: Iterable[T], *, label: str = "registry item") -> None:
        self._items = {item.name: item for item in items}
        self.label = label

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            raise ValueError(f"unknown {self.label}: {name}") from exc
