from __future__ import annotations

import re
from dataclasses import dataclass

from .domain import ParsedDocument, RouteDecision

_ROUTE_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("sector", "semiconductors", ("ai", "hbm", "nvidia", "packaging", "tsmc")),
    ("macro", "rates", ("curve", "duration", "fed", "treasury", "yields")),
)


@dataclass(frozen=True)
class TaskRouter:
    def route(self, parsed: ParsedDocument) -> list[RouteDecision]:
        haystack = self._normalize(parsed)
        decisions: list[RouteDecision] = []

        for lane, topic, keywords in _ROUTE_RULES:
            matched = [keyword for keyword in keywords if keyword in haystack]
            if matched:
                decisions.append(
                    RouteDecision(
                        topic=topic,
                        lane=lane,
                        rationale=f"matched_keywords: {', '.join(matched)}",
                    )
                )

        if decisions:
            return decisions

        return [RouteDecision(topic="general", lane="macro", rationale="fallback:no_taxonomy_match")]

    @staticmethod
    def _normalize(parsed: ParsedDocument) -> set[str]:
        text = " ".join(
            part.strip().lower()
            for part in [parsed.title, parsed.content, *parsed.sections, *parsed.entities, *parsed.tickers]
            if part.strip()
        )
        return set(re.findall(r"[a-z0-9]+", text))
