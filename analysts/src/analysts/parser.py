from __future__ import annotations

import re
from dataclasses import dataclass

from .domain import ParseQuality, ParsedDocument, ReportRecord

_SECTION_SPLIT_RE = re.compile(r"\n\s*\n+")
_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+[A-Z][a-z]+)*\b")
_ENTITY_STOP_WORDS = {
    "AI",
    "Executive Summary",
    "Macro",
    "Notes",
    "Risks",
}


@dataclass(frozen=True)
class DocumentParser:
    def parse(self, report: ReportRecord) -> ParsedDocument:
        text, parse_quality, degraded_reason = self._extract_text(report)
        sections = self._split_sections(text)
        tickers = self._extract_tickers(text)
        return ParsedDocument(
            title=report.title,
            content=text,
            sections=sections,
            entities=self._extract_entities(text, excluded=tickers),
            tickers=tickers,
            routes=[],
            parse_quality=parse_quality,
            degraded_reason=degraded_reason,
        )

    @staticmethod
    def _extract_text(report: ReportRecord) -> tuple[str, ParseQuality, str | None]:
        if report.content.strip():
            return report.content.strip(), ParseQuality.HIGH, None

        payload = report.pdf_path.read_bytes()
        try:
            text = payload.decode("utf-8").strip()
        except UnicodeDecodeError:
            return "", ParseQuality.DEGRADED, "unable_to_decode_pdf_payload"

        if not text or "\x00" in text:
            return "", ParseQuality.DEGRADED, "unable_to_decode_pdf_payload"

        return text, ParseQuality.HIGH, None

    @staticmethod
    def _split_sections(text: str) -> list[str]:
        if not text:
            return []
        return [section.strip() for section in _SECTION_SPLIT_RE.split(text) if section.strip()]

    @staticmethod
    def _extract_tickers(text: str) -> list[str]:
        seen: set[str] = set()
        tickers: list[str] = []
        for ticker in _TICKER_RE.findall(text):
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
        return tickers

    @staticmethod
    def _extract_entities(text: str, *, excluded: list[str]) -> list[str]:
        if not text:
            return []

        seen: set[str] = set()
        entities: list[str] = []
        for candidate in _ENTITY_RE.findall(text):
            if candidate in _ENTITY_STOP_WORDS:
                continue
            if candidate in excluded:
                continue
            if len(candidate) == 1:
                continue
            if candidate not in seen:
                seen.add(candidate)
                entities.append(candidate)
        return sorted(entities)
