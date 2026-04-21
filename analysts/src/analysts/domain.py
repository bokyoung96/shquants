from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ParseQuality(StrEnum):
    HIGH = "high"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class ReportRecord:
    id: int | None
    source: str
    channel: str
    message_id: int
    published_at: str | None
    title: str
    pdf_path: Path
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalDocument:
    source: str
    source_message_id: str
    source_thread_id: str | None
    source_feed: str
    document_kind: str
    title: str
    published_at: str | None
    sender_or_origin: str | None
    mime_type: str
    dedupe_key: str
    raw_path: Path
    normalized_text_path: Path | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    title: str
    content: str
    sections: list[str]
    entities: list[str]
    tickers: list[str]
    routes: list[str]
    parse_quality: ParseQuality
    degraded_reason: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    topic: str
    lane: str
    rationale: str


@dataclass(frozen=True)
class ExtractionPacket:
    source_document_id: int
    report_title: str
    report_channel: str
    message_id: int
    published_at: str | None
    raw_pdf_path: Path
    extraction_quality: str
    extraction_reason: str | None
    preferred_text: str
    text_excerpt: str
    route_hints: list[str]
    entities: list[str]
    tickers: list[str]
    page_previews: list[str] = field(default_factory=list)
    important_pages: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class AnalystSummary:
    lane: str
    topic: str
    headline: str
    executive_summary: str
    key_points: list[str]
    key_numbers: list[str]
    risks: list[str]
    confidence: str
    cited_pages: list[int]
    follow_up_questions: list[str]


@dataclass(frozen=True)
class PipelineRunSummary:
    downloaded: int
    duplicates: int
    ignored: int
    next_offset: int | None


@dataclass(frozen=True)
class PipelineExecution:
    summary: PipelineRunSummary
    processed_files: list[Path]
    summaries: list[AnalystSummary]
