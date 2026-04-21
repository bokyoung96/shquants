from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ArasConfig
from .domain import ExtractionPacket, ParsedDocument, ReportRecord, RouteDecision

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractionArtifacts:
    packet: ExtractionPacket
    raw_text_path: Path
    summary_input_path: Path


@dataclass(frozen=True)
class SummaryReadyExtractor:
    config: ArasConfig

    def build_packet(
        self,
        *,
        report: ReportRecord,
        parsed: ParsedDocument,
        routes: list[RouteDecision],
    ) -> ExtractionPacket:
        preferred_text, quality, reason = self._preferred_text(report=report, parsed=parsed)
        excerpt = preferred_text[: self.config.summary.max_input_chars].strip()
        return ExtractionPacket(
            source_document_id=report.id or 0,
            report_title=report.title,
            report_channel=report.channel,
            message_id=report.message_id,
            published_at=report.published_at,
            raw_pdf_path=report.pdf_path,
            extraction_quality=quality,
            extraction_reason=reason,
            preferred_text=preferred_text,
            text_excerpt=excerpt,
            route_hints=[f"{route.lane}:{route.topic}" for route in routes],
            entities=parsed.entities,
            tickers=parsed.tickers,
            page_previews=list(report.metadata.get('page_previews', [])),
            important_pages=list(report.metadata.get('important_pages', [])),
        )

    def write_artifacts(self, packet: ExtractionPacket) -> ExtractionArtifacts:
        slug = self._slug(packet)
        raw_text_path = self.config.paths.processed_dir / f"{slug}-raw-text.txt"
        summary_input_path = self.config.paths.processed_dir / f"{slug}-summary-input.json"
        raw_text_path.write_text(packet.preferred_text + ("\n" if packet.preferred_text and not packet.preferred_text.endswith("\n") else ""))
        summary_input_path.write_text(json.dumps({
            "report_title": packet.report_title,
            "report_channel": packet.report_channel,
            "message_id": packet.message_id,
            "published_at": packet.published_at,
            "raw_pdf_path": str(packet.raw_pdf_path),
            "extraction_quality": packet.extraction_quality,
            "extraction_reason": packet.extraction_reason,
            "text_excerpt": packet.text_excerpt,
            "route_hints": packet.route_hints,
            "entities": packet.entities,
            "tickers": packet.tickers,
            "page_previews": packet.page_previews,
            "important_pages": packet.important_pages,
        }, ensure_ascii=False, indent=2) + "\n")
        return ExtractionArtifacts(packet=packet, raw_text_path=raw_text_path, summary_input_path=summary_input_path)

    @staticmethod
    def _slug(packet: ExtractionPacket) -> str:
        return f"report-{packet.source_document_id or packet.message_id}"

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(cleaned_lines).strip()

    def _preferred_text(self, *, report: ReportRecord, parsed: ParsedDocument) -> tuple[str, str, str | None]:
        if parsed.content.strip():
            return self._clean_text(parsed.content), parsed.parse_quality.value, parsed.degraded_reason

        metadata_text = str(report.metadata.get("telegram_caption_text") or "").strip()
        title_text = str(report.title).strip()
        fallback_parts = [part for part in [metadata_text, title_text] if part]
        fallback_text = self._clean_text("\n\n".join(fallback_parts))
        if fallback_text:
            reason = parsed.degraded_reason or "used_title_or_caption_fallback"
            return fallback_text, "fallback", reason

        return "", parsed.parse_quality.value, parsed.degraded_reason
