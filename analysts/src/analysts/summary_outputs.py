from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import ArasConfig
from .domain import AnalystSummary, ExtractionPacket


@dataclass(frozen=True)
class SummaryOutputPaths:
    json_path: Path
    markdown_path: Path


class SummaryArtifactWriter:
    def __init__(self, config: ArasConfig) -> None:
        self.config = config

    def write(self, *, packet: ExtractionPacket, summaries: list[AnalystSummary]) -> SummaryOutputPaths:
        slug = f"report-{packet.source_document_id or packet.message_id}"
        json_path = self.config.paths.processed_dir / f"{slug}-summary.json"
        markdown_path = self.config.paths.processed_dir / f"{slug}-summary.md"
        json_path.write_text(json.dumps(self._json_payload(packet=packet, summaries=summaries), ensure_ascii=False, indent=2) + "\n")
        markdown_path.write_text(self._markdown_payload(packet=packet, summaries=summaries))
        return SummaryOutputPaths(json_path=json_path, markdown_path=markdown_path)

    @staticmethod
    def _json_payload(*, packet: ExtractionPacket, summaries: list[AnalystSummary]) -> dict:
        return {
            "report_title": packet.report_title,
            "message_id": packet.message_id,
            "published_at": packet.published_at,
            "raw_pdf_path": str(packet.raw_pdf_path),
            "extraction_quality": packet.extraction_quality,
            "extraction_reason": packet.extraction_reason,
            "route_hints": packet.route_hints,
            "important_pages": packet.important_pages,
            "page_previews": packet.page_previews,
            "summaries": [
                {
                    "lane": summary.lane,
                    "topic": summary.topic,
                    "headline": summary.headline,
                    "executive_summary": summary.executive_summary,
                    "key_points": summary.key_points,
                    "key_numbers": summary.key_numbers,
                    "risks": summary.risks,
                    "confidence": summary.confidence,
                    "cited_pages": summary.cited_pages,
                    "follow_up_questions": summary.follow_up_questions,
                }
                for summary in summaries
            ],
        }

    @staticmethod
    def _markdown_payload(*, packet: ExtractionPacket, summaries: list[AnalystSummary]) -> str:
        lines = [
            f"# {packet.report_title}",
            "",
            f"- Message ID: {packet.message_id}",
            f"- Published at: {packet.published_at}",
            f"- Raw PDF: `{packet.raw_pdf_path}`",
            f"- Extraction quality: {packet.extraction_quality}",
            f"- Extraction reason: {packet.extraction_reason}",
            f"- Important pages: {', '.join(str(page) for page in packet.important_pages) or 'none'}",
            "",
        ]
        for summary in summaries:
            lines.extend(
                [
                    f"## {summary.lane.title()} analyst",
                    f"- Topic: {summary.topic}",
                    f"- Confidence: {summary.confidence}",
                    f"- Headline: {summary.headline}",
                    f"- Cited pages: {', '.join(str(page) for page in summary.cited_pages) or 'none'}",
                    "",
                    summary.executive_summary,
                    "",
                    "### Key points",
                    *[f"- {item}" for item in summary.key_points],
                    "",
                    "### Key numbers",
                    *[f"- {item}" for item in summary.key_numbers],
                    "",
                    "### Risks",
                    *[f"- {item}" for item in summary.risks],
                    "",
                    "### Follow-up questions",
                    *[f"- {item}" for item in summary.follow_up_questions],
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"
