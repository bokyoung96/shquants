from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import ArasConfig
from .domain import AnalystSummary, CanonicalDocument, ExtractionPacket, PipelineExecution, PipelineRunSummary, ReportRecord
from .extraction import SummaryReadyExtractor
from .sources.telegram.fetcher import TelegramFetcher
from .pdf_ingest import PdfIngestionPipeline
from .summary_outputs import SummaryArtifactWriter
from .raw_reports import RawReportCatalog
from .parser import DocumentParser
from .router import TaskRouter
from .storage import SqliteArasStore
from .summarizer import CodexAnalystSummarizer


@dataclass(frozen=True)
class ArasPipeline:
    client: Any
    store: SqliteArasStore
    config: ArasConfig
    summarizer: CodexAnalystSummarizer | None = None

    def run_once(self, *, channel: str) -> PipelineExecution:
        fetcher = TelegramFetcher(client=self.client, store=self.store, config=self.config)
        batch = fetcher.poll_once(channel=channel)
        processed_files: list[Path] = []
        summaries: list[AnalystSummary] = []

        for report in batch.downloaded:
            stored_report = self._hydrate_report(report)
            result = self.summarize_report(stored_report)
            processed_files.extend(result.processed_files)
            summaries.extend(result.summaries)

        return PipelineExecution(
            summary=PipelineRunSummary(
                downloaded=len(batch.downloaded),
                duplicates=len(batch.skipped_duplicates),
                ignored=len(batch.ignored_updates),
                next_offset=batch.next_offset,
            ),
            processed_files=processed_files,
            summaries=summaries,
        )

    def summarize_latest(self, *, channel: str) -> PipelineExecution:
        report = self.store.get_latest_report(channel)
        if report is None:
            report = RawReportCatalog(raw_dir=self.config.paths.telegram_raw_dir, channel=channel).latest_report()
        if report is None:
            report = RawReportCatalog(raw_dir=self.config.paths.raw_dir, channel=channel).latest_report()
        if report is None:
            raise RuntimeError(f"No stored or raw report found for channel {channel}")
        return self.summarize_report(report)

    def summarize_report(self, report: ReportRecord) -> PipelineExecution:
        report = self._resolve_report_path(report)
        parsed = DocumentParser().parse(report)
        routes = TaskRouter().route(parsed)
        ingestion = PdfIngestionPipeline(self.config).ingest(report=report, parsed=parsed, routes=routes)
        summaries = self._build_summaries(ingestion.packet)
        outputs = SummaryArtifactWriter(self.config).write(packet=ingestion.packet, summaries=summaries)

        return PipelineExecution(
            summary=PipelineRunSummary(downloaded=0, duplicates=0, ignored=0, next_offset=report.message_id),
            processed_files=[*ingestion.processed_files, outputs.json_path, outputs.markdown_path],
            summaries=summaries,
        )

    def summarize_canonical(self, document: CanonicalDocument) -> PipelineExecution:
        if document.mime_type == "application/pdf":
            report = ReportRecord(
                id=None,
                source=document.source,
                channel=document.source_feed,
                message_id=document.source_message_id,  # type: ignore[arg-type]
                published_at=document.published_at,
                title=document.title,
                pdf_path=document.raw_path,
                content="",
                metadata=document.metadata,
            )
            return self.summarize_report(report)

        text_path = document.normalized_text_path or document.raw_path
        report = ReportRecord(
            id=None,
            source=document.source,
            channel=document.source_feed,
            message_id=document.source_message_id,  # type: ignore[arg-type]
            published_at=document.published_at,
            title=document.title,
            pdf_path=document.raw_path,
            content=text_path.read_text(),
            metadata={**document.metadata, "telegram_caption_text": document.title},
        )
        parsed = DocumentParser().parse(report)
        routes = TaskRouter().route(parsed)
        packet = SummaryReadyExtractor(self.config).build_packet(report=report, parsed=parsed, routes=routes)
        summaries = self._build_summaries(packet)
        outputs = SummaryArtifactWriter(self.config).write(packet=packet, summaries=summaries)
        artifacts = SummaryReadyExtractor(self.config).write_artifacts(packet)
        return PipelineExecution(
            summary=PipelineRunSummary(downloaded=0, duplicates=0, ignored=0, next_offset=document.source_message_id),  # type: ignore[arg-type]
            processed_files=[artifacts.raw_text_path, artifacts.summary_input_path, outputs.json_path, outputs.markdown_path],
            summaries=summaries,
        )

    def _build_summaries(self, packet: ExtractionPacket) -> list[AnalystSummary]:
        summarizer = self.summarizer or CodexAnalystSummarizer(config=self.config, base_dir=self.config.paths.base_dir)
        return [
            summarizer.summarize(packet=packet, lane=lane, topic=topic)
            for lane, topic in summarizer.lane_plan(packet)
        ]

    def _hydrate_report(self, report: ReportRecord) -> ReportRecord:
        file_unique_id = str(report.metadata["file_unique_id"])
        for stored in self.store.list_reports():
            if str(stored.metadata.get("file_unique_id")) == file_unique_id:
                return stored
        return report

    def _resolve_report_path(self, report: ReportRecord) -> ReportRecord:
        if report.pdf_path.is_absolute() or report.pdf_path.exists():
            return report
        resolved = self.config.paths.base_dir / report.pdf_path
        if resolved.exists():
            return replace(report, pdf_path=resolved)
        if report.pdf_path.parts and report.pdf_path.parts[0] == "analysts":
            trimmed = Path(*report.pdf_path.parts[1:])
            trimmed_resolved = self.config.paths.base_dir / trimmed
            if trimmed_resolved.exists():
                return replace(report, pdf_path=trimmed_resolved)
        fallback_matches = sorted(self.config.paths.telegram_raw_dir.glob(f"{report.message_id}-*"))
        if not fallback_matches:
            fallback_matches = sorted(self.config.paths.raw_dir.glob(f"{report.message_id}-*"))
        if fallback_matches:
            return replace(report, pdf_path=fallback_matches[-1])
        return replace(report, pdf_path=resolved)
