from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from .chunking import TextChunker
from .config import ArasConfig
from .domain import ExtractionPacket, ParsedDocument, ReportRecord, RouteDecision
from .embeddings import EmbeddingArtifactBuilder
from .extraction import SummaryReadyExtractor
from .pdf_images import PdfImageExtractor
from .page_selection import ImportantPageSelector
from .pdf_text import PdfTextExtractor


@dataclass(frozen=True)
class PdfIngestionResult:
    packet: ExtractionPacket
    processed_files: list[Path]


class PdfIngestionPipeline:
    def __init__(self, config: ArasConfig) -> None:
        self.config = config
        self.summary_extractor = SummaryReadyExtractor(config)
        self.pdf_text = PdfTextExtractor(config)
        self.pdf_images = PdfImageExtractor(config)
        self.chunker = TextChunker(config)
        self.embeddings = EmbeddingArtifactBuilder(config)
        self.page_selector = ImportantPageSelector(config)

    def ingest(
        self,
        *,
        report: ReportRecord,
        parsed: ParsedDocument,
        routes: list[RouteDecision],
    ) -> PdfIngestionResult:
        slug = f"report-{report.id or report.message_id}"
        fallback_text = str(report.metadata.get('telegram_caption_text') or report.title or '')
        text_result = self.pdf_text.extract(pdf_path=report.pdf_path, slug=slug, fallback_text=fallback_text)
        image_metadata, images_path = self.pdf_images.extract_metadata(pdf_path=report.pdf_path, slug=slug)
        chunks, chunks_path = self.chunker.chunk_text(text=text_result.full_text, slug=slug)
        _, embeddings_path = self.embeddings.build_pending_records(chunks=chunks, slug=slug)
        selected_pages, important_pages_path = self.page_selector.select(page_texts=text_result.pages, image_metadata=image_metadata, slug=slug)
        selected_preview_map = self.pdf_images.render_previews_for_pages(
            pdf_path=report.pdf_path,
            slug=slug,
            page_numbers=[item.page_number for item in selected_pages],
        )

        enhanced_parsed = ParsedDocument(
            title=parsed.title,
            content=text_result.full_text,
            sections=parsed.sections if parsed.sections else [text_result.full_text] if text_result.full_text else [],
            entities=parsed.entities,
            tickers=parsed.tickers,
            routes=parsed.routes,
            parse_quality=parsed.parse_quality,
            degraded_reason=parsed.degraded_reason,
        )
        enriched_report = ReportRecord(
            id=report.id,
            source=report.source,
            channel=report.channel,
            message_id=report.message_id,
            published_at=report.published_at,
            title=report.title,
            pdf_path=report.pdf_path,
            content=report.content,
            metadata={**report.metadata, "page_previews": [selected_preview_map.get(item.page_number) or item.preview_path for item in selected_pages if (selected_preview_map.get(item.page_number) or item.preview_path)], "important_pages": [item.page_number for item in selected_pages]},
        )
        packet = self.summary_extractor.build_packet(report=enriched_report, parsed=enhanced_parsed, routes=routes)
        packet = replace(
            packet,
            extraction_quality=text_result.quality,
            extraction_reason=text_result.reason,
            preferred_text=text_result.full_text or packet.preferred_text,
            text_excerpt=(text_result.full_text or packet.preferred_text)[: self.config.summary.max_input_chars].strip(),
        )
        summary_artifacts = self.summary_extractor.write_artifacts(packet)
        return PdfIngestionResult(
            packet=packet,
            processed_files=[
                text_result.fulltext_path,
                text_result.metadata_path,
                images_path,
                important_pages_path,
                chunks_path,
                embeddings_path,
                summary_artifacts.raw_text_path,
                summary_artifacts.summary_input_path,
            ],
        )
