from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ArasConfig
from .pdf_images import PdfImageMetadata
from .pdf_text import PdfPageText

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
_TABLE_ROW_RE = re.compile(r"(?:\b\d[\d,\.]*%?\b\s+){3,}")
_SECTION_HINTS = ("Executive Summary", "목차", "Summary", "결론", "투자포인트")
_CHART_HINTS = ("자료:", "자료 ", "YoY", "QoQ", "CAGR", "Spot Price", "Price", "증가율", "비중")


@dataclass(frozen=True)
class ImportantPage:
    page_number: int
    score: int
    reasons: list[str]
    preview_path: str | None


class ImportantPageSelector:
    def __init__(self, config: ArasConfig) -> None:
        self.config = config
        self.max_pages = 3

    def select(
        self,
        *,
        page_texts: list[PdfPageText],
        image_metadata: list[PdfImageMetadata],
        slug: str,
    ) -> tuple[list[ImportantPage], Path]:
        image_map = {item.page_number: item for item in image_metadata}
        ranked: list[ImportantPage] = []
        for page in page_texts:
            reasons: list[str] = []
            score = 0
            lowered = page.text.lower()
            if any(hint.lower() in lowered for hint in _SECTION_HINTS):
                score += 3
                reasons.append('section_hint')
            numeric_hits = len(_NUMBER_RE.findall(page.text))
            if numeric_hits:
                score += min(4, numeric_hits // 5 + 1)
                reasons.append(f'numeric_density:{numeric_hits}')
            if _TABLE_ROW_RE.search(page.text):
                score += 3
                reasons.append('table_like_pattern')
            chart_hits = sum(1 for hint in _CHART_HINTS if hint.lower() in lowered)
            if chart_hits:
                score += min(3, chart_hits)
                reasons.append(f'chart_hint:{chart_hits}')
            image_count = image_map.get(page.page_number).image_count if page.page_number in image_map else 0
            if image_count:
                score += min(4, image_count)
                reasons.append(f'image_count:{image_count}')
            if page.char_count > 500:
                score += 1
                reasons.append('text_density')
            ranked.append(
                ImportantPage(
                    page_number=page.page_number,
                    score=score,
                    reasons=reasons,
                    preview_path=image_map.get(page.page_number).preview_path if page.page_number in image_map else None,
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.page_number))
        selected = ranked[: self.max_pages]
        path = self.config.paths.processed_dir / f"{slug}-important-pages.json"
        path.write_text(
            json.dumps(
                [
                    {
                        'page_number': item.page_number,
                        'score': item.score,
                        'reasons': item.reasons,
                        'preview_path': item.preview_path,
                    }
                    for item in selected
                ],
                ensure_ascii=False,
                indent=2,
            )
            + '\n'
        )
        return selected, path
