from __future__ import annotations

import re
from pathlib import Path

from .domain import ReportRecord


_RAW_RE = re.compile(r'^live-(?P<message_id>\d+)-(?P<file_unique_id>[^-]+(?:-[^-]+)*)-(?P<title>.+)\.pdf$')


class RawReportCatalog:
    def __init__(self, *, raw_dir: Path, channel: str = 'DOC_POOL') -> None:
        self.raw_dir = raw_dir
        self.channel = channel

    def list_reports(self) -> list[ReportRecord]:
        reports = [self._to_report(path, index) for index, path in enumerate(sorted(self.raw_dir.glob('*.pdf')), start=1)]
        return sorted(reports, key=lambda report: report.message_id)

    def latest_report(self) -> ReportRecord | None:
        reports = self.list_reports()
        return reports[-1] if reports else None

    def recent_reports(self, limit: int) -> list[ReportRecord]:
        reports = self.list_reports()
        return reports[-limit:]

    def _to_report(self, path: Path, fallback_index: int) -> ReportRecord:
        match = _RAW_RE.match(path.name)
        if match:
            message_id = int(match.group('message_id'))
            file_unique_id = match.group('file_unique_id')
            title = match.group('title').replace('_', ' ')
        else:
            message_id = fallback_index
            file_unique_id = f'raw-{fallback_index}'
            title = path.stem.replace('_', ' ')
        return ReportRecord(
            id=fallback_index,
            source='telegram',
            channel=self.channel,
            message_id=message_id,
            published_at=None,
            title=title,
            pdf_path=path,
            content='',
            metadata={
                'file_unique_id': file_unique_id,
                'telegram_caption_text': title,
                'reconstructed_from_raw': True,
            },
        )
