from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ...config import ArasConfig
from ...domain import ReportRecord
from ...storage import SqliteArasStore
from .watcher import WatchMessageResult


class TelegramBotClient(Protocol):
    def get_updates(self, *, offset: int | None, limit: int) -> list[dict[str, Any]]: ...

    def get_file(self, file_id: str) -> dict[str, str]: ...

    def download_file(self, file_path: str) -> bytes: ...


class TelethonChannelClient(Protocol):
    def get_latest_message_id(self, *, channel: str) -> int | None: ...

    def iter_channel_messages(
        self,
        *,
        channel: str,
        after_message_id: int | None,
        limit: int,
    ) -> list[dict[str, Any]]: ...

    def recent_messages(self, *, channel: str, limit: int) -> list[dict[str, Any]]: ...

    def download_document(self, message: dict[str, Any]) -> bytes: ...


@dataclass(frozen=True)
class FetchBatch:
    downloaded: list[ReportRecord] = field(default_factory=list)
    skipped_duplicates: list[dict[str, Any]] = field(default_factory=list)
    ignored_updates: list[int] = field(default_factory=list)
    next_offset: int | None = None


@dataclass(frozen=True)
class DownloadableMessage:
    message_id: int
    published_at: str | None
    title: str
    file_unique_id: str
    file_name: str | None
    payload: dict[str, Any]


class TelegramFetcher:
    def __init__(self, *, client: TelegramBotClient | TelethonChannelClient, store: SqliteArasStore, config: ArasConfig) -> None:
        self.client = client
        self.store = store
        self.config = config

    def poll_once(self, *, channel: str) -> FetchBatch:
        if hasattr(self.client, "get_updates"):
            return self._poll_bot_updates_once(channel=channel)
        return self._poll_telethon_once(channel=channel)

    def catch_up(self, *, channel: str, limit: int = 300) -> FetchBatch:
        if not hasattr(self.client, "recent_messages"):
            return FetchBatch()
        messages = self.client.recent_messages(channel=channel, limit=limit)
        downloaded: list[ReportRecord] = []
        skipped_duplicates: list[dict[str, Any]] = []
        ignored_updates: list[int] = []
        safe_last_seen = self.store.get_last_seen_message_id(channel)

        for message in sorted(messages, key=lambda item: item["message_id"]):
            parsed = self._extract_downloadable_message(message, expected_channel=channel)
            if parsed is None:
                ignored_updates.append(message["message_id"])
                safe_last_seen = self._newer(safe_last_seen, message["message_id"])
                continue

            result = self._ingest_downloadable_message(channel=channel, parsed=parsed)
            if result.status in {"duplicate", "existing_unsummarized"}:
                skipped_duplicates.append(message)
            elif result.report is not None:
                downloaded.append(result.report)
            safe_last_seen = self._newer(safe_last_seen, parsed.message_id)

        self._checkpoint_last_seen(channel=channel, message_id=safe_last_seen)
        return FetchBatch(
            downloaded=downloaded,
            skipped_duplicates=skipped_duplicates,
            ignored_updates=ignored_updates,
            next_offset=safe_last_seen,
        )

    def ingest_message(self, *, channel: str, message: dict[str, Any]) -> WatchMessageResult:
        parsed = self._extract_downloadable_message(message, expected_channel=channel)
        message_id = int(message.get("message_id") or 0)
        if parsed is None:
            if message_id:
                self.store.set_last_seen_message_id(channel, message_id)
            return WatchMessageResult(status="ignored")
        result = self._ingest_downloadable_message(channel=channel, parsed=parsed)
        self.store.set_last_seen_message_id(channel, parsed.message_id)
        return result

    def _poll_bot_updates_once(self, *, channel: str) -> FetchBatch:
        offset = self.store.get_next_update_offset()
        updates = self.client.get_updates(offset=offset, limit=self.config.polling_limit)
        downloaded: list[ReportRecord] = []
        skipped_duplicates: list[dict[str, Any]] = []
        ignored_updates: list[int] = []
        safe_next_offset = offset

        for update in sorted(updates, key=lambda item: item["update_id"]):
            parsed = self._extract_pdf_message(update, expected_channel=channel)
            if parsed is None:
                ignored_updates.append(update["update_id"])
                safe_next_offset = update["update_id"] + 1
                continue

            file_unique_id = parsed["document"]["file_unique_id"]
            if self.store.has_seen_file(file_unique_id):
                skipped_duplicates.append(update)
                safe_next_offset = update["update_id"] + 1
                continue

            file_id = parsed["document"]["file_id"]
            file_info = self.client.get_file(file_id)
            file_bytes = self.client.download_file(file_info["file_path"])
            report = self._record_report(
                channel=channel,
                message_id=parsed["message_id"],
                published_at=self._format_timestamp(parsed.get("date")),
                title=parsed.get("caption") or Path(parsed["document"].get("file_name", "report.pdf")).stem,
                file_unique_id=file_unique_id,
                file_name=parsed["document"].get("file_name"),
                payload=file_bytes,
                metadata={
                    "file_unique_id": file_unique_id,
                    "telegram_file_id": file_id,
                    "telegram_file_path": file_info["file_path"],
                    "telegram_update_id": update["update_id"],
                },
            )
            if report is None:
                skipped_duplicates.append(update)
            else:
                downloaded.append(report)
            safe_next_offset = update["update_id"] + 1

        if safe_next_offset is not None:
            self.store.set_next_update_offset(safe_next_offset)

        return FetchBatch(
            downloaded=downloaded,
            skipped_duplicates=skipped_duplicates,
            ignored_updates=ignored_updates,
            next_offset=safe_next_offset,
        )

    def _poll_telethon_once(self, *, channel: str) -> FetchBatch:
        last_seen_message_id = self._resolve_starting_last_seen_message_id(channel=channel)
        if last_seen_message_id is None:
            latest_message_id = self.client.get_latest_message_id(channel=channel)
            self._checkpoint_last_seen(channel=channel, message_id=latest_message_id)
            return FetchBatch(next_offset=latest_message_id)

        messages = self.client.iter_channel_messages(
            channel=channel,
            after_message_id=last_seen_message_id,
            limit=self.config.polling_limit,
        )
        downloaded: list[ReportRecord] = []
        skipped_duplicates: list[dict[str, Any]] = []
        ignored_updates: list[int] = []
        safe_last_seen = last_seen_message_id

        for message in sorted(messages, key=lambda item: item["message_id"]):
            parsed = self._extract_downloadable_message(message, expected_channel=channel)
            if parsed is None:
                ignored_updates.append(message["message_id"])
                safe_last_seen = message["message_id"]
                self._checkpoint_last_seen(channel=channel, message_id=safe_last_seen)
                continue

            result = self._ingest_downloadable_message(channel=channel, parsed=parsed)
            if result.status == "duplicate":
                skipped_duplicates.append(message)
            else:
                downloaded.append(result.report)
            safe_last_seen = message["message_id"]
            self._checkpoint_last_seen(channel=channel, message_id=safe_last_seen)

        return FetchBatch(
            downloaded=downloaded,
            skipped_duplicates=skipped_duplicates,
            ignored_updates=ignored_updates,
            next_offset=safe_last_seen,
        )

    def _resolve_starting_last_seen_message_id(self, *, channel: str) -> int | None:
        last_seen_message_id = self.store.get_last_seen_message_id(channel)
        if last_seen_message_id is not None:
            return last_seen_message_id
        latest_stored_report = self.store.get_latest_report(channel)
        if latest_stored_report is None:
            return None
        self._checkpoint_last_seen(channel=channel, message_id=latest_stored_report.message_id)
        return latest_stored_report.message_id

    def _record_report(
        self,
        *,
        channel: str,
        message_id: int,
        published_at: str | None,
        title: str,
        file_unique_id: str,
        file_name: str | None,
        payload: bytes,
        metadata: dict[str, Any],
    ) -> ReportRecord | None:
        pdf_path = self._write_pdf(
            update_id=message_id,
            file_unique_id=file_unique_id,
            file_name=file_name,
            payload=payload,
        )
        report = ReportRecord(
            id=None,
            source="telegram",
            channel=channel,
            message_id=message_id,
            published_at=published_at,
            title=title,
            pdf_path=pdf_path,
            content="",
            metadata=metadata,
        )
        inserted = self.store.record_download(report)
        return report if inserted else None

    def _ingest_downloadable_message(self, *, channel: str, parsed: DownloadableMessage) -> WatchMessageResult:
        if self.store.has_seen_file(parsed.file_unique_id):
            existing_report = self.store.get_report_by_file_unique_id(parsed.file_unique_id)
            if existing_report is not None and not self._summary_exists(existing_report):
                return WatchMessageResult(status="existing_unsummarized", report=existing_report)
            return WatchMessageResult(status="duplicate")

        payload = self.client.download_document(parsed.payload)
        report = self._record_report(
            channel=channel,
            message_id=parsed.message_id,
            published_at=parsed.published_at,
            title=parsed.title,
            file_unique_id=parsed.file_unique_id,
            file_name=parsed.file_name,
            payload=payload,
            metadata=self._telethon_metadata(parsed),
        )
        if report is None:
            existing_report = self.store.get_report_by_file_unique_id(parsed.file_unique_id)
            if existing_report is not None and not self._summary_exists(existing_report):
                return WatchMessageResult(status="existing_unsummarized", report=existing_report)
            return WatchMessageResult(status="duplicate")
        return WatchMessageResult(status="downloaded", report=report)

    @staticmethod
    def _telethon_metadata(parsed: DownloadableMessage) -> dict[str, Any]:
        return {
            "file_unique_id": parsed.file_unique_id,
            "telegram_message_id": parsed.message_id,
            "source": "telethon",
            "telegram_caption_text": parsed.payload.get("caption") or "",
        }

    def _summary_exists(self, report: ReportRecord) -> bool:
        slug = f"report-{report.id or report.message_id}"
        return (self.config.paths.processed_dir / f"{slug}-summary.json").exists()

    @staticmethod
    def _extract_pdf_message(update: dict[str, Any], *, expected_channel: str) -> dict[str, Any] | None:
        payload = update.get("channel_post") or update.get("message")
        return TelegramFetcher._extract_supported_pdf_payload(payload, expected_channel=expected_channel)

    @staticmethod
    def _extract_downloadable_message(message: dict[str, Any], *, expected_channel: str) -> DownloadableMessage | None:
        supported = TelegramFetcher._extract_supported_pdf_payload(message, expected_channel=expected_channel)
        if supported is None:
            return None
        document = supported["document"]
        title = TelegramFetcher._downloadable_title(supported)
        return DownloadableMessage(
            message_id=supported["message_id"],
            published_at=TelegramFetcher._format_timestamp(supported.get("date") or supported.get("published_at")),
            title=title,
            file_unique_id=str(document["file_unique_id"]),
            file_name=document.get("file_name"),
            payload=supported,
        )

    @staticmethod
    def _extract_supported_pdf_payload(
        payload: dict[str, Any] | None,
        *,
        expected_channel: str,
    ) -> dict[str, Any] | None:
        if not payload:
            return None
        if not TelegramFetcher._is_expected_channel(payload, expected_channel=expected_channel):
            return None
        if not TelegramFetcher._payload_has_pdf_document(payload):
            return None
        return payload

    @staticmethod
    def _downloadable_title(message: dict[str, Any]) -> str:
        document = message["document"]
        return message.get("caption") or Path(document.get("file_name") or "report.pdf").stem

    @staticmethod
    def _is_expected_channel(payload: dict[str, Any], *, expected_channel: str) -> bool:
        chat = payload.get("chat") or {}
        return chat.get("title") == expected_channel

    @staticmethod
    def _payload_has_pdf_document(payload: dict[str, Any]) -> bool:
        document = payload.get("document")
        if not document:
            return False
        file_name = str(document.get("file_name", ""))
        mime_type = str(document.get("mime_type", ""))
        return mime_type == "application/pdf" or file_name.lower().endswith(".pdf")

    def _write_pdf(self, *, update_id: int, file_unique_id: str, file_name: str | None, payload: bytes) -> Path:
        safe_name = (file_name or "report.pdf").replace("/", "-")
        target = self.config.paths.telegram_raw_dir / f"{update_id}-{file_unique_id}-{safe_name}"
        target.write_bytes(payload)
        return target

    def _checkpoint_last_seen(self, *, channel: str, message_id: int | None) -> None:
        if message_id is None:
            return
        self.store.set_last_seen_message_id(channel, message_id)

    @staticmethod
    def _newer(current: int | None, candidate: int) -> int:
        return candidate if current is None or candidate > current else current

    @staticmethod
    def _format_timestamp(timestamp: int | str | None) -> str | None:
        if timestamp is None:
            return None
        if isinstance(timestamp, str):
            return timestamp
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")
