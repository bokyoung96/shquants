from __future__ import annotations

from base64 import urlsafe_b64decode
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from . import dirs
from .models import GmailAttachmentRecord, GmailMessageRecord
from .storage import GmailStore


@dataclass(frozen=True)
class GmailSyncResult:
    fetched: int
    skipped_existing: int
    last_history_id: str | None


class GmailPollingSync:
    def __init__(
        self,
        *,
        api: Any,
        store: GmailStore,
        account_name: str,
        query: str,
        raw_root: Path,
    ) -> None:
        self.api = api
        self.store = store
        self.account_name = account_name
        self.query = query
        self.raw_root = Path(raw_root)

    def sync_once(self, *, limit: int) -> GmailSyncResult:
        listed = self.api.list_message_ids(query=self.query, limit=limit)
        fetched = 0
        skipped_existing = 0
        last_history_id: str | None = None

        for item in listed.get("messages", []):
            message_id = item["id"]
            if self.store.get_message(message_id) is not None:
                skipped_existing += 1
                continue
            payload = self.api.get_message(message_id=message_id)
            record = build_message_record(account_name=self.account_name, query=self.query, payload=payload)
            self._write_raw_container(record=record)
            self.store.record_message(record)
            fetched += 1
            last_history_id = payload.get("historyId") or last_history_id

        if fetched or last_history_id is not None:
            self.store.set_sync_state(
                account_name=self.account_name,
                last_history_id=last_history_id,
                full_sync_required=False,
            )
        return GmailSyncResult(
            fetched=fetched,
            skipped_existing=skipped_existing,
            last_history_id=last_history_id,
        )

    def _write_raw_container(self, *, record: GmailMessageRecord) -> None:
        container = dirs.ensure(self.raw_root, message_id=record.gmail_message_id, title=record.subject)
        attachments_dir = container / "attachments"
        originals_dir = attachments_dir / "original"
        attachments_dir.mkdir(parents=True, exist_ok=True)
        originals_dir.mkdir(parents=True, exist_ok=True)
        (container / "message.json").write_text(json.dumps(record.raw_payload_json, ensure_ascii=False, indent=2) + "\n")
        if record.body_plain:
            (container / "body.txt").write_text(record.body_plain)
        if record.body_html:
            (container / "body.html").write_text(record.body_html)
        attachments = _extract_attachment_records(payload=record.raw_payload_json, message_id=record.gmail_message_id)
        attachment_entries = []
        for item in attachments:
            saved_path = None
            if item.attachment_id and item.filename:
                payload = self.api.get_attachment_data(message_id=record.gmail_message_id, attachment_id=item.attachment_id)
                target = originals_dir / Path(item.filename).name
                target.write_bytes(payload)
                saved_path = str(target.relative_to(container))
            attachment_entries.append(
                {
                    "attachment_id": item.attachment_id,
                    "filename": item.filename,
                    "mime_type": item.mime_type,
                    "is_zip": item.is_zip,
                    "saved_path": saved_path,
                }
            )
        manifest_payload = {
            "gmail_message_id": record.gmail_message_id,
            "attachments": attachment_entries,
        }
        (attachments_dir / "manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n")


def build_message_record(*, account_name: str, query: str, payload: dict[str, Any]) -> GmailMessageRecord:
    headers = {
        header.get("name", "").lower(): header.get("value")
        for header in (payload.get("payload", {}).get("headers") or [])
    }
    body_plain, body_html = _extract_message_bodies(payload.get("payload", {}))
    return GmailMessageRecord(
        gmail_message_id=str(payload["id"]),
        gmail_thread_id=str(payload["threadId"]),
        history_id=str(payload.get("historyId") or ""),
        account_name=account_name,
        subject=str(headers.get("subject") or ""),
        sender=headers.get("from"),
        internal_date=str(payload.get("internalDate")) if payload.get("internalDate") is not None else None,
        label_ids=tuple(str(item) for item in payload.get("labelIds", [])),
        snippet=str(payload.get("snippet") or ""),
        body_plain=body_plain,
        body_html=body_html,
        raw_payload_json=payload,
        sync_status="synced",
        query_fingerprint=query,
    )


def _extract_attachment_records(*, payload: dict[str, Any], message_id: str) -> list[GmailAttachmentRecord]:
    attachments: list[GmailAttachmentRecord] = []

    def walk(part: dict[str, Any]) -> None:
        filename = str(part.get("filename") or "")
        body = part.get("body") or {}
        attachment_id = body.get("attachmentId")
        mime_type = str(part.get("mimeType") or "")
        if filename or attachment_id:
            attachments.append(
                GmailAttachmentRecord(
                    gmail_message_id=message_id,
                    attachment_id=str(attachment_id or filename or "inline"),
                    filename=filename,
                    mime_type=mime_type,
                    size_bytes=int(body.get("size") or 0),
                    sha256="",
                    raw_path=Path(filename or str(attachment_id or "attachment.bin")),
                    is_zip=filename.lower().endswith(".zip") or mime_type == "application/zip",
                    extraction_status="listed",
                )
            )
        for child in part.get("parts") or []:
            walk(child)

    walk(payload.get("payload", {}))
    return attachments


def _extract_body_text(message_part: dict[str, Any]) -> str | None:
    body = message_part.get("body") or {}
    data = body.get("data")
    if data:
        padded = data + "=" * (-len(data) % 4)
        return urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8")
    for child in message_part.get("parts", []) or []:
        child_text = _extract_body_text(child)
        if child_text:
            return child_text
    return None


def _extract_message_bodies(message_part: dict[str, Any]) -> tuple[str | None, str | None]:
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part: dict[str, Any]) -> None:
        mime_type = str(part.get("mimeType") or "")
        filename = str(part.get("filename") or "")
        body = part.get("body") or {}
        if not filename and not body.get("attachmentId"):
            text = _extract_body_text(part)
            if text:
                if mime_type == "text/plain":
                    plain_parts.append(text)
                elif mime_type == "text/html":
                    html_parts.append(text)
        for child in part.get("parts") or []:
            walk(child)

    walk(message_part)
    plain = "\n\n".join(part.strip() for part in plain_parts if part.strip()) or None
    html = "\n\n".join(part.strip() for part in html_parts if part.strip()) or None
    if plain is None and html is None:
        plain = _extract_body_text(message_part)
    return plain, html
