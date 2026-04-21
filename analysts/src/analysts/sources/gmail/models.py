from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GmailMessageRecord:
    gmail_message_id: str
    gmail_thread_id: str
    history_id: str
    account_name: str
    subject: str
    sender: str | None
    internal_date: str | None
    label_ids: tuple[str, ...]
    snippet: str
    body_plain: str | None
    body_html: str | None
    raw_payload_json: dict[str, Any]
    sync_status: str
    query_fingerprint: str


@dataclass(frozen=True)
class GmailCandidateDocument:
    candidate_id: str
    gmail_message_id: str
    gmail_thread_id: str | None
    candidate_kind: str
    source_path: str
    title: str
    mime_type: str
    dedupe_key: str
    sha256: str
    promotion_reason: str | None
    raw_path: Path
    normalized_text_path: Path | None
    status: str


@dataclass(frozen=True)
class GmailAttachmentRecord:
    gmail_message_id: str
    attachment_id: str
    filename: str
    mime_type: str
    size_bytes: int
    sha256: str
    raw_path: Path
    is_zip: bool
    extraction_status: str


@dataclass(frozen=True)
class GmailSyncState:
    account_name: str
    last_history_id: str | None
    full_sync_required: bool
    updated_at: str
