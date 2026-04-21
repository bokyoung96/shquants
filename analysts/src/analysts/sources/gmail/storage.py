from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from .models import GmailCandidateDocument, GmailMessageRecord, GmailSyncState


class GmailStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS gmail_messages (
                    gmail_message_id TEXT PRIMARY KEY,
                    gmail_thread_id TEXT NOT NULL,
                    history_id TEXT NOT NULL,
                    account_name TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    sender TEXT,
                    internal_date TEXT,
                    label_ids_json TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    body_plain TEXT,
                    body_html TEXT,
                    raw_payload_json TEXT NOT NULL,
                    sync_status TEXT NOT NULL,
                    query_fingerprint TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS gmail_candidate_documents (
                    candidate_id TEXT PRIMARY KEY,
                    gmail_message_id TEXT NOT NULL,
                    gmail_thread_id TEXT,
                    candidate_kind TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    sha256 TEXT NOT NULL,
                    promotion_reason TEXT,
                    raw_path TEXT NOT NULL,
                    normalized_text_path TEXT,
                    status TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS gmail_sync_state (
                    account_name TEXT PRIMARY KEY,
                    last_history_id TEXT,
                    full_sync_required INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def record_message(self, message: GmailMessageRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO gmail_messages (
                    gmail_message_id,
                    gmail_thread_id,
                    history_id,
                    account_name,
                    subject,
                    sender,
                    internal_date,
                    label_ids_json,
                    snippet,
                    body_plain,
                    body_html,
                    raw_payload_json,
                    sync_status,
                    query_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(gmail_message_id) DO UPDATE SET
                    gmail_thread_id = excluded.gmail_thread_id,
                    history_id = excluded.history_id,
                    account_name = excluded.account_name,
                    subject = excluded.subject,
                    sender = excluded.sender,
                    internal_date = excluded.internal_date,
                    label_ids_json = excluded.label_ids_json,
                    snippet = excluded.snippet,
                    body_plain = excluded.body_plain,
                    body_html = excluded.body_html,
                    raw_payload_json = excluded.raw_payload_json,
                    sync_status = excluded.sync_status,
                    query_fingerprint = excluded.query_fingerprint
                """,
                (
                    message.gmail_message_id,
                    message.gmail_thread_id,
                    message.history_id,
                    message.account_name,
                    message.subject,
                    message.sender,
                    message.internal_date,
                    json.dumps(list(message.label_ids)),
                    message.snippet,
                    message.body_plain,
                    message.body_html,
                    json.dumps(message.raw_payload_json, sort_keys=True),
                    message.sync_status,
                    message.query_fingerprint,
                ),
            )

    def get_message(self, gmail_message_id: str) -> GmailMessageRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM gmail_messages WHERE gmail_message_id = ?",
                (gmail_message_id,),
            ).fetchone()
        return None if row is None else self._row_to_message(row)

    def list_messages(self, *, account_name: str | None = None) -> list[GmailMessageRecord]:
        query = "SELECT * FROM gmail_messages"
        params: tuple[str, ...] = ()
        if account_name is not None:
            query += " WHERE account_name = ?"
            params = (account_name,)
        query += " ORDER BY internal_date DESC, gmail_message_id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_message(row) for row in rows]

    def record_candidate(self, candidate: GmailCandidateDocument) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO gmail_candidate_documents (
                    candidate_id,
                    gmail_message_id,
                    gmail_thread_id,
                    candidate_kind,
                    source_path,
                    title,
                    mime_type,
                    dedupe_key,
                    sha256,
                    promotion_reason,
                    raw_path,
                    normalized_text_path,
                    status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    gmail_message_id = excluded.gmail_message_id,
                    gmail_thread_id = excluded.gmail_thread_id,
                    candidate_kind = excluded.candidate_kind,
                    source_path = excluded.source_path,
                    title = excluded.title,
                    mime_type = excluded.mime_type,
                    dedupe_key = excluded.dedupe_key,
                    sha256 = excluded.sha256,
                    promotion_reason = excluded.promotion_reason,
                    raw_path = excluded.raw_path,
                    normalized_text_path = excluded.normalized_text_path,
                    status = excluded.status
                """,
                (
                    candidate.candidate_id,
                    candidate.gmail_message_id,
                    candidate.gmail_thread_id,
                    candidate.candidate_kind,
                    candidate.source_path,
                    candidate.title,
                    candidate.mime_type,
                    candidate.dedupe_key,
                    candidate.sha256,
                    candidate.promotion_reason,
                    str(candidate.raw_path),
                    None if candidate.normalized_text_path is None else str(candidate.normalized_text_path),
                    candidate.status,
                ),
            )

    def list_candidates_for_message(self, gmail_message_id: str) -> list[GmailCandidateDocument]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM gmail_candidate_documents
                WHERE gmail_message_id = ?
                ORDER BY candidate_id
                """,
                (gmail_message_id,),
            ).fetchall()
        return [self._row_to_candidate(row) for row in rows]

    def set_sync_state(self, *, account_name: str, last_history_id: str | None, full_sync_required: bool) -> None:
        updated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO gmail_sync_state (account_name, last_history_id, full_sync_required, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(account_name) DO UPDATE SET
                    last_history_id = excluded.last_history_id,
                    full_sync_required = excluded.full_sync_required,
                    updated_at = excluded.updated_at
                """,
                (account_name, last_history_id, int(full_sync_required), updated_at),
            )

    def get_sync_state(self, account_name: str) -> GmailSyncState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM gmail_sync_state WHERE account_name = ?",
                (account_name,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Missing Gmail sync state for account {account_name}")
        return GmailSyncState(
            account_name=row["account_name"],
            last_history_id=row["last_history_id"],
            full_sync_required=bool(row["full_sync_required"]),
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> GmailMessageRecord:
        return GmailMessageRecord(
            gmail_message_id=row["gmail_message_id"],
            gmail_thread_id=row["gmail_thread_id"],
            history_id=row["history_id"],
            account_name=row["account_name"],
            subject=row["subject"],
            sender=row["sender"],
            internal_date=row["internal_date"],
            label_ids=tuple(json.loads(row["label_ids_json"])),
            snippet=row["snippet"],
            body_plain=row["body_plain"],
            body_html=row["body_html"],
            raw_payload_json=json.loads(row["raw_payload_json"]),
            sync_status=row["sync_status"],
            query_fingerprint=row["query_fingerprint"],
        )

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> GmailCandidateDocument:
        normalized_text_path = row["normalized_text_path"]
        return GmailCandidateDocument(
            candidate_id=row["candidate_id"],
            gmail_message_id=row["gmail_message_id"],
            gmail_thread_id=row["gmail_thread_id"],
            candidate_kind=row["candidate_kind"],
            source_path=row["source_path"],
            title=row["title"],
            mime_type=row["mime_type"],
            dedupe_key=row["dedupe_key"],
            sha256=row["sha256"],
            promotion_reason=row["promotion_reason"],
            raw_path=Path(row["raw_path"]),
            normalized_text_path=None if normalized_text_path is None else Path(normalized_text_path),
            status=row["status"],
        )
