from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .domain import ReportRecord


class SqliteArasStore:
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
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    message_id INTEGER NOT NULL,
                    published_at TEXT,
                    title TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    file_unique_id TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS pipeline_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    lane TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS signal_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def record_download(self, report: ReportRecord) -> bool:
        file_unique_id = str(report.metadata["file_unique_id"])
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO reports (
                    source,
                    channel,
                    message_id,
                    published_at,
                    title,
                    pdf_path,
                    content,
                    metadata_json,
                    file_unique_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.source,
                    report.channel,
                    report.message_id,
                    report.published_at,
                    report.title,
                    str(report.pdf_path),
                    report.content,
                    json.dumps(report.metadata, sort_keys=True),
                    file_unique_id,
                ),
            )
            return cursor.rowcount == 1

    def has_seen_file(self, file_unique_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM reports WHERE file_unique_id = ?",
                (file_unique_id,),
            ).fetchone()
            return row is not None

    def list_reports(self) -> list[ReportRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, source, channel, message_id, published_at, title, pdf_path, content, metadata_json
                FROM reports
                ORDER BY id ASC
                """
            ).fetchall()
        return [self._row_to_report(row) for row in rows]

    def get_latest_report(self, channel: str) -> ReportRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, source, channel, message_id, published_at, title, pdf_path, content, metadata_json
                FROM reports
                WHERE channel = ?
                ORDER BY message_id DESC, id DESC
                LIMIT 1
                """,
                (channel,),
            ).fetchone()
        return None if row is None else self._row_to_report(row)

    def get_report_by_file_unique_id(self, file_unique_id: str) -> ReportRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, source, channel, message_id, published_at, title, pdf_path, content, metadata_json
                FROM reports
                WHERE file_unique_id = ?
                LIMIT 1
                """,
                (file_unique_id,),
            ).fetchone()
        return None if row is None else self._row_to_report(row)

    def get_next_update_offset(self) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM pipeline_state WHERE key = 'telegram_next_update_offset'"
            ).fetchone()
            return None if row is None else int(row["value"])

    def set_next_update_offset(self, update_offset: int) -> None:
        self._set_pipeline_state("telegram_next_update_offset", update_offset)

    def get_last_seen_message_id(self, channel: str) -> int | None:
        key = self._last_seen_key(channel)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM pipeline_state WHERE key = ?",
                (key,),
            ).fetchone()
            return None if row is None else int(row["value"])

    def set_last_seen_message_id(self, channel: str, message_id: int) -> None:
        self._set_pipeline_state(self._last_seen_key(channel), message_id)

    def _set_pipeline_state(self, key: str, value: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pipeline_state (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, str(value)),
            )

    @staticmethod
    def _last_seen_key(channel: str) -> str:
        return f"telethon_last_seen_message_id::{channel}"

    @staticmethod
    def _row_to_report(row: sqlite3.Row) -> ReportRecord:
        return ReportRecord(
            id=row["id"],
            source=row["source"],
            channel=row["channel"],
            message_id=row["message_id"],
            published_at=row["published_at"],
            title=row["title"],
            pdf_path=Path(row["pdf_path"]),
            content=row["content"],
            metadata=json.loads(row["metadata_json"]),
        )
