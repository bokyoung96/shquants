from pathlib import Path
import zipfile

import pytest

from analysts.config import BodyCandidateRules, build_config
from analysts.domain import AnalystSummary
from analysts.pipeline import ArasPipeline
from analysts.sources.gmail import dirs
from analysts.sources.gmail.storage import GmailStore
from analysts.sources.gmail.models import GmailMessageRecord
from analysts.sources.gmail.pipeline import GmailSourcePipeline
from analysts.sources.gmail.sync import GmailPollingSync
from analysts.sources.gmail.web_capture import WebSnapshot
from analysts.storage import SqliteArasStore


class FakeGmailApi:
    def list_message_ids(self, *, query: str, page_token: str | None = None, limit: int = 50):
        return {"messages": [{"id": "msg-1", "threadId": "thread-1"}], "next_page_token": None}

    def get_message(self, *, message_id: str) -> dict:
        return {
            "id": "msg-1",
            "threadId": "thread-1",
            "historyId": "200",
            "labelIds": ["Label_Reports"],
            "snippet": "Morning wrap",
            "internalDate": "1713247200000",
            "payload": {
                "headers": [
                    {"name": "From", "value": "broker@example.com"},
                    {"name": "Subject", "value": "Morning wrap"},
                ],
                "mimeType": "text/plain",
                "body": {"data": "U3RydWN0dXJlZCByZXBvcnQgYm9keQ"},
            },
        }

    def get_attachment_data(self, *, message_id: str, attachment_id: str) -> bytes:
        return f"{message_id}:{attachment_id}".encode("utf-8")


class FakeSummarizer:
    @staticmethod
    def lane_plan(packet) -> list[tuple[str, str]]:
        return [("sector", "general"), ("macro", "general")]

    def summarize(self, *, packet, lane: str, topic: str) -> AnalystSummary:
        return AnalystSummary(
            lane=lane,
            topic=topic,
            headline=f"{lane} headline",
            executive_summary=f"{lane} summary",
            key_points=[f"{lane} point"],
            key_numbers=["42%"],
            risks=[f"{lane} risk"],
            confidence="medium",
            cited_pages=[1],
            follow_up_questions=[f"{lane} question"],
        )


class FakeWebCapturer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def capture(self, *, message_id: str, title: str, url: str, index: int) -> WebSnapshot:
        target = dirs.ensure(self.root, message_id=message_id, title=title) / "web"
        target.mkdir(parents=True, exist_ok=True)
        html_path = target / f"page-{index}.html"
        text_path = target / f"page-{index}.txt"
        html_path.write_text(f"<html><body>{url}</body></html>")
        text_path.write_text(f"Captured {url}")
        return WebSnapshot(url=url, html_path=html_path, text_path=text_path, screenshot_path=None)


def test_polling_sync_records_new_messages(tmp_path) -> None:
    store = GmailStore(tmp_path / "gmail.sqlite3")
    sync = GmailPollingSync(
        api=FakeGmailApi(),
        store=store,
        account_name="reports-primary",
        query="label:broker-reports",
        raw_root=tmp_path / "raw" / "gmail",
    )

    result = sync.sync_once(limit=10)

    assert result.fetched == 1
    assert store.get_message("msg-1").subject == "Morning wrap"
    assert store.get_sync_state("reports-primary").last_history_id == "200"
    container = dirs.find(tmp_path / "raw" / "gmail", message_id="msg-1", title="Morning wrap")
    assert (container / "message.json").exists()
    assert (container / "body.txt").read_text() == "Structured report body"
    manifest = (container / "attachments" / "manifest.json").read_text()
    assert '"attachments": []' in manifest


def test_polling_sync_persists_original_attachment_files(tmp_path) -> None:
    class AttachmentApi(FakeGmailApi):
        def get_message(self, *, message_id: str) -> dict:
            payload = super().get_message(message_id=message_id)
            payload["payload"] = {
                "mimeType": "multipart/mixed",
                "headers": [
                    {"name": "From", "value": "broker@example.com"},
                    {"name": "Subject", "value": "Morning wrap"},
                ],
                "parts": [
                    {
                        "mimeType": "application/pdf",
                        "filename": "report.pdf",
                        "body": {"attachmentId": "att-1", "size": 12},
                    },
                    {
                        "mimeType": "application/zip",
                        "filename": "bundle.zip",
                        "body": {"attachmentId": "att-2", "size": 20},
                    },
                ],
            }
            return payload

    store = GmailStore(tmp_path / "gmail.sqlite3")
    sync = GmailPollingSync(
        api=AttachmentApi(),
        store=store,
        account_name="reports-primary",
        query="label:broker-reports",
        raw_root=tmp_path / "raw" / "gmail",
    )

    result = sync.sync_once(limit=10)

    assert result.fetched == 1
    base = dirs.find(tmp_path / "raw" / "gmail", message_id="msg-1", title="Morning wrap")
    container = base / "attachments" / "original"
    assert (container / "report.pdf").read_bytes() == b"msg-1:att-1"
    assert (container / "bundle.zip").read_bytes() == b"msg-1:att-2"
    manifest = (base / "attachments" / "manifest.json").read_text()
    assert '"saved_path":' in manifest


def test_polling_sync_persists_plain_and_html_bodies(tmp_path) -> None:
    class MultipartApi(FakeGmailApi):
        def get_message(self, *, message_id: str) -> dict:
            return {
                "id": "msg-1",
                "threadId": "thread-1",
                "historyId": "200",
                "labelIds": ["Label_Reports"],
                "snippet": "Morning wrap",
                "internalDate": "1713247200000",
                "payload": {
                    "mimeType": "multipart/alternative",
                    "headers": [
                        {"name": "From", "value": "broker@example.com"},
                        {"name": "Subject", "value": "Morning wrap"},
                    ],
                    "parts": [
                        {
                            "mimeType": "text/plain",
                            "body": {"data": "U3RydWN0dXJlZCByZXBvcnQgYm9keQ"},
                        },
                        {
                            "mimeType": "text/html",
                            "body": {"data": "PGh0bWw-PGJvZHk-PHA-UmV2ZW51ZSB1cCAxMCU8L3A-PC9ib2R5PjwvaHRtbD4"},
                        },
                    ],
                },
            }

    store = GmailStore(tmp_path / "gmail.sqlite3")
    sync = GmailPollingSync(
        api=MultipartApi(),
        store=store,
        account_name="reports-primary",
        query="label:broker-reports",
        raw_root=tmp_path / "raw" / "gmail",
    )

    result = sync.sync_once(limit=10)

    assert result.fetched == 1
    message = store.get_message("msg-1")
    assert message.body_plain == "Structured report body"
    assert "Revenue up 10%" in (message.body_html or "")
    container = dirs.find(tmp_path / "raw" / "gmail", message_id="msg-1", title="Morning wrap")
    assert (container / "body.txt").read_text() == "Structured report body"
    assert "Revenue up 10%" in (container / "body.html").read_text()


def test_polling_sync_uses_readable_folder_name(tmp_path) -> None:
    store = GmailStore(tmp_path / "gmail.sqlite3")
    sync = GmailPollingSync(
        api=FakeGmailApi(),
        store=store,
        account_name="reports-primary",
        query="label:broker-reports",
        raw_root=tmp_path / "raw" / "gmail",
    )

    sync.sync_once(limit=10)

    folders = [path.name for path in (tmp_path / "raw" / "gmail").iterdir() if path.is_dir()]
    assert any(name.startswith("msg-1-Morning_wrap") for name in folders)


def test_gmail_source_pipeline_summarizes_latest_body_candidate(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    gmail_store = GmailStore(tmp_path / "gmail.sqlite3")
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-1",
            gmail_thread_id="thread-1",
            history_id="200",
            account_name="reports-primary",
            subject="Morning wrap",
            sender="broker@example.com",
            internal_date="2026-04-16T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="Section A\n\nRevenue up 10%\n\nRisk factors listed below." * 40,
            body_html=None,
            raw_payload_json={"id": "msg-1"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    analysts_pipeline = ArasPipeline(
        client=object(),
        store=SqliteArasStore(config.paths.state_db),
        config=config,
        summarizer=FakeSummarizer(),
    )
    source_pipeline = GmailSourcePipeline(
        config=config,
        api=FakeGmailApi(),
        store=gmail_store,
        analysts_pipeline=analysts_pipeline,
        account_name="reports-primary",
        query="label:broker-reports",
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
        raw_root=config.paths.raw_dir / "gmail",
    )

    execution = source_pipeline.summarize_latest()

    assert len(execution.summaries) == 2
    assert execution.summary.next_offset == "msg-1"
    assert (config.paths.processed_dir / "gmail" / "msg-1-body.txt").exists()
    assert {path.name for path in execution.processed_files} == {
        "report-msg-1-raw-text.txt",
        "report-msg-1-summary-input.json",
        "report-msg-1-summary.json",
        "report-msg-1-summary.md",
    }
    assert {path.parent for path in execution.processed_files} == {config.paths.processed_dir}


def test_gmail_source_pipeline_uses_latest_html_body(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    gmail_store = GmailStore(tmp_path / "gmail.sqlite3")
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-1",
            gmail_thread_id="thread-1",
            history_id="200",
            account_name="reports-primary",
            subject="Structured wrap",
            sender="broker@example.com",
            internal_date="2026-04-16T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="Section A\n\nRevenue up 10%\n\nRisk factors listed below." * 40,
            body_html=None,
            raw_payload_json={"id": "msg-1"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-2",
            gmail_thread_id="thread-2",
            history_id="201",
            account_name="reports-primary",
            subject="HTML-only wrap",
            sender="broker@example.com",
            internal_date="2026-04-17T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="<html><body><p>Revenue up 10%</p><p>Margins stable</p></body></html>" * 40,
            body_html=None,
            raw_payload_json={"id": "msg-2"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    analysts_pipeline = ArasPipeline(
        client=object(),
        store=SqliteArasStore(config.paths.state_db),
        config=config,
        summarizer=FakeSummarizer(),
    )
    source_pipeline = GmailSourcePipeline(
        config=config,
        api=FakeGmailApi(),
        store=gmail_store,
        analysts_pipeline=analysts_pipeline,
        account_name="reports-primary",
        query="label:broker-reports",
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
        raw_root=config.paths.raw_dir / "gmail",
    )

    execution = source_pipeline.summarize_latest()

    assert execution.summary.next_offset == "msg-2"
    assert {path.name for path in execution.processed_files} == {
        "report-msg-2-raw-text.txt",
        "report-msg-2-summary-input.json",
        "report-msg-2-summary.json",
        "report-msg-2-summary.md",
    }


def test_gmail_source_pipeline_skips_latest_empty_message(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    gmail_store = GmailStore(tmp_path / "gmail.sqlite3")
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-1",
            gmail_thread_id="thread-1",
            history_id="200",
            account_name="reports-primary",
            subject="Structured wrap",
            sender="broker@example.com",
            internal_date="2026-04-16T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="Section A\n\nRevenue up 10%\n\nRisk factors listed below." * 40,
            body_html=None,
            raw_payload_json={"id": "msg-1"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-2",
            gmail_thread_id="thread-2",
            history_id="201",
            account_name="reports-primary",
            subject="Empty wrap",
            sender="broker@example.com",
            internal_date="2026-04-17T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="too short",
            body_html=None,
            raw_payload_json={"id": "msg-2"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    analysts_pipeline = ArasPipeline(
        client=object(),
        store=SqliteArasStore(config.paths.state_db),
        config=config,
        summarizer=FakeSummarizer(),
    )
    source_pipeline = GmailSourcePipeline(
        config=config,
        api=FakeGmailApi(),
        store=gmail_store,
        analysts_pipeline=analysts_pipeline,
        account_name="reports-primary",
        query="label:broker-reports",
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
        raw_root=config.paths.raw_dir / "gmail",
    )

    execution = source_pipeline.summarize_latest()

    assert execution.summary.next_offset == "msg-1"
    assert {path.name for path in execution.processed_files} == {
        "report-msg-1-raw-text.txt",
        "report-msg-1-summary-input.json",
        "report-msg-1-summary.json",
        "report-msg-1-summary.md",
    }


def test_gmail_source_pipeline_uses_latest_attachment_candidate(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    gmail_store = GmailStore(tmp_path / "gmail.sqlite3")
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-1",
            gmail_thread_id="thread-1",
            history_id="200",
            account_name="reports-primary",
            subject="Older wrap",
            sender="broker@example.com",
            internal_date="2026-04-16T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="Section A\n\nRevenue up 10%\n\nRisk factors listed below." * 40,
            body_html=None,
            raw_payload_json={"id": "msg-1"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-2",
            gmail_thread_id="thread-2",
            history_id="201",
            account_name="reports-primary",
            subject="Latest attachment wrap",
            sender="broker@example.com",
            internal_date="2026-04-17T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="too short",
            body_html=None,
            raw_payload_json={"id": "msg-2"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    base_dir = dirs.ensure(config.paths.gmail_raw_dir, message_id="msg-2", title="Latest attachment wrap")
    raw_dir = base_dir / "attachments" / "original"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_path = raw_dir / "report.txt"
    report_path.write_text("Revenue up 10%\n\nMargins stable")
    manifest_path = raw_dir.parent / "manifest.json"
    manifest_path.write_text(
        '{"gmail_message_id":"msg-2","attachments":[{"attachment_id":"att-1","filename":"report.txt","mime_type":"text/plain","is_zip":false,"saved_path":"attachments/original/report.txt"}]}\n'
    )
    analysts_pipeline = ArasPipeline(
        client=object(),
        store=SqliteArasStore(config.paths.state_db),
        config=config,
        summarizer=FakeSummarizer(),
    )
    source_pipeline = GmailSourcePipeline(
        config=config,
        api=FakeGmailApi(),
        store=gmail_store,
        analysts_pipeline=analysts_pipeline,
        account_name="reports-primary",
        query="label:broker-reports",
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
        raw_root=config.paths.raw_dir / "gmail",
    )

    execution = source_pipeline.summarize_latest()

    assert execution.summary.next_offset == "msg-2"
    assert {path.name for path in execution.processed_files} == {
        "report-msg-2-raw-text.txt",
        "report-msg-2-summary-input.json",
        "report-msg-2-summary.json",
        "report-msg-2-summary.md",
    }


def test_gmail_source_pipeline_uses_latest_zip_html_candidate(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    gmail_store = GmailStore(tmp_path / "gmail.sqlite3")
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-1",
            gmail_thread_id="thread-1",
            history_id="200",
            account_name="reports-primary",
            subject="Older wrap",
            sender="broker@example.com",
            internal_date="2026-04-16T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="Section A\n\nRevenue up 10%\n\nRisk factors listed below." * 40,
            body_html=None,
            raw_payload_json={"id": "msg-1"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-2",
            gmail_thread_id="thread-2",
            history_id="201",
            account_name="reports-primary",
            subject="Latest zip wrap",
            sender="broker@example.com",
            internal_date="2026-04-17T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="too short",
            body_html=None,
            raw_payload_json={"id": "msg-2"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    raw_dir = config.paths.raw_dir / "gmail" / "msg-2" / "attachments" / "original"
    raw_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = raw_dir / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as archive:
        archive.writestr("inside/report.html", "<html><body><p>Revenue up 10%</p><p>Margins stable</p></body></html>")
    manifest_path = raw_dir.parent / "manifest.json"
    manifest_path.write_text(
        '{"gmail_message_id":"msg-2","attachments":[{"attachment_id":"att-1","filename":"bundle.zip","mime_type":"application/zip","is_zip":true,"saved_path":"attachments/original/bundle.zip"}]}\n'
    )
    analysts_pipeline = ArasPipeline(
        client=object(),
        store=SqliteArasStore(config.paths.state_db),
        config=config,
        summarizer=FakeSummarizer(),
    )
    source_pipeline = GmailSourcePipeline(
        config=config,
        api=FakeGmailApi(),
        store=gmail_store,
        analysts_pipeline=analysts_pipeline,
        account_name="reports-primary",
        query="label:broker-reports",
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
        raw_root=config.paths.raw_dir / "gmail",
    )

    execution = source_pipeline.summarize_latest()

    assert execution.summary.next_offset == "msg-2"
    assert {path.name for path in execution.processed_files} == {
        "report-msg-2-raw-text.txt",
        "report-msg-2-summary-input.json",
        "report-msg-2-summary.json",
        "report-msg-2-summary.md",
    }


def test_gmail_source_pipeline_uses_web_candidate_when_body_is_too_short(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    gmail_store = GmailStore(tmp_path / "gmail.sqlite3")
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-2",
            gmail_thread_id="thread-2",
            history_id="201",
            account_name="reports-primary",
            subject="Latest web wrap",
            sender="broker@example.com",
            internal_date="2026-04-17T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain="Read here: https://example.com/report",
            body_html=None,
            raw_payload_json={"id": "msg-2"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    analysts_pipeline = ArasPipeline(
        client=object(),
        store=SqliteArasStore(config.paths.state_db),
        config=config,
        summarizer=FakeSummarizer(),
    )
    source_pipeline = GmailSourcePipeline(
        config=config,
        api=FakeGmailApi(),
        store=gmail_store,
        analysts_pipeline=analysts_pipeline,
        account_name="reports-primary",
        query="label:broker-reports",
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
        raw_root=config.paths.gmail_raw_dir,
        web_capturer=FakeWebCapturer(config.paths.gmail_raw_dir),
    )

    execution = source_pipeline.summarize_latest()

    assert execution.summary.next_offset == "msg-2"
    assert {path.name for path in execution.processed_files} == {
        "report-msg-2-raw-text.txt",
        "report-msg-2-summary-input.json",
        "report-msg-2-summary.json",
        "report-msg-2-summary.md",
    }
    web_dir = dirs.find(config.paths.gmail_raw_dir, message_id="msg-2", title="Latest web wrap") / "web"
    assert (web_dir / "page-1.html").exists()
    assert (web_dir / "page-1.txt").exists()


def test_gmail_source_pipeline_prefers_web_candidate_for_html_only_link_mail(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    gmail_store = GmailStore(tmp_path / "gmail.sqlite3")
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-2",
            gmail_thread_id="thread-2",
            history_id="201",
            account_name="reports-primary",
            subject="Latest html link wrap",
            sender="broker@example.com",
            internal_date="2026-04-17T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain=None,
            body_html="""
            <html><body>
            <p>weekly digest</p>
            <a href="http://tracking.example/?URL=https%3A%2F%2Fexample.com%2Freport.pdf">report</a>
            </body></html>
            """,
            raw_payload_json={"id": "msg-2"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )
    analysts_pipeline = ArasPipeline(
        client=object(),
        store=SqliteArasStore(config.paths.state_db),
        config=config,
        summarizer=FakeSummarizer(),
    )
    source_pipeline = GmailSourcePipeline(
        config=config,
        api=FakeGmailApi(),
        store=gmail_store,
        analysts_pipeline=analysts_pipeline,
        account_name="reports-primary",
        query="label:broker-reports",
        body_rules=BodyCandidateRules(min_chars=10, require_structure=False),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
        raw_root=config.paths.gmail_raw_dir,
        web_capturer=FakeWebCapturer(config.paths.gmail_raw_dir),
    )

    execution = source_pipeline.summarize_latest()

    assert execution.summary.next_offset == "msg-2"
    web_dir = dirs.find(config.paths.gmail_raw_dir, message_id="msg-2", title="Latest html link wrap") / "web"
    assert (web_dir / "page-1.html").exists()
    assert (web_dir / "page-1.txt").exists()


def test_gmail_source_pipeline_prefers_pdf_link_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = build_config(tmp_path)
    gmail_store = GmailStore(tmp_path / "gmail.sqlite3")
    gmail_store.record_message(
        GmailMessageRecord(
            gmail_message_id="msg-2",
            gmail_thread_id="thread-2",
            history_id="201",
            account_name="reports-primary",
            subject="Latest pdf link wrap",
            sender="broker@example.com",
            internal_date="2026-04-17T06:00:00Z",
            label_ids=("Label_Reports",),
            snippet="Top line",
            body_plain=None,
            body_html="""
            <html><body>
            <a href="http://tracking.example/?URL=https%3A%2F%2Fexample.com%2Freport.pdf">report</a>
            </body></html>
            """,
            raw_payload_json={"id": "msg-2"},
            sync_status="synced",
            query_fingerprint="label:broker-reports",
        )
    )

    def fake_download(*, raw_root: Path, message: GmailMessageRecord, url: str, index: int):
        pdf_dir = dirs.ensure(raw_root, message_id=message.gmail_message_id, title=message.subject) / "web"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"page-{index}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        from analysts.sources.gmail.models import GmailCandidateDocument

        return GmailCandidateDocument(
            candidate_id=f"web-pdf::{message.gmail_message_id}::{index}",
            gmail_message_id=message.gmail_message_id,
            gmail_thread_id=message.gmail_thread_id,
            candidate_kind="web_pdf",
            source_path=url,
            title=message.subject,
            mime_type="application/pdf",
            dedupe_key=f"web-pdf::{message.gmail_message_id}::{url}",
            sha256="fake",
            promotion_reason="web_link_pdf",
            raw_path=pdf_path,
            normalized_text_path=None,
            status="ready",
        )

    monkeypatch.setattr("analysts.sources.gmail.pipeline._download_link_pdf", fake_download)

    analysts_pipeline = ArasPipeline(
        client=object(),
        store=SqliteArasStore(config.paths.state_db),
        config=config,
        summarizer=FakeSummarizer(),
    )
    source_pipeline = GmailSourcePipeline(
        config=config,
        api=FakeGmailApi(),
        store=gmail_store,
        analysts_pipeline=analysts_pipeline,
        account_name="reports-primary",
        query="label:broker-reports",
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
        raw_root=config.paths.gmail_raw_dir,
        web_capturer=FakeWebCapturer(config.paths.gmail_raw_dir),
    )

    execution = source_pipeline.summarize_latest()

    assert execution.summary.next_offset == "msg-2"
    pdf_dir = dirs.find(config.paths.gmail_raw_dir, message_id="msg-2", title="Latest pdf link wrap") / "web"
    assert (pdf_dir / "page-1.pdf").exists()
