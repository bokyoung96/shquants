from pathlib import Path
import zipfile

from analysts.config import BodyCandidateRules
from analysts.sources.gmail.models import GmailAttachmentRecord, GmailMessageRecord
from analysts.sources.gmail.normalize import GmailCandidateBuilder


def test_body_candidate_is_created_only_when_rules_match(tmp_path: Path) -> None:
    message = GmailMessageRecord(
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
        raw_payload_json={},
        sync_status="synced",
        query_fingerprint="qhash",
    )
    builder = GmailCandidateBuilder(
        tmp_path,
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
    )

    candidates = builder.build_candidates(message=message, attachments=[])

    assert [candidate.candidate_kind for candidate in candidates] == ["email_body"]


def test_html_body_is_cleaned_into_a_body_candidate(tmp_path: Path) -> None:
    message = GmailMessageRecord(
        gmail_message_id="msg-1",
        gmail_thread_id="thread-1",
        history_id="200",
        account_name="reports-primary",
        subject="Morning wrap",
        sender="broker@example.com",
        internal_date="2026-04-16T06:00:00Z",
        label_ids=("Label_Reports",),
        snippet="Top line",
        body_plain="<html><body><p>Revenue up 10%</p><p>Margins stable</p></body></html>" * 40,
        body_html=None,
        raw_payload_json={},
        sync_status="synced",
        query_fingerprint="qhash",
    )
    builder = GmailCandidateBuilder(
        tmp_path,
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
    )

    candidates = builder.build_candidates(message=message, attachments=[])

    assert [candidate.candidate_kind for candidate in candidates] == ["email_body"]


def test_body_html_can_win_over_short_body_plain(tmp_path: Path) -> None:
    message = GmailMessageRecord(
        gmail_message_id="msg-1",
        gmail_thread_id="thread-1",
        history_id="200",
        account_name="reports-primary",
        subject="Morning wrap",
        sender="broker@example.com",
        internal_date="2026-04-16T06:00:00Z",
        label_ids=("Label_Reports",),
        snippet="Top line",
        body_plain="short",
        body_html="<html><body><p>Revenue up 10%</p><p>Margins stable</p></body></html>" * 40,
        raw_payload_json={},
        sync_status="synced",
        query_fingerprint="qhash",
    )
    builder = GmailCandidateBuilder(
        tmp_path,
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
    )

    candidates = builder.build_candidates(message=message, attachments=[])

    assert [candidate.candidate_kind for candidate in candidates] == ["email_body"]
    text = candidates[0].raw_path.read_text()
    assert text.startswith("Revenue up 10%")
    assert "Margins stable" in text
    assert "<html>" not in text


def test_plain_text_with_angle_brackets_stays_plain_text(tmp_path: Path) -> None:
    message = GmailMessageRecord(
        gmail_message_id="msg-1",
        gmail_thread_id="thread-1",
        history_id="200",
        account_name="reports-primary",
        subject="Morning wrap",
        sender="broker@example.com",
        internal_date="2026-04-16T06:00:00Z",
        label_ids=("Label_Reports",),
        snippet="Top line",
        body_plain=("EV/EBITDA < 10x\n\nRevenue > costs\n\nFCF < capex" * 40),
        body_html=None,
        raw_payload_json={},
        sync_status="synced",
        query_fingerprint="qhash",
    )
    builder = GmailCandidateBuilder(
        tmp_path,
        body_rules=BodyCandidateRules(min_chars=200, require_structure=True),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
    )

    candidates = builder.build_candidates(message=message, attachments=[])

    assert [candidate.candidate_kind for candidate in candidates] == ["email_body"]
    assert candidates[0].raw_path.read_text() == ("EV/EBITDA < 10x\n\nRevenue > costs\n\nFCF < capex" * 40)


def test_direct_attachment_emits_allowlisted_candidate(tmp_path: Path) -> None:
    file_path = tmp_path / "report.txt"
    file_path.write_text("Revenue up 10%\n\nMargins stable")
    attachment = GmailAttachmentRecord(
        gmail_message_id="msg-1",
        attachment_id="att-1",
        filename="report.txt",
        mime_type="text/plain",
        size_bytes=file_path.stat().st_size,
        sha256="",
        raw_path=file_path,
        is_zip=False,
        extraction_status="stored",
    )

    builder = GmailCandidateBuilder(
        tmp_path,
        body_rules=BodyCandidateRules(),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
    )
    candidates = builder.extract_attachment_candidates(message_id="msg-1", thread_id="thread-1", attachment=attachment)

    assert len(candidates) == 1
    assert candidates[0].candidate_kind == "file_txt"
    assert candidates[0].raw_path == file_path


def test_html_attachment_gets_a_clean_text_copy(tmp_path: Path) -> None:
    file_path = tmp_path / "report.html"
    file_path.write_text("<html><body><p>Revenue up 10%</p><p>Margins stable</p></body></html>")
    attachment = GmailAttachmentRecord(
        gmail_message_id="msg-1",
        attachment_id="att-1",
        filename="report.html",
        mime_type="text/html",
        size_bytes=file_path.stat().st_size,
        sha256="",
        raw_path=file_path,
        is_zip=False,
        extraction_status="stored",
    )

    builder = GmailCandidateBuilder(
        tmp_path,
        body_rules=BodyCandidateRules(),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
    )
    candidates = builder.extract_attachment_candidates(message_id="msg-1", thread_id="thread-1", attachment=attachment)

    assert len(candidates) == 1
    assert candidates[0].candidate_kind == "file_html"
    assert candidates[0].normalized_text_path is not None
    assert candidates[0].normalized_text_path != file_path
    assert candidates[0].normalized_text_path.read_text() == "Revenue up 10%\n\nMargins stable"


def test_zip_only_emits_allowlisted_entries(tmp_path: Path) -> None:
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("inside/report.txt", "text report")
        archive.writestr("inside/report.pdf", b"%PDF-1.4 fake")
        archive.writestr("inside/model.xlsx", b"xlsx bytes")

    attachment = GmailAttachmentRecord(
        gmail_message_id="msg-1",
        attachment_id="att-1",
        filename="bundle.zip",
        mime_type="application/zip",
        size_bytes=zip_path.stat().st_size,
        sha256="ziphash",
        raw_path=zip_path,
        is_zip=True,
        extraction_status="stored",
    )

    builder = GmailCandidateBuilder(
        tmp_path,
        body_rules=BodyCandidateRules(),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
    )
    candidates = builder.extract_attachment_candidates(message_id="msg-1", thread_id="thread-1", attachment=attachment)

    assert {candidate.candidate_kind for candidate in candidates} == {"zip_entry_txt", "zip_entry_pdf"}


def test_zip_html_entry_gets_a_clean_text_copy(tmp_path: Path) -> None:
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("inside/report.html", "<html><body><p>Revenue up 10%</p><p>Margins stable</p></body></html>")

    attachment = GmailAttachmentRecord(
        gmail_message_id="msg-1",
        attachment_id="att-1",
        filename="bundle.zip",
        mime_type="application/zip",
        size_bytes=zip_path.stat().st_size,
        sha256="ziphash",
        raw_path=zip_path,
        is_zip=True,
        extraction_status="stored",
    )

    builder = GmailCandidateBuilder(
        tmp_path,
        body_rules=BodyCandidateRules(),
        zip_allow_extensions=(".pdf", ".txt", ".html"),
    )
    candidates = builder.extract_attachment_candidates(message_id="msg-1", thread_id="thread-1", attachment=attachment)

    assert len(candidates) == 1
    assert candidates[0].candidate_kind == "zip_entry_html"
    assert candidates[0].normalized_text_path is not None
    assert candidates[0].normalized_text_path != candidates[0].raw_path
    assert candidates[0].normalized_text_path.read_text() == "Revenue up 10%\n\nMargins stable"
