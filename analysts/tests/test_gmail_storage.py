from pathlib import Path

from analysts.sources.gmail.models import GmailCandidateDocument, GmailMessageRecord
from analysts.sources.gmail.storage import GmailStore


def test_record_message_and_get_message_round_trip(tmp_path: Path) -> None:
    store = GmailStore(tmp_path / "gmail.sqlite3")
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
        body_plain="Structured report body",
        body_html=None,
        raw_payload_json={"id": "msg-1"},
        sync_status="synced",
        query_fingerprint="qhash",
    )

    store.record_message(message)

    assert store.get_message("msg-1") == message


def test_record_candidate_and_list_candidates_for_message_round_trip(tmp_path: Path) -> None:
    store = GmailStore(tmp_path / "gmail.sqlite3")
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
        body_plain="Structured report body",
        body_html=None,
        raw_payload_json={"id": "msg-1"},
        sync_status="synced",
        query_fingerprint="qhash",
    )
    candidate = GmailCandidateDocument(
        candidate_id="cand-1",
        gmail_message_id="msg-1",
        gmail_thread_id="thread-1",
        candidate_kind="email_body",
        source_path="body://msg-1",
        title="Morning wrap",
        mime_type="text/plain",
        dedupe_key="body::msg-1::hash",
        sha256="hash",
        promotion_reason="body_rule:structured",
        raw_path=Path("raw/body.txt"),
        normalized_text_path=Path("processed/body.txt"),
        status="ready",
    )

    store.record_message(message)
    store.record_candidate(candidate)

    assert store.list_candidates_for_message("msg-1") == [candidate]


def test_set_sync_state_and_get_sync_state_round_trip(tmp_path: Path) -> None:
    store = GmailStore(tmp_path / "gmail.sqlite3")

    store.set_sync_state(
        account_name="reports-primary",
        last_history_id="200",
        full_sync_required=False,
    )

    sync_state = store.get_sync_state("reports-primary")

    assert sync_state.account_name == "reports-primary"
    assert sync_state.last_history_id == "200"
    assert sync_state.full_sync_required is False
    assert sync_state.updated_at
