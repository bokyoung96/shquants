from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlopen

from analysts.config import ArasConfig, BodyCandidateRules
from analysts.pipeline import ArasPipeline

from . import dirs
from .models import GmailAttachmentRecord, GmailCandidateDocument, GmailMessageRecord
from .normalize import GmailCandidateBuilder
from .storage import GmailStore
from .sync import GmailPollingSync
from .web_capture import PlaywrightWebCapturer, WebSnapshot

_URL_RE = re.compile(r"https?://[^\s<>\"]+")


@dataclass(frozen=True)
class GmailSourcePipeline:
    config: ArasConfig
    api: object
    store: GmailStore
    analysts_pipeline: ArasPipeline
    account_name: str
    query: str
    body_rules: BodyCandidateRules
    zip_allow_extensions: tuple[str, ...]
    raw_root: Path
    web_capturer: PlaywrightWebCapturer | None = None

    def sync_once(self, *, limit: int):
        return GmailPollingSync(
            api=self.api,
            store=self.store,
            account_name=self.account_name,
            query=self.query,
            raw_root=self.raw_root,
        ).sync_once(limit=limit)

    def summarize_latest(self):
        candidate = self._find_candidate()
        return self.analysts_pipeline.summarize_canonical(_candidate_to_canonical(candidate, account_name=self.account_name))

    def _find_candidate(self) -> GmailCandidateDocument:
        messages = self.store.list_messages(account_name=self.account_name)
        if not messages:
            raise RuntimeError(f"No Gmail messages found for account {self.account_name}")
        builder = GmailCandidateBuilder(
            self.config.paths.processed_dir / "gmail",
            body_rules=self.body_rules,
            zip_allow_extensions=self.zip_allow_extensions,
        )
        for message in messages:
            attachments = self._read_attachments(message)
            candidates: list[GmailCandidateDocument] = []
            if _prefer_web(message, attachments):
                candidates = self._web_candidates(message)
            if not candidates:
                candidates = builder.build_candidates(message=message, attachments=attachments)
            if not candidates:
                candidates = self._web_candidates(message)
            if candidates:
                return candidates[0]
        raise RuntimeError(f"No Gmail candidates available for account {self.account_name}")

    def _read_attachments(self, message: GmailMessageRecord) -> list[GmailAttachmentRecord]:
        container = dirs.find(self.raw_root, message_id=message.gmail_message_id, title=message.subject)
        manifest_path = container / "attachments" / "manifest.json"
        if not manifest_path.exists():
            return []
        payload = json.loads(manifest_path.read_text())
        attachments: list[GmailAttachmentRecord] = []
        for item in payload.get("attachments", []):
            saved_path = item.get("saved_path")
            if not saved_path:
                continue
            file_path = manifest_path.parent.parent / str(saved_path)
            if not file_path.exists():
                continue
            attachments.append(
                GmailAttachmentRecord(
                    gmail_message_id=message.gmail_message_id,
                    attachment_id=str(item.get("attachment_id") or file_path.name),
                    filename=str(item.get("filename") or file_path.name),
                    mime_type=str(item.get("mime_type") or ""),
                    size_bytes=file_path.stat().st_size,
                    sha256="",
                    raw_path=file_path,
                    is_zip=bool(item.get("is_zip")),
                    extraction_status="stored",
                )
            )
        return attachments

    def _web_candidates(self, message: GmailMessageRecord) -> list[GmailCandidateDocument]:
        capturer = self.web_capturer
        if capturer is None:
            return []
        urls = _candidate_urls(message)
        candidates: list[GmailCandidateDocument] = []
        for index, url in enumerate(urls, start=1):
            if _is_pdf_url(url):
                try:
                    candidates.append(_download_link_pdf(raw_root=self.raw_root, message=message, url=url, index=index))
                    continue
                except Exception:
                    pass
            try:
                snapshot = capturer.capture(message_id=message.gmail_message_id, title=message.subject, url=url, index=index)
            except Exception:
                continue
            candidates.append(_web_candidate(message=message, snapshot=snapshot, index=index))
        return candidates


def _candidate_to_canonical(candidate: GmailCandidateDocument, *, account_name: str):
    from analysts.domain import CanonicalDocument

    return CanonicalDocument(
        source="gmail",
        source_message_id=candidate.gmail_message_id,
        source_thread_id=candidate.gmail_thread_id,
        source_feed=account_name,
        document_kind=candidate.candidate_kind,
        title=candidate.title,
        published_at=None,
        sender_or_origin=None,
        mime_type=candidate.mime_type,
        dedupe_key=candidate.dedupe_key,
        raw_path=candidate.raw_path,
        normalized_text_path=candidate.normalized_text_path,
        metadata={},
    )

def _candidate_urls(message: GmailMessageRecord) -> list[str]:
    plain_urls = _urls(message.body_plain or "")
    if plain_urls:
        return _rank_urls(plain_urls)
    return _rank_urls(_urls(message.body_html or ""))


def _prefer_web(message: GmailMessageRecord, attachments: list[GmailAttachmentRecord]) -> bool:
    return bool(not attachments and not (message.body_plain or "").strip() and _candidate_urls(message))


def _web_candidate(*, message: GmailMessageRecord, snapshot: WebSnapshot, index: int) -> GmailCandidateDocument:
    return GmailCandidateDocument(
        candidate_id=f"web::{message.gmail_message_id}::{index}",
        gmail_message_id=message.gmail_message_id,
        gmail_thread_id=message.gmail_thread_id,
        candidate_kind="web_page",
        source_path=snapshot.url,
        title=message.subject,
        mime_type="text/html",
        dedupe_key=f"web::{message.gmail_message_id}::{snapshot.url}",
        sha256=f"web::{message.gmail_message_id}::{index}",
        promotion_reason="web_link",
        raw_path=snapshot.html_path,
        normalized_text_path=snapshot.text_path,
        status="ready",
    )


def _urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.findall(unescape(text)):
        url = _normalize_url(match.rstrip(").,;"))
        if not url or url in seen or _skip_url(url):
            continue
        urls.append(url)
        seen.add(url)
    return urls


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key in ("URL", "url"):
        if key in query and query[key]:
            return unquote(query[key][0])
    return url


def _skip_url(url: str) -> bool:
    lower = url.lower()
    return lower.endswith((".gif", ".jpg", ".jpeg", ".png", ".svg", ".css"))


def _rank_urls(urls: list[str]) -> list[str]:
    def key(url: str) -> tuple[int, str]:
        lower = url.lower()
        if lower.endswith(".pdf"):
            return (0, url)
        if "viewer" in lower:
            return (1, url)
        return (2, url)

    return sorted(urls, key=key)


def _is_pdf_url(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def _download_link_pdf(*, raw_root: Path, message: GmailMessageRecord, url: str, index: int) -> GmailCandidateDocument:
    container = dirs.ensure(raw_root, message_id=message.gmail_message_id, title=message.subject) / "web"
    container.mkdir(parents=True, exist_ok=True)
    pdf_path = container / f"page-{index}.pdf"
    with urlopen(url, timeout=30) as response:  # noqa: S310
        pdf_path.write_bytes(response.read())
    return GmailCandidateDocument(
        candidate_id=f"web-pdf::{message.gmail_message_id}::{index}",
        gmail_message_id=message.gmail_message_id,
        gmail_thread_id=message.gmail_thread_id,
        candidate_kind="web_pdf",
        source_path=url,
        title=message.subject,
        mime_type="application/pdf",
        dedupe_key=f"web-pdf::{message.gmail_message_id}::{url}",
        sha256=f"web-pdf::{message.gmail_message_id}::{index}",
        promotion_reason="web_link_pdf",
        raw_path=pdf_path,
        normalized_text_path=None,
        status="ready",
    )
