from __future__ import annotations

from html import unescape
from hashlib import sha256
from pathlib import Path
import re
import zipfile

from analysts.config import BodyCandidateRules

from .models import GmailAttachmentRecord, GmailCandidateDocument, GmailMessageRecord

_HTML_TAG_RE = re.compile(r"<[/!a-zA-Z][^>]*>")


class GmailCandidateBuilder:
    def __init__(
        self,
        output_dir: Path,
        *,
        body_rules: BodyCandidateRules,
        zip_allow_extensions: tuple[str, ...],
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.body_rules = body_rules
        self.zip_allow_extensions = tuple(ext.lower() for ext in zip_allow_extensions)

    def build_candidates(
        self,
        *,
        message: GmailMessageRecord,
        attachments: list[GmailAttachmentRecord],
    ) -> list[GmailCandidateDocument]:
        candidates: list[GmailCandidateDocument] = []
        body_candidate = self._build_body_candidate(message)
        if body_candidate is not None:
            candidates.append(body_candidate)
        for attachment in attachments:
            candidates.extend(
                self.extract_attachment_candidates(
                    message_id=message.gmail_message_id,
                    thread_id=message.gmail_thread_id,
                    attachment=attachment,
                )
            )
        return candidates

    def extract_attachment_candidates(
        self,
        *,
        message_id: str,
        thread_id: str | None,
        attachment: GmailAttachmentRecord,
    ) -> list[GmailCandidateDocument]:
        if attachment.is_zip:
            return self._extract_zip_candidates(message_id=message_id, thread_id=thread_id, attachment=attachment)
        extension = Path(attachment.filename).suffix.lower()
        if extension not in self.zip_allow_extensions or not attachment.raw_path.exists():
            return []
        return [self._file_candidate(message_id=message_id, thread_id=thread_id, attachment=attachment, extension=extension)]

    def _build_body_candidate(self, message: GmailMessageRecord) -> GmailCandidateDocument | None:
        text = _body_text(message)
        if len(text) < self.body_rules.min_chars:
            return None
        if self.body_rules.require_structure and "\n\n" not in text:
            return None
        body_hash = sha256(text.encode("utf-8")).hexdigest()
        body_path = self.output_dir / f"{message.gmail_message_id}-body.txt"
        body_path.write_text(text)
        return GmailCandidateDocument(
            candidate_id=f"body::{message.gmail_message_id}",
            gmail_message_id=message.gmail_message_id,
            gmail_thread_id=message.gmail_thread_id,
            candidate_kind="email_body",
            source_path=f"body://{message.gmail_message_id}",
            title=message.subject,
            mime_type="text/plain",
            dedupe_key=f"body::{message.gmail_message_id}::{body_hash}",
            sha256=body_hash,
            promotion_reason="body_rule:structured",
            raw_path=body_path,
            normalized_text_path=body_path,
            status="ready",
        )

    def _file_candidate(
        self,
        *,
        message_id: str,
        thread_id: str | None,
        attachment: GmailAttachmentRecord,
        extension: str,
    ) -> GmailCandidateDocument:
        file_hash = sha256(attachment.raw_path.read_bytes()).hexdigest()
        text_path = (
            attachment.raw_path
            if extension == ".txt"
            else _text_path(
                output_dir=self.output_dir,
                name=f"{message_id}-{attachment.attachment_id}",
                extension=extension,
                text=attachment.raw_path.read_text(errors="ignore"),
            )
        )
        return GmailCandidateDocument(
            candidate_id=f"file::{message_id}::{attachment.attachment_id}",
            gmail_message_id=message_id,
            gmail_thread_id=thread_id,
            candidate_kind=f"file_{extension.lstrip('.')}",
            source_path=f"file://{attachment.attachment_id}/{attachment.filename}",
            title=attachment.filename,
            mime_type=_mime_type_for_extension(extension),
            dedupe_key=f"file::{message_id}::{attachment.attachment_id}::{file_hash}",
            sha256=file_hash,
            promotion_reason="file_allowlist",
            raw_path=attachment.raw_path,
            normalized_text_path=text_path,
            status="ready",
        )

    def _extract_zip_candidates(
        self,
        *,
        message_id: str,
        thread_id: str | None,
        attachment: GmailAttachmentRecord,
    ) -> list[GmailCandidateDocument]:
        candidates: list[GmailCandidateDocument] = []
        extract_dir = self.output_dir / f"{message_id}-{attachment.attachment_id}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(attachment.raw_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                extension = Path(member.filename).suffix.lower()
                if extension not in self.zip_allow_extensions:
                    continue
                entry_bytes = archive.read(member.filename)
                target_path = extract_dir / Path(member.filename).name
                target_path.write_bytes(entry_bytes)
                entry_hash = sha256(entry_bytes).hexdigest()
                candidates.append(
                    GmailCandidateDocument(
                        candidate_id=f"zip::{message_id}::{attachment.attachment_id}::{member.filename}",
                        gmail_message_id=message_id,
                        gmail_thread_id=thread_id,
                        candidate_kind=f"zip_entry_{extension.lstrip('.')}",
                        source_path=f"zip://{attachment.attachment_id}/{member.filename}",
                        title=Path(member.filename).name,
                        mime_type=_mime_type_for_extension(extension),
                        dedupe_key=f"zip-entry::{message_id}::{attachment.attachment_id}::{member.filename}::{entry_hash}",
                        sha256=entry_hash,
                        promotion_reason="zip_allowlist",
                        raw_path=target_path,
                        normalized_text_path=(
                            target_path
                            if extension == ".txt"
                            else _text_path(
                                output_dir=self.output_dir,
                                name=f"{message_id}-{attachment.attachment_id}-{Path(member.filename).stem}",
                                extension=extension,
                                text=entry_bytes.decode("utf-8", errors="ignore"),
                            )
                        ),
                        status="ready",
                    )
                )
        return candidates


def _mime_type_for_extension(extension: str) -> str:
    return {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".html": "text/html",
    }[extension]


def _clean_body(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if not _HTML_TAG_RE.search(text):
        return text
    text = re.sub(r"(?i)<\s*(br|/p|/div|/tr|/li|/h\d)\b[^>]*>", "\n\n", text)
    text = re.sub(r"(?i)<\s*(p|div|tr|li|h\d)\b[^>]*>", "\n\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n\n".join(line for line in lines if line)


def _body_text(message: GmailMessageRecord) -> str:
    plain = _clean_body(message.body_plain or "")
    html = _clean_body(message.body_html or "")
    return html if len(html) > len(plain) else plain


def _text_path(*, output_dir: Path, name: str, extension: str, text: str) -> Path | None:
    if extension == ".txt":
        return None
    if extension != ".html":
        return None
    text = _clean_body(text)
    path = output_dir / f"{name}.txt"
    path.write_text(text)
    return path
