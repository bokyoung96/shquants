from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REDACTED = "<redacted>"


@dataclass(frozen=True)
class ArasPaths:
    base_dir: Path
    data_dir: Path
    raw_dir: Path
    telegram_raw_dir: Path
    gmail_raw_dir: Path
    processed_dir: Path
    wiki_dir: Path
    signals_dir: Path
    state_dir: Path
    state_db: Path
    local_config_path: Path
    telethon_session_path: Path


@dataclass(frozen=True)
class TelethonConfig:
    api_id: int
    api_hash: str
    phone_number: str
    channel: str
    session_name: str
    mode: str = "telethon"
    pdf_only: bool = True

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "api_id": REDACTED,
            "api_hash": REDACTED,
            "phone_number": REDACTED,
            "channel": self.channel,
            "session_name": self.session_name,
            "pdf_only": self.pdf_only,
        }


@dataclass(frozen=True)
class SummaryConfig:
    provider: str = "codex_cli"
    model: str = "gpt-5.4-mini"
    reasoning_effort: str = "low"
    max_input_chars: int = 3200
    max_key_points: int = 4
    cli_command: str = "codex"
    render_page_previews: bool = True
    max_preview_pages: int = 2

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_input_chars": self.max_input_chars,
            "max_key_points": self.max_key_points,
            "cli_command": self.cli_command,
            "render_page_previews": self.render_page_previews,
            "max_preview_pages": self.max_preview_pages,
        }


@dataclass(frozen=True)
class BodyCandidateRules:
    min_chars: int = 800
    require_structure: bool = True


@dataclass(frozen=True)
class GmailConfig:
    account_name: str
    client_secret_path: Path | None
    client_secret_json: dict[str, Any] | None
    token_path: Path
    query: str
    label_filters: tuple[str, ...] = ()
    body_candidate_rules: BodyCandidateRules = BodyCandidateRules()
    zip_allow_extensions: tuple[str, ...] = (".pdf", ".txt", ".html")
    poll_interval_seconds: int = 300

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "account_name": self.account_name,
            "client_secret_path": None if self.client_secret_path is None else str(self.client_secret_path),
            "client_secret_json": "<inline>" if self.client_secret_json is not None else None,
            "token_path": str(self.token_path),
            "query": self.query,
            "label_filters": list(self.label_filters),
            "body_candidate_rules": {
                "min_chars": self.body_candidate_rules.min_chars,
                "require_structure": self.body_candidate_rules.require_structure,
            },
            "zip_allow_extensions": list(self.zip_allow_extensions),
            "poll_interval_seconds": self.poll_interval_seconds,
        }


@dataclass(frozen=True)
class ArasConfig:
    paths: ArasPaths
    polling_limit: int = 100
    telethon: TelethonConfig | None = None
    gmail: GmailConfig | None = None
    summary: SummaryConfig = SummaryConfig()

    def to_display_dict(self) -> dict[str, Any]:
        return {
            "paths": {
                "base_dir": str(self.paths.base_dir),
                "data_dir": str(self.paths.data_dir),
                "raw_dir": str(self.paths.raw_dir),
                "telegram_raw_dir": str(self.paths.telegram_raw_dir),
                "gmail_raw_dir": str(self.paths.gmail_raw_dir),
                "processed_dir": str(self.paths.processed_dir),
                "wiki_dir": str(self.paths.wiki_dir),
                "signals_dir": str(self.paths.signals_dir),
                "state_dir": str(self.paths.state_dir),
                "state_db": str(self.paths.state_db),
                "local_config_path": str(self.paths.local_config_path),
                "telethon_session_path": str(self.paths.telethon_session_path),
            },
            "polling_limit": self.polling_limit,
            "telethon": None if self.telethon is None else self.telethon.to_display_dict(),
            "gmail": None if self.gmail is None else self.gmail.to_display_dict(),
            "summary": self.summary.to_display_dict(),
        }


@dataclass(frozen=True)
class LocalRuntimeConfig:
    telethon: TelethonConfig | None
    gmail: GmailConfig | None
    summary: SummaryConfig


def build_config(base_dir: Path) -> ArasConfig:
    base_dir = Path(base_dir)
    data_dir = base_dir / "data"
    local_config_path = base_dir / "config.local.json"
    runtime = _load_local_runtime_config(local_config_path)
    session_name = runtime.telethon.session_name if runtime.telethon is not None else "telethon"
    paths = ArasPaths(
        base_dir=base_dir,
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        telegram_raw_dir=data_dir / "raw" / "telegram",
        gmail_raw_dir=data_dir / "raw" / "gmail",
        processed_dir=data_dir / "processed",
        wiki_dir=data_dir / "wiki",
        signals_dir=data_dir / "signals",
        state_dir=data_dir / "state",
        state_db=data_dir / "state" / "aras.sqlite3",
        local_config_path=local_config_path,
        telethon_session_path=data_dir / "state" / f"{session_name}.session",
    )
    for directory in (
        paths.data_dir,
        paths.raw_dir,
        paths.telegram_raw_dir,
        paths.gmail_raw_dir,
        paths.processed_dir,
        paths.wiki_dir,
        paths.signals_dir,
        paths.state_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return ArasConfig(paths=paths, telethon=runtime.telethon, gmail=runtime.gmail, summary=runtime.summary)


def require_telethon_config(config: ArasConfig) -> TelethonConfig:
    if config.telethon is None:
        raise RuntimeError(
            "Missing Telethon config. Create analysts/config.local.json with telegram.api_id, "
            "telegram.api_hash, telegram.phone_number, telegram.channel, and telegram.session_name."
        )
    return config.telethon


def _load_local_runtime_config(local_config_path: Path) -> LocalRuntimeConfig:
    if not local_config_path.exists():
        return LocalRuntimeConfig(telethon=None, gmail=None, summary=SummaryConfig())

    payload = json.loads(local_config_path.read_text())
    telegram_payload = payload.get("telegram") or {}
    gmail_payload = payload.get("gmail") or {}
    summary_payload = payload.get("summary") or {}

    telethon = None
    if telegram_payload:
        mode = str(telegram_payload.get("mode", "telethon"))
        if mode == "telethon":
            telethon = TelethonConfig(
                api_id=int(telegram_payload["api_id"]),
                api_hash=str(telegram_payload["api_hash"]),
                phone_number=str(telegram_payload["phone_number"]),
                channel=str(telegram_payload["channel"]),
                session_name=str(telegram_payload.get("session_name", "telethon")),
                mode=mode,
                pdf_only=bool(telegram_payload.get("pdf_only", True)),
            )

    gmail = None
    if gmail_payload:
        body_rules_payload = gmail_payload.get("body_candidate_rules") or {}
        gmail = GmailConfig(
            account_name=str(gmail_payload["account_name"]),
            client_secret_path=(
                Path(str(gmail_payload["client_secret_path"]))
                if gmail_payload.get("client_secret_path")
                else None
            ),
            client_secret_json=gmail_payload.get("client_secret_json"),
            token_path=Path(str(gmail_payload["token_path"])),
            query=str(gmail_payload["query"]),
            label_filters=tuple(str(item) for item in gmail_payload.get("label_filters", [])),
            body_candidate_rules=BodyCandidateRules(
                min_chars=int(body_rules_payload.get("min_chars", 800)),
                require_structure=bool(body_rules_payload.get("require_structure", True)),
            ),
            zip_allow_extensions=tuple(
                str(item) for item in gmail_payload.get("zip_allow_extensions", [".pdf", ".txt", ".html"])
            ),
            poll_interval_seconds=int(gmail_payload.get("poll_interval_seconds", 300)),
        )

    summary = SummaryConfig(
        provider=str(summary_payload.get("provider", "codex_cli")),
        model=str(summary_payload.get("model", "gpt-5.4-mini")),
        reasoning_effort=str(summary_payload.get("reasoning_effort", "low")),
        max_input_chars=int(summary_payload.get("max_input_chars", 3200)),
        max_key_points=int(summary_payload.get("max_key_points", 4)),
        cli_command=str(summary_payload.get("cli_command", "codex")),
        render_page_previews=bool(summary_payload.get("render_page_previews", True)),
        max_preview_pages=int(summary_payload.get("max_preview_pages", 2)),
    )
    return LocalRuntimeConfig(telethon=telethon, gmail=gmail, summary=summary)
