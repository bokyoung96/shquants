from __future__ import annotations

import asyncio
from contextlib import contextmanager
import inspect
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from getpass import getpass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, TypeVar

from ...config import ArasConfig, require_telethon_config

T = TypeVar("T")


@dataclass(frozen=True)
class TelethonMessageAdapter:
    message_id: int
    date: int | str | None
    chat: dict[str, Any]
    caption: str | None
    document: dict[str, Any]
    _message: Any

    def to_fetcher_payload(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "date": self.date,
            "chat": self.chat,
            "caption": self.caption,
            "document": self.document,
            "_message": self._message,
        }


class TelethonChannelClient:
    def __init__(self, *, base_dir: Path, config: ArasConfig) -> None:
        self.base_dir = Path(base_dir)
        self.config = config
        self.settings = require_telethon_config(config)

    def ensure_authorized(self) -> None:
        with self._build_client() as client:
            if client.is_user_authorized():
                return
            client.send_code_request(self.settings.phone_number)
            code = input("Enter Telegram login code: ").strip()
            try:
                client.sign_in(phone=self.settings.phone_number, code=code)
            except self._session_password_error():
                password = getpass("Enter Telegram 2FA password: ")
                client.sign_in(password=password)

    def get_latest_message_id(self, *, channel: str) -> int | None:
        def read_latest(client: Any, entity: Any) -> int | None:
            messages = client.get_messages(entity, limit=1)
            return None if not messages else int(messages[0].id)

        return self._read_channel(channel=channel, reader=read_latest)

    def iter_channel_messages(
        self,
        *,
        channel: str,
        after_message_id: int | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        def read_messages(client: Any, entity: Any) -> list[dict[str, Any]]:
            messages = list(
                client.iter_messages(
                    entity,
                    limit=limit,
                    min_id=after_message_id or 0,
                    reverse=True,
                )
            )
            return [self._adapt_message(channel=channel, message=message).to_fetcher_payload() for message in messages]

        return self._read_channel(channel=channel, reader=read_messages)

    def recent_messages(self, *, channel: str, limit: int) -> list[dict[str, Any]]:
        def read_messages(client: Any, entity: Any) -> list[dict[str, Any]]:
            messages = list(client.iter_messages(entity, limit=limit))
            return [self._adapt_message(channel=channel, message=message).to_fetcher_payload() for message in messages]

        return self._read_channel(channel=channel, reader=read_messages)

    def download_document(self, message: dict[str, Any]) -> bytes:
        telethon_message = message.get("_message")
        if telethon_message is None:
            raise RuntimeError("Telethon download requires the original message object.")
        try:
            return self._download_document_via_message(telethon_message=telethon_message, message=message)
        except (ConnectionError, TypeError):
            return self._download_document_via_refetch(message=message)

    def _download_document_via_message(self, *, telethon_message: Any, message: dict[str, Any]) -> bytes:
        with TemporaryDirectory(dir=self.config.paths.state_dir) as tmp_dir:
            tmp_path = Path(tmp_dir) / (message["document"].get("file_name") or f"{message['message_id']}.pdf")
            result = telethon_message.download_media(file=str(tmp_path))
            if inspect.isawaitable(result):
                close = getattr(result, "close", None)
                if callable(close):
                    close()
                raise TypeError("download_media returned awaitable in sync path")
            if result is None:
                raise RuntimeError(f"Failed to download media for message {message['message_id']}")
            return Path(result).read_bytes()

    def _download_document_via_refetch(self, *, message: dict[str, Any]) -> bytes:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            with self._isolated_session_path() as isolated_session_path:
                return self._download_document_via_refetch_sync(message=message, session_path=isolated_session_path)
        return self._download_document_via_refetch_in_thread(message=message)

    def _download_document_via_refetch_sync(self, *, message: dict[str, Any], session_path: Path | None = None) -> bytes:
        channel = str((message.get("chat") or {}).get("title") or "")
        message_id = int(message["message_id"])
        with self._sync_client_context(session_path=session_path) as client:
            entity = self._resolve_entity(client, channel)
            telethon_message = client.get_messages(entity, ids=message_id)
            if telethon_message is None:
                raise RuntimeError(f"Failed to refetch message {message_id} for channel {channel}")
            return self._download_document_via_message(telethon_message=telethon_message, message=message)

    def _download_document_via_refetch_in_thread(self, *, message: dict[str, Any]) -> bytes:
        result: dict[str, bytes] = {}
        error: dict[str, BaseException] = {}

        def run() -> None:
            try:
                with self._isolated_session_path() as isolated_session_path:
                    result["payload"] = self._download_document_via_refetch_sync(message=message, session_path=isolated_session_path)
            except BaseException as exc:  # pragma: no cover - re-raised on caller thread
                error["exc"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join()
        if "exc" in error:
            raise error["exc"]
        return result["payload"]

    async def watch_channel(self, *, channel: str, until, on_message) -> None:
        await self.watch_channels(channels=[channel], until=until, on_message=on_message)

    async def watch_channels(self, *, channels: list[str], until, on_message) -> None:
        from telethon import TelegramClient, events

        remaining_seconds = max((until - self._now_like(until)).total_seconds(), 0)
        if remaining_seconds <= 0:
            return

        session_stem = self.config.paths.telethon_session_path.with_suffix("")
        async with TelegramClient(str(session_stem), self.settings.api_id, self.settings.api_hash) as client:
            if not await client.is_user_authorized():
                raise RuntimeError("Telethon session is not authorized. Run auth-login first.")
            resolved_channels = list(dict.fromkeys(channels))
            pending_tasks: set[asyncio.Task] = set()
            accepting_messages = True

            async def process_payload(payload: dict[str, Any]) -> None:
                await self._maybe_await(on_message(payload))

            handlers = []
            for channel in resolved_channels:
                entity = await self._resolve_entity_async(client, channel)

                @client.on(events.NewMessage(chats=entity))
                async def handler(event, *, channel: str = channel) -> None:
                    accepted_at = self._now_like(until)
                    if not accepting_messages or accepted_at >= until:
                        return
                    payload = self._adapt_message(channel=channel, message=event.message).to_fetcher_payload()
                    payload["_accepted_at"] = accepted_at.isoformat()
                    task = asyncio.create_task(process_payload(payload))
                    pending_tasks.add(task)
                    task.add_done_callback(pending_tasks.discard)

                handlers.append(handler)

            try:
                await asyncio.sleep(remaining_seconds)
            finally:
                accepting_messages = False
                for handler in handlers:
                    client.remove_event_handler(handler)
                if pending_tasks:
                    await asyncio.gather(*pending_tasks, return_exceptions=True)

    def _build_client(self, session_path: Path | None = None):
        from telethon.sync import TelegramClient

        session_stem = (session_path or self.config.paths.telethon_session_path).with_suffix("")
        return TelegramClient(str(session_stem), self.settings.api_id, self.settings.api_hash)

    @contextmanager
    def _sync_client_context(self, *, session_path: Path | None = None):
        params = inspect.signature(self._build_client).parameters
        client_context = (
            self._build_client(session_path=session_path)
            if "session_path" in params
            else self._build_client()
        )
        with client_context as client:
            yield client

    @contextmanager
    def _isolated_sync_client(self):
        with self._isolated_session_path() as isolated_session_path:
            with self._sync_client_context(session_path=isolated_session_path) as client:
                yield client

    def _read_channel(self, *, channel: str, reader: Callable[[Any, Any], T]) -> T:
        with self._isolated_sync_client() as client:
            entity = self._resolve_entity(client, channel)
            return reader(client, entity)

    @contextmanager
    def _isolated_session_path(self):
        source_session_path = self.config.paths.telethon_session_path
        with TemporaryDirectory(dir=self.config.paths.state_dir) as tmp_dir:
            isolated_session_path = Path(tmp_dir) / source_session_path.name
            if source_session_path.exists():
                shutil.copy2(source_session_path, isolated_session_path)
            yield isolated_session_path

    @staticmethod
    def _session_password_error():
        from telethon.errors import SessionPasswordNeededError

        return SessionPasswordNeededError

    @staticmethod
    def _resolve_entity(client: Any, channel: str):
        try:
            return client.get_entity(channel)
        except ValueError:
            if channel.startswith("@"):
                raise
            return client.get_entity(f"@{channel}")

    @staticmethod
    async def _resolve_entity_async(client: Any, channel: str):
        try:
            return await client.get_entity(channel)
        except ValueError:
            if channel.startswith("@"):
                raise
            return await client.get_entity(f"@{channel}")

    @staticmethod
    def _adapt_message(*, channel: str, message: Any) -> TelethonMessageAdapter:
        file_id, file_unique_id, file_name, mime_type = TelethonChannelClient._extract_document_payload(message=message)
        return TelethonMessageAdapter(
            message_id=int(message.id),
            date=int(message.date.timestamp()) if getattr(message, "date", None) else None,
            chat={"title": channel},
            caption=getattr(message, "message", None),
            document={
                "file_id": file_id or f"message-{message.id}",
                "file_unique_id": file_unique_id or f"telethon-message-{message.id}",
                "file_name": file_name,
                "mime_type": mime_type,
            },
            _message=message,
        )

    @staticmethod
    def _extract_document_payload(*, message: Any) -> tuple[str | None, str | None, str | None, str]:
        document = getattr(message, "document", None)
        if document is None:
            return None, None, None, ""
        file_id = str(getattr(document, "id", "")) or None
        file_unique_id = f"telethon-{file_id}" if file_id else f"telethon-{message.id}"
        mime_type = str(getattr(document, "mime_type", "") or "")
        file_name = None
        for attribute in getattr(document, "attributes", []) or []:
            candidate = getattr(attribute, "file_name", None)
            if candidate:
                file_name = str(candidate)
                break
        return file_id, file_unique_id, file_name, mime_type

    @staticmethod
    async def _maybe_await(result) -> None:
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            await result

    @staticmethod
    def _now_like(reference):
        return datetime.now(reference.tzinfo)


class FixtureTelegramClient:
    def __init__(self, updates: list[dict[str, Any]]) -> None:
        self._updates = updates

    def get_updates(self, *, offset: int | None, limit: int) -> list[dict[str, Any]]:
        return [update for update in self._updates if offset is None or update["update_id"] >= offset][:limit]

    @staticmethod
    def get_file(file_id: str) -> dict[str, str]:
        return {"file_path": f"docs/{file_id}.pdf"}

    @staticmethod
    def download_file(file_path: str) -> bytes:
        file_id = Path(file_path).stem
        fixture_bytes = {
            "file-001": (
                b"Executive Summary:\nNVIDIA (NVDA) and TSMC are expanding advanced packaging.\n\n"
                b"Risks:\nSupply concentration remains a risk for AI accelerators."
            ),
            "file-002": b"Duplicate PDF bytes should be skipped because file_unique_id already exists.",
        }
        return fixture_bytes.get(file_id, f"PDF bytes for {file_id}".encode())

    @classmethod
    def from_fixture_path(cls, fixture_path: Path) -> "FixtureTelegramClient":
        import json

        return cls(json.loads(Path(fixture_path).read_text()))



def auth_login(*, base_dir: Path, config: ArasConfig) -> None:
    TelethonChannelClient(base_dir=base_dir, config=config).ensure_authorized()
