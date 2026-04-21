import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

from analysts.config import build_config
from analysts.telethon_client import TelethonChannelClient


class FakeDisconnectedMessage:
    def download_media(self, *, file: str):
        raise ConnectionError("Cannot send requests while disconnected")


class FakeAsyncMessage:
    async def download_media(self, *, file: str):
        return file


class FakeDownloadedMessage:
    def download_media(self, *, file: str):
        Path(file).write_bytes(b"PDF bytes via refetch")
        return file


class FakeRefetchClient:
    def get_messages(self, entity, ids: int):
        return FakeDownloadedMessage()


class FakeLatestMessage:
    def __init__(self, message_id: int):
        self.id = message_id


class FakeIterMessage:
    def __init__(self, message_id: int):
        self.id = message_id
        self.date = None
        self.message = None
        self.document = None


class FakeDocument:
    def __init__(self, *, doc_id: int, mime_type: str = "", file_name: str | None = None):
        self.id = doc_id
        self.mime_type = mime_type
        self.attributes = [] if file_name is None else [type("Attr", (), {"file_name": file_name})()]


class FakeSyncClientContext:
    def __enter__(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return FakeRefetchClient()
        raise RuntimeError('You must use "async with" if the event loop is running (i.e. you are inside an "async def")')

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeReadClient:
    def __init__(self, messages=None):
        self.messages = messages or []

    def get_messages(self, entity, limit=None, ids=None):
        if ids is not None:
            return None
        return [FakeLatestMessage(message_id=message_id) for message_id in self.messages[:limit]]

    def iter_messages(self, entity, limit, min_id, reverse):
        for message_id in self.messages:
            if message_id > min_id:
                yield FakeIterMessage(message_id)


def test_download_document_refetches_when_original_message_client_is_disconnected(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'telegram': {
                    'mode': 'telethon',
                    'api_id': 123456,
                    'api_hash': 'super-secret-hash',
                    'phone_number': '+821012345678',
                    'channel': 'DOC_POOL',
                    'session_name': 'doc-pool',
                    'pdf_only': True,
                }
            }
        )
    )
    config = build_config(tmp_path)
    client = TelethonChannelClient(base_dir=tmp_path, config=config)
    payload = {
        "message_id": 503,
        "chat": {"title": "DOC_POOL"},
        "document": {
            "file_name": "fresh.pdf",
        },
        "_message": FakeDisconnectedMessage(),
    }
    calls: list[dict] = []

    def fake_refetch(*, message: dict) -> bytes:
        calls.append(message)
        return b"PDF bytes via refetch"

    monkeypatch.setattr(client, "_download_document_via_refetch", fake_refetch)

    result = client.download_document(payload)

    assert result == b"PDF bytes via refetch"
    assert calls == [payload]


def test_download_document_refetches_when_download_media_returns_coroutine(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'telegram': {
                    'mode': 'telethon',
                    'api_id': 123456,
                    'api_hash': 'super-secret-hash',
                    'phone_number': '+821012345678',
                    'channel': 'DOC_POOL',
                    'session_name': 'doc-pool',
                    'pdf_only': True,
                }
            }
        )
    )
    config = build_config(tmp_path)
    client = TelethonChannelClient(base_dir=tmp_path, config=config)
    payload = {
        "message_id": 504,
        "chat": {"title": "DOC_POOL"},
        "document": {
            "file_name": "fresh.pdf",
        },
        "_message": FakeAsyncMessage(),
    }
    calls: list[dict] = []

    def fake_refetch(*, message: dict) -> bytes:
        calls.append(message)
        return b"PDF bytes via refetch"

    monkeypatch.setattr(client, "_download_document_via_refetch", fake_refetch)

    result = client.download_document(payload)

    assert result == b"PDF bytes via refetch"
    assert calls == [payload]


def test_download_document_refetches_inside_running_event_loop(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'telegram': {
                    'mode': 'telethon',
                    'api_id': 123456,
                    'api_hash': 'super-secret-hash',
                    'phone_number': '+821012345678',
                    'channel': 'DOC_POOL',
                    'session_name': 'doc-pool',
                    'pdf_only': True,
                }
            }
        )
    )
    config = build_config(tmp_path)
    client = TelethonChannelClient(base_dir=tmp_path, config=config)
    payload = {
        "message_id": 505,
        "chat": {"title": "DOC_POOL"},
        "document": {
            "file_name": "fresh.pdf",
        },
        "_message": FakeAsyncMessage(),
    }

    monkeypatch.setattr(client, "_build_client", lambda: FakeSyncClientContext())
    monkeypatch.setattr(client, "_resolve_entity", lambda raw_client, channel: channel)

    async def run() -> bytes:
        return client.download_document(payload)

    result = asyncio.run(run())

    assert result == b"PDF bytes via refetch"


def test_download_document_refetches_with_isolated_session_without_running_loop(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'telegram': {
                    'mode': 'telethon',
                    'api_id': 123456,
                    'api_hash': 'super-secret-hash',
                    'phone_number': '+821012345678',
                    'channel': 'DOC_POOL',
                    'session_name': 'doc-pool',
                    'pdf_only': True,
                }
            }
        )
    )
    config = build_config(tmp_path)
    client = TelethonChannelClient(base_dir=tmp_path, config=config)
    payload = {
        "message_id": 506,
        "chat": {"title": "DOC_POOL"},
        "document": {
            "file_name": "fresh.pdf",
        },
        "_message": FakeDisconnectedMessage(),
    }
    seen: list[Path] = []

    @contextmanager
    def fake_isolated_session_path():
        path = tmp_path / "isolated.session"
        path.write_bytes(b"session")
        seen.append(path)
        yield path

    def fake_refetch_sync(*, message: dict, session_path: Path | None = None) -> bytes:
        assert message == payload
        assert session_path == seen[0]
        return b"PDF bytes via isolated refetch"

    monkeypatch.setattr(client, "_isolated_session_path", fake_isolated_session_path)
    monkeypatch.setattr(client, "_download_document_via_refetch_sync", fake_refetch_sync)

    result = client.download_document(payload)

    assert result == b"PDF bytes via isolated refetch"
    assert seen


def test_get_latest_message_id_uses_isolated_session_copy(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'telegram': {
                    'mode': 'telethon',
                    'api_id': 123456,
                    'api_hash': 'super-secret-hash',
                    'phone_number': '+821012345678',
                    'channel': 'DOC_POOL',
                    'session_name': 'doc-pool',
                    'pdf_only': True,
                }
            }
        )
    )
    config = build_config(tmp_path)
    config.paths.telethon_session_path.write_bytes(b'session-bytes')
    client = TelethonChannelClient(base_dir=tmp_path, config=config)
    seen_session_paths: list[Path] = []
    seen_session_bytes: list[bytes] = []

    @contextmanager
    def fake_sync_client_context(*, session_path: Path | None = None):
        assert session_path is not None
        seen_session_paths.append(session_path)
        seen_session_bytes.append(session_path.read_bytes())
        yield FakeReadClient(messages=[700])

    monkeypatch.setattr(client, "_sync_client_context", fake_sync_client_context)
    monkeypatch.setattr(client, "_resolve_entity", lambda raw_client, channel: channel)

    result = client.get_latest_message_id(channel='DOC_POOL')

    assert result == 700
    assert seen_session_paths
    assert seen_session_paths[0] != config.paths.telethon_session_path
    assert seen_session_bytes == [b'session-bytes']


def test_iter_channel_messages_uses_isolated_session_copy(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'telegram': {
                    'mode': 'telethon',
                    'api_id': 123456,
                    'api_hash': 'super-secret-hash',
                    'phone_number': '+821012345678',
                    'channel': 'DOC_POOL',
                    'session_name': 'doc-pool',
                    'pdf_only': True,
                }
            }
        )
    )
    config = build_config(tmp_path)
    config.paths.telethon_session_path.write_bytes(b'session-bytes')
    client = TelethonChannelClient(base_dir=tmp_path, config=config)
    seen_session_paths: list[Path] = []

    @contextmanager
    def fake_sync_client_context(*, session_path: Path | None = None):
        assert session_path is not None
        seen_session_paths.append(session_path)
        yield FakeReadClient(messages=[701, 702])

    monkeypatch.setattr(client, "_sync_client_context", fake_sync_client_context)
    monkeypatch.setattr(client, "_resolve_entity", lambda raw_client, channel: channel)

    payloads = client.iter_channel_messages(channel='DOC_POOL', after_message_id=700, limit=10)

    assert [payload['message_id'] for payload in payloads] == [701, 702]
    assert seen_session_paths
    assert seen_session_paths[0] != config.paths.telethon_session_path


def test_adapt_message_does_not_fake_pdf_filename_when_missing(tmp_path: Path) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'telegram': {
                    'mode': 'telethon',
                    'api_id': 123456,
                    'api_hash': 'super-secret-hash',
                    'phone_number': '+821012345678',
                    'channel': 'DOC_POOL',
                    'session_name': 'doc-pool',
                    'pdf_only': True,
                }
            }
        )
    )
    config = build_config(tmp_path)
    client = TelethonChannelClient(base_dir=tmp_path, config=config)
    message = type(
        "Message",
        (),
        {
            "id": 900,
            "date": None,
            "message": "",
            "document": FakeDocument(doc_id=111, mime_type="", file_name=None),
        },
    )()

    payload = client._adapt_message(channel='DOC_POOL', message=message).to_fetcher_payload()

    assert payload['document']['file_name'] is None
    assert payload['document']['mime_type'] == ''


def test_adapt_message_preserves_pdf_mime_without_filename(tmp_path: Path) -> None:
    (tmp_path / 'config.local.json').write_text(
        json.dumps(
            {
                'telegram': {
                    'mode': 'telethon',
                    'api_id': 123456,
                    'api_hash': 'super-secret-hash',
                    'phone_number': '+821012345678',
                    'channel': 'DOC_POOL',
                    'session_name': 'doc-pool',
                    'pdf_only': True,
                }
            }
        )
    )
    config = build_config(tmp_path)
    client = TelethonChannelClient(base_dir=tmp_path, config=config)
    message = type(
        "Message",
        (),
        {
            "id": 901,
            "date": None,
            "message": "",
            "document": FakeDocument(doc_id=222, mime_type="application/pdf", file_name=None),
        },
    )()

    payload = client._adapt_message(channel='DOC_POOL', message=message).to_fetcher_payload()

    assert payload['document']['file_name'] is None
    assert payload['document']['mime_type'] == 'application/pdf'
