from pathlib import Path
from urllib.parse import parse_qs, urlparse

from analysts.sources.gmail.client import GmailApiClient


def test_client_uses_inline_secret_json_without_path(tmp_path: Path) -> None:
    client = GmailApiClient(
        credentials_path=None,
        credentials_json={
            "installed": {
                "client_id": "abc.apps.googleusercontent.com",
                "client_secret": "secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        },
        token_path=tmp_path / "gmail-token.json",
    )

    config = client.client_config()

    assert config["client_id"] == "abc.apps.googleusercontent.com"
    assert config["client_secret"] == "secret"


def test_build_authorization_url_contains_expected_google_params(tmp_path: Path) -> None:
    client = GmailApiClient(
        credentials_path=None,
        credentials_json={
            "installed": {
                "client_id": "abc.apps.googleusercontent.com",
                "client_secret": "secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        },
        token_path=tmp_path / "gmail-token.json",
    )

    url = client.build_authorization_url(redirect_uri="http://localhost:8765/")
    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert params["client_id"] == ["abc.apps.googleusercontent.com"]
    assert params["access_type"] == ["offline"]
    assert params["response_type"] == ["code"]
    assert "https://www.googleapis.com/auth/gmail.readonly" in params["scope"][0]


def test_get_attachment_data_decodes_gmail_base64_payload(tmp_path: Path, monkeypatch) -> None:
    client = GmailApiClient(
        credentials_path=None,
        credentials_json={
            "installed": {
                "client_id": "abc.apps.googleusercontent.com",
                "client_secret": "secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        },
        token_path=tmp_path / "gmail-token.json",
    )

    monkeypatch.setattr(
        client,
        "_gmail_get",
        lambda *_args, **_kwargs: {"data": "bXNnLTE6YXR0LTE"},
    )

    assert client.get_attachment_data(message_id="msg-1", attachment_id="att-1") == b"msg-1:att-1"
