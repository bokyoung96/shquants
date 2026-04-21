from __future__ import annotations

from base64 import urlsafe_b64decode
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import json
import secrets
from pathlib import Path
import subprocess
from threading import Thread
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen
from http.server import BaseHTTPRequestHandler, HTTPServer


class GmailApiClient:
    GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"

    def __init__(
        self,
        *,
        credentials_path: Path | None,
        credentials_json: dict | None,
        token_path: Path,
    ) -> None:
        self.credentials_path = credentials_path
        self.credentials_json = credentials_json
        self.token_path = token_path

    def client_config(self) -> dict[str, Any]:
        payload = self.credentials_json
        if payload is None:
            if self.credentials_path is None:
                raise RuntimeError("Missing Gmail client credentials. Configure client_secret_json or client_secret_path.")
            payload = json.loads(self.credentials_path.read_text())
        installed = payload.get("installed") or payload.get("web")
        if not isinstance(installed, dict):
            raise RuntimeError("Gmail client credentials must include an 'installed' or 'web' object.")
        return installed

    def build_authorization_url(self, *, redirect_uri: str) -> str:
        config = self.client_config()
        params = {
            "client_id": config["client_id"],
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": self.GMAIL_READONLY_SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "state": secrets.token_urlsafe(24),
        }
        return f"{config['auth_uri']}?{urlencode(params)}"

    def ensure_authorized(self) -> None:
        token = self._load_token()
        if token and self._token_is_valid(token):
            return
        if token and token.get("refresh_token"):
            refreshed = self._refresh_access_token(token)
            if refreshed:
                return
        self._run_local_oauth_flow()

    def list_message_ids(self, *, query: str, page_token: str | None = None, limit: int = 50) -> dict:
        payload = self._gmail_get(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            query_params={
                "maxResults": str(limit),
                **({"q": query} if query else {}),
                **({"pageToken": page_token} if page_token else {}),
            },
        )
        return {
            "messages": payload.get("messages", []),
            "next_page_token": payload.get("nextPageToken"),
        }

    def get_message(self, *, message_id: str) -> dict:
        return self._gmail_get(
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            query_params={"format": "full"},
        )

    def get_attachment_data(self, *, message_id: str, attachment_id: str) -> bytes:
        payload = self._gmail_get(
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}/attachments/{attachment_id}",
            query_params={},
        )
        data = payload.get("data")
        if not data:
            return b""
        padded = data + "=" * (-len(data) % 4)
        return urlsafe_b64decode(padded.encode("utf-8"))

    def _gmail_get(self, url: str, *, query_params: dict[str, str]) -> dict[str, Any]:
        self.ensure_authorized()
        token = self._load_token()
        if not token or "access_token" not in token:
            raise RuntimeError("Gmail access token is unavailable after authorization.")
        request = Request(
            f"{url}?{urlencode(query_params)}",
            headers={"Authorization": f"Bearer {token['access_token']}"},
        )
        with urlopen(request, timeout=30) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    def _load_token(self) -> dict[str, Any] | None:
        if not self.token_path.exists():
            return None
        return json.loads(self.token_path.read_text())

    def _save_token(self, token: dict[str, Any]) -> None:
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(json.dumps(token, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _token_is_valid(token: dict[str, Any]) -> bool:
        expires_at = token.get("expires_at")
        if not expires_at:
            return False
        expiry = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        return expiry > datetime.now(UTC) + timedelta(seconds=60)

    def _refresh_access_token(self, token: dict[str, Any]) -> bool:
        config = self.client_config()
        refresh_token = token.get("refresh_token")
        if not refresh_token:
            return False
        payload = urlencode(
            {
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode("utf-8")
        request = Request(
            config["token_uri"],
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urlopen(request, timeout=30) as response:  # noqa: S310
                refreshed = json.loads(response.read().decode("utf-8"))
        except HTTPError:
            return False
        refreshed["refresh_token"] = refresh_token
        refreshed["expires_at"] = (datetime.now(UTC) + timedelta(seconds=int(refreshed.get("expires_in", 0)))).isoformat().replace("+00:00", "Z")
        self._save_token(refreshed)
        return True

    def _run_local_oauth_flow(self) -> None:
        config = self.client_config()
        with self._oauth_callback_server() as server:
            redirect_uri = f"http://127.0.0.1:{server.server_port}/"
            url = self.build_authorization_url(redirect_uri=redirect_uri)
            self._open_browser(url)
            print(f"Open this URL if the browser does not open automatically:\n{url}")
            server.handle_request()
            if not getattr(server, "authorization_code", None):
                raise RuntimeError("Gmail OAuth did not return an authorization code.")
            self._exchange_code_for_token(code=server.authorization_code, redirect_uri=redirect_uri)

    @staticmethod
    def _open_browser(url: str) -> None:
        try:
            subprocess.run(["open", url], check=False, capture_output=True)
        except Exception:
            pass

    def _exchange_code_for_token(self, *, code: str, redirect_uri: str) -> None:
        config = self.client_config()
        payload = urlencode(
            {
                "code": code,
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")
        request = Request(
            config["token_uri"],
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urlopen(request, timeout=30) as response:  # noqa: S310
            token = json.loads(response.read().decode("utf-8"))
        token["expires_at"] = (datetime.now(UTC) + timedelta(seconds=int(token.get("expires_in", 0)))).isoformat().replace("+00:00", "Z")
        self._save_token(token)

    @contextmanager
    def _oauth_callback_server(self):
        class OAuthHandler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                params = parse_qs(urlparse(self.path).query)
                self.server.authorization_code = params.get("code", [None])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Gmail authorization received.</h1>You can return to Codex.</body></html>")

            def log_message(self, format, *args):  # noqa: A003
                return

        server = HTTPServer(("127.0.0.1", 0), OAuthHandler)
        yield server
        Thread(target=server.server_close, daemon=True).start()
