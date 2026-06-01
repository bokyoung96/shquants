from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx

DEFAULT_BASE_URL = "https://api.massive.com"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class MassiveConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> MassiveConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    api_key = str(data.get("api_key", "")).strip()
    if not api_key:
        raise ValueError(f"Missing api_key in {path}")

    base_url = str(data.get("base_url", DEFAULT_BASE_URL)).strip().rstrip("/")
    if not base_url:
        raise ValueError(f"Missing base_url in {path}")

    return MassiveConfig(api_key=api_key, base_url=base_url)


class MassiveClient:
    def __init__(
        self,
        config: MassiveConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        self._http_client = http_client

    def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> dict[str, Any]:
        request = self.build_get_request(path, params=params)

        if self._http_client is not None:
            return self._send(request, self._http_client)

        with httpx.Client(timeout=10.0) as managed_client:
            return self._send(request, managed_client)

    def build_get_request(
        self,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> httpx.Request:
        query_params: list[tuple[str, str | int]] = []
        if params:
            query_params.extend(params.items())
        query_params.append(("apiKey", self._config.api_key))

        normalized_path = path if path.startswith("/") else f"/{path}"
        url = f"{self._config.base_url.rstrip('/')}{normalized_path}"
        return httpx.Request("GET", url, params=query_params)

    @staticmethod
    def _send(request: httpx.Request, client: httpx.Client) -> dict[str, Any]:
        response = client.send(request)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            redacted_url = request.url.copy_set_param("apiKey", "REDACTED")
            raise RuntimeError(
                f"Massive API request failed with HTTP {status_code} for {redacted_url}"
            ) from None
        return response.json()
