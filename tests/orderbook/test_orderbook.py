from __future__ import annotations

import json

import httpx
import pytest

from orderbook.client import (
    MassiveConfig,
    MassiveClient,
    load_config,
)
from orderbook.orderbook import (
    build_quote_request,
    fetch_quotes,
    summarize_quote,
)


def test_load_config_reads_api_key_from_json(tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"api_key": "test-key", "base_url": "https://api.example.com"}),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.api_key == "test-key"
    assert config.base_url == "https://api.example.com"


def test_build_quote_request_targets_massive_quotes_endpoint() -> None:
    request = build_quote_request(
        api_key="test-key",
        ticker="aapl",
        timestamp="2026-05-29",
        limit=5,
    )

    assert request.url == httpx.URL(
        "https://api.massive.com/v3/quotes/AAPL"
        "?timestamp=2026-05-29&order=desc&limit=5&sort=timestamp&apiKey=test-key"
    )


def test_summarize_quote_calculates_spread_from_first_result() -> None:
    summary = summarize_quote(
        {
            "status": "OK",
            "results": [
                {
                    "bid_price": 101.1,
                    "bid_size": 7,
                    "ask_price": 101.4,
                    "ask_size": 9,
                    "sip_timestamp": 1_714_567_890_123_456_789,
                }
            ],
        }
    )

    assert summary == {
        "bid_price": 101.1,
        "bid_size": 7,
        "ask_price": 101.4,
        "ask_size": 9,
        "spread": 0.3,
        "sip_timestamp": 1_714_567_890_123_456_789,
    }


def test_fetch_quotes_redacts_api_key_from_http_errors() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(403, json={"status": "ERROR"}, request=request)
    )

    with httpx.Client(transport=transport) as client:
        with pytest.raises(RuntimeError) as exc_info:
            fetch_quotes(
                config=MassiveConfig(api_key="secret-key"),
                ticker="AAPL",
                client=client,
            )

    message = str(exc_info.value)
    assert "403" in message
    assert "apiKey=REDACTED" in message
    assert "secret-key" not in message


def test_massive_client_can_request_other_data_paths() -> None:
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json={"status": "OK", "results": []}, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = MassiveClient(MassiveConfig(api_key="test-key"), http_client=http_client)
        payload = client.get_json("/v3/reference/tickers", params={"market": "stocks"})

    assert payload == {"status": "OK", "results": []}
    assert seen_urls == [
        "https://api.massive.com/v3/reference/tickers?market=stocks&apiKey=test-key"
    ]
