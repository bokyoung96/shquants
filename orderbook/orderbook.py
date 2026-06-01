from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import httpx

from orderbook.client import (
    DEFAULT_BASE_URL,
    DEFAULT_CONFIG_PATH,
    MassiveClient,
    MassiveConfig,
    load_config,
)


def build_quote_request(
    *,
    api_key: str,
    ticker: str,
    timestamp: str | None = None,
    limit: int = 10,
    order: str = "desc",
    sort: str = "timestamp",
    base_url: str = DEFAULT_BASE_URL,
) -> httpx.Request:
    symbol = _normalize_ticker(ticker)
    params = _quote_params(timestamp=timestamp, limit=limit, order=order, sort=sort)
    return MassiveClient(MassiveConfig(api_key=api_key, base_url=base_url)).build_get_request(
        f"/v3/quotes/{symbol}",
        params=params,
    )


def fetch_quotes(
    *,
    config: MassiveConfig,
    ticker: str,
    timestamp: str | None = None,
    limit: int = 10,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    symbol = _normalize_ticker(ticker)
    params = _quote_params(timestamp=timestamp, limit=limit)
    return MassiveClient(config, http_client=client).get_json(f"/v3/quotes/{symbol}", params=params)


def summarize_quote(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results") or []
    if not results:
        raise ValueError("Massive response did not include quote results")

    quote = results[0]
    bid_price = quote.get("bid_price")
    ask_price = quote.get("ask_price")
    spread = None
    if bid_price is not None and ask_price is not None:
        spread = round(float(ask_price) - float(bid_price), 10)

    return {
        "bid_price": bid_price,
        "bid_size": quote.get("bid_size"),
        "ask_price": ask_price,
        "ask_size": quote.get("ask_size"),
        "spread": spread,
        "sip_timestamp": quote.get("sip_timestamp"),
    }


def _quote_params(
    *,
    timestamp: str | None = None,
    limit: int = 10,
    order: str = "desc",
    sort: str = "timestamp",
) -> dict[str, str | int]:
    if limit < 1:
        raise ValueError("limit must be greater than zero")

    params: dict[str, str | int] = {}
    if timestamp:
        params["timestamp"] = timestamp
    params["order"] = order
    params["limit"] = limit
    params["sort"] = sort
    return params


def _normalize_ticker(ticker: str) -> str:
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("ticker must not be empty")
    return symbol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Massive NBBO/top-of-book quotes for a stock ticker."
    )
    parser.add_argument("ticker", nargs="?", default="AAPL")
    parser.add_argument("--timestamp", help="YYYY-MM-DD date or nanosecond timestamp")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--raw", action="store_true", help="Print the full Massive response")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        payload = fetch_quotes(
            config=config,
            ticker=args.ticker,
            timestamp=args.timestamp,
            limit=args.limit,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from None

    output = payload if args.raw else summarize_quote(payload)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
