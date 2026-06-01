# Massive Orderbook Example

This folder contains a minimal Massive REST example for stock NBBO/top-of-book
quotes. Massive documents stock quotes at `GET /v3/quotes/{stockTicker}` and
describes each record as bid/ask prices, sizes, exchanges, and timestamps.

`config.json` contains the local API key and is ignored by git. Keep the file
local because it contains credentials.

Files:

- `client.py`: config loading, Massive connection, generic GET JSON requests.
- `orderbook.py`: stock quote/top-of-book request helpers and CLI.

Run:

```powershell
uv run python -m orderbook.orderbook AAPL --timestamp 2026-05-29 --limit 1
```

Print the raw Massive response:

```powershell
uv run python -m orderbook.orderbook AAPL --timestamp 2026-05-29 --limit 1 --raw
```
