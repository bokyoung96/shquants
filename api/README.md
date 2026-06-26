# Shinhan iIndi API

This folder wraps Shinhan iIndi's OCX-style API with small Python objects.

## Pipeline

1. Put local credentials in `api/config.json`.
2. Build a client with `api.make()`.
3. Call `client.connect()`.
4. Subscribe to ticks with `client.subscribe_quote(code, handler)`.
5. Send orders with `client.buy(...)`, `client.sell(...)`, or `client.order(...)`.
6. Stop real-time streams with `client.unsubscribe_quote()`.

`api/config.json` is ignored by git. Copy `api/config.example.json` and fill only
the fields needed by your local iIndi session.

## Runtime

iIndi is Windows/OCX based. Run these commands from the Python environment where
`GiExpertControl` is available and iIndi is installed.

```powershell
uv run python -m api.connect --config api/config.json
uv run python -m api.quotes 005930 000660 --config api/config.json
uv run python -m api.orders --side buy --code 005930 --qty 1 --price 71000 --hoga 0 --send
```

Orders default to dry-run unless `--send` is passed.

## Code Map

- `factory.py`: `make()` loads config and wires live iIndi controls.
- `client.py`: `Indi` owns connect, quote subscription, and order flow.
- `control.py`: thin adapter over `GiExpertControl` or QAx-style `dynamicCall`.
- `models.py`: plain `Quote`, `Order`, and `OrderResult` data objects.
- `connect.py`, `quotes.py`, `orders.py`: small runnable examples.

## iIndi Call Shape

The wrapper follows the public examples' sequence:

- login/session: `RunIndiPython`, `GetCommState`, optionally `StartIndi`
- request: `SetQueryName`, `SetSingleData`, `RequestData`
- real-time quote: `SetQueryName("SC")`, `RequestRTReg`, `ReceiveRTData`
- order: `SetQueryName("SABA101U1")`, order fields, `RequestData`
