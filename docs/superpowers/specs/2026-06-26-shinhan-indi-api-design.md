# Shinhan iIndi API Factory Design

## Goal

Create a clear `api/` package for Shinhan iIndi connection, real-time quotes, and stock orders.

## Context

iIndi is an OCX/ActiveX-style API, not a REST API. Public examples show request
flows built around `SetQueryName`, `SetSingleData`, request methods, and callback
handlers. The user already keeps iIndi logged in, so the client must reuse an
active session and only call `StartIndi` when the session is closed and config
credentials are present.

## Approach

Use a small object factory:

- `make(config_path)` returns an `Indi` client.
- `Indi.connect()` starts the local iIndi bridge and checks session state.
- `Indi.subscribe_quote(code, handler)` registers `SC` real-time quote flow.
- `Indi.buy()` and `Indi.sell()` send the `SABA101U1` order TR.

The runtime OCX control is wrapped by `Control`, so tests can use fake controls
without installing iIndi. Runtime imports of `GiExpertControl` are delayed until
`make()` is called without injected controls.

## Data Flow

```text
config.json -> make() -> Indi
Indi.connect() -> RunIndiPython/GetCommState/StartIndi
subscribe_quote() -> SC initial registration -> ReceiveData -> RequestRTReg -> ReceiveRTData
buy/sell/order() -> SABA101U1 fields -> RequestData -> ReceiveData -> OrderResult
```

## Safety

`api/config.json` is gitignored. `api/orders.py` uses dry-run by default and only
sends a live order when `--send` is passed.

## Testing

Unit tests use fake controls to verify:

- config loading
- session reuse vs login start
- real-time quote registration and parsing
- SABA101U1 field mapping
- script helper behavior
