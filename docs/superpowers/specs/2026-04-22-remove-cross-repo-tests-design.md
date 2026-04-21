# Remove Cross-Repo Test References Design

## Goal

Remove test references in `shquants` that depend on modules owned by `1w1a` rather than this repository.

## Scope

The current repo-wide uv environment is working, but several tests still fail during collection because they depend on modules that are not part of `shquants`:

- `kis`
- `crypto.dashboard`

The user confirmed those integrations are not needed in `shquants`.

## Chosen Approach

Apply the smallest possible change:

1. delete `tests/crypto_dashboard/test_app.py`
2. remove the `KISConfig` import and assertion from `tests/test_smoke.py`
3. delete the KIS-only test files under `tests/kis/`

This preserves the remaining `backtesting` smoke coverage while removing cross-repo coupling.

## Verification

Verify in two stages:

1. run the previously failing targeted tests to confirm the collection errors are gone
2. run the full repo test suite with `uv run python -m pytest -q`
