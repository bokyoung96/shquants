# ARAS MVP Review Notes

## Scope reviewed
- `src/analysts/parser.py`
- `src/analysts/router.py`
- `src/analysts/agents.py`
- `tests/test_parser.py`
- `tests/test_router.py`
- `tests/test_agents.py`

## Code quality findings
- **Parser contract is explicit.** Text-heavy inputs preserve title/content/sections/entities/tickers with `parse_quality="high"`. Undecodable payloads return a degraded-but-stable document shape with `degraded_reason` instead of throwing.
- **Routing avoids substring false positives.** Matching is token-based, which prevents accidental semiconductor hits from words like `daily`.
- **Agent output is deterministic.** Insights derive from parsed sections, route metadata, and parse quality only; degraded documents get a stable fallback summary, confidence, and risk note.
- **Verification is targeted and repeatable.** Lane-2 tests cover parser extraction, routing taxonomy/fallback behavior, and deterministic insight generation.

## Verification snapshot
- `cd analysts && PYTHONPATH=src ../.venv/bin/pytest tests/test_parser.py -q`
- `cd analysts && PYTHONPATH=src ../.venv/bin/pytest tests/test_router.py -q`
- `cd analysts && PYTHONPATH=src ../.venv/bin/pytest tests/test_agents.py -q`
- `cd analysts && PYTHONPATH=src ../.venv/bin/pytest tests -q`

All commands passed in the worker review lane on 2026-04-14.
