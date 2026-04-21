# Watch-Until Cleanup Plan

## Scope
Touched async watcher path only:
- `src/analysts/fetcher.py`
- `src/analysts/cli.py`
- `src/analysts/watcher.py`
- related tests/docs if imports or command text change

## Behavior lock
- `cd /Users/bkchoi/Desktop/GitHub/1w1a && .venv/bin/pytest analysts/tests/test_fetcher.py analysts/tests/test_cli.py analysts/tests/test_watcher.py -q`
- `cd /Users/bkchoi/Desktop/GitHub/1w1a && .venv/bin/pytest analysts/tests -q`

## Smell-focused pass order
1. Remove duplication between one-shot Telethon ingest and watch-time Telethon ingest in `fetcher.py`.
2. Keep `watch-until` CLI wiring shallow and command-focused.
3. Avoid widening scope beyond touched watcher path.

## Non-goals
- No broader pipeline rewrite.
- No dependency changes.
- No behavior changes to dedupe, retry, or deadline semantics.
