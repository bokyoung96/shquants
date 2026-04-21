# 2026-04-16 watch/fetch cleanup plan

## Scope
Refactor only the recently touched Telegram watch/fetch code in `analysts/` without changing behavior.

## Guardrails
- Preserve the current PDF-only watch behavior.
- Preserve catch-up-before-watch behavior.
- Preserve async refetch fallback behavior.
- Keep diffs small and reversible.

## Planned cleanup
1. Extract watch catch-up aggregation from `cli.py` into a helper.
2. Extract repeated `last_seen` persistence in `fetcher.py` into a helper.
3. Extract isolated Telethon session-copy setup in `telethon_client.py` into a helper.
4. Re-run focused regression tests after refactor.
5. Restart watcher until 2026-04-16 13:30:00+09:00 and confirm healthy logs.
