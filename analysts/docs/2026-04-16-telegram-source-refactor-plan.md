# Telegram Source Refactor Cleanup Plan

## Goal
Move Telegram-specific ingestion/runtime code toward `analysts/src/analysts/sources/telegram/` so the source structure becomes symmetric with Gmail, while preserving all existing Telegram behavior.

## Non-negotiable safety rule
Telegram behavior must not regress. Existing CLI commands, watcher semantics, and summary flow must keep working throughout the migration.

## Smells to remove
1. **Asymmetric source structure** — Gmail is under `sources/gmail`, Telegram is still top-level.
2. **Source boundary leakage** — Telegram-specific client/fetch/watch logic sits beside shared pipeline modules.
3. **Naming inconsistency** — source-specific modules are not grouped under a common source namespace.

## Safe migration strategy
1. Lock behavior with the focused Telegram regression slice.
2. Create `sources/telegram/` package with mirrored filenames:
   - `client.py`
   - `fetcher.py`
   - `watcher.py`
   - `runner.py`
3. First migration slice should use **compatibility wrappers/shims** at the old module paths so imports keep working.
4. Update internal imports gradually.
5. Re-run the Telegram regression slice after every move.
6. Only remove top-level wrappers after all imports/tests are green and the package structure is stable.

## Boundaries
- **Telegram-specific** code should end up under `sources/telegram/`.
- **Shared** code remains top-level (`pipeline.py`, `parser.py`, `router.py`, `summarizer.py`, etc.).
- Do not widen the refactor beyond Telegram source modules and the minimum import sites needed.

## Verification gates
- `analysts/tests/test_telethon_client.py`
- `analysts/tests/test_fetcher.py`
- `analysts/tests/test_watcher.py`
- `analysts/tests/test_cli.py`
- `analysts/tests/test_pipeline.py`

## Sequencing recommendation
### Slice 1
- Add `sources/telegram/` package
- Copy/move implementation there
- Keep top-level wrappers that re-export old symbols
- Verify all Telegram tests remain green

### Slice 2
- Move import sites onto `sources/telegram/*`
- Keep wrappers in place
- Verify again

### Slice 3
- Remove wrappers only if zero remaining imports rely on them
- Re-run regression slice and broader analysts checks

## Risk flags
- Large rename-only diffs can still break runtime imports.
- Team workers may create noisy merge conflicts on file moves; leader should prefer small, verifiable integration slices.
- The repo currently has unrelated working tree changes outside this refactor scope; do not mix them into Telegram moves.
