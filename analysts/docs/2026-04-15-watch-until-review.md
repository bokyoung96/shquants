# Analysts watch-until review notes

## Scope reviewed
- `.omx/context/analysts-watch-until-20260415T050335Z.md`
- `.omx/plans/prd-analysts-watch-until.md`
- `.omx/plans/test-spec-analysts-watch-until.md`
- `src/analysts/cli.py`
- `src/analysts/pipeline.py`
- `src/analysts/fetcher.py`
- `src/analysts/telethon_client.py`
- `src/analysts/storage.py`
- `tests/test_cli.py`
- `tests/test_fetcher.py`
- `tests/test_pipeline.py`

## Safe test-first sequence
1. **Lock the CLI surface first.**
   - Add a red test in `tests/test_cli.py` for `watch-until --channel DOC_POOL --until 2026-04-15T17:30:00+09:00`.
   - Keep the assertion narrow: parser + dispatch only. Monkeypatch the runtime entrypoint instead of booting Telethon.
2. **Add watcher-runtime unit tests in a dedicated file.**
   - Create `tests/test_watcher.py` for deadline handling, duplicate skipping, immediate summarize start, summarize retry, and continue-after-failure behavior.
   - Inject `now`, `sleep`, and the per-report processing callback so deadline tests stay deterministic and fast.
3. **Reuse existing fetcher semantics instead of re-testing them indirectly.**
   - `tests/test_fetcher.py` already locks first-run seeding, duplicate skipping, and no-advance-on-download-failure behavior.
   - Only extend fetcher tests if the watcher changes the fetch contract itself.
4. **Add a single pipeline-level red test only where orchestration changes.**
   - If watch mode needs a new per-report processing helper, cover that helper directly in `tests/test_pipeline.py` with fake reports/summarizer objects.
   - Do not push retry semantics down into the summarizer tests; the requirement belongs at the watch orchestration boundary.
5. **Update README examples last.**
   - Document the new command only after CLI output and argument names are stable.

## Current-state integration risks
- **`cli.py` is fully branch-driven today.** A long-running async command added directly into `main()` can make sync/async control flow and monkeypatching awkward. Keep one small boundary such as `asyncio.run(...)` in CLI and move the runtime loop elsewhere.
- **`ArasPipeline.run_once()` couples fetch + summarize in one synchronous pass.** Reusing it inside a watcher would make one summarize failure abort the entire watch iteration, which conflicts with the requirement to retry immediately and keep watching.
- **First-run seeding is intentionally non-backfilling.** `TelegramFetcher._poll_telethon_once()` seeds `last_seen_message_id` to the latest post and returns no downloads on first run. `watch-until` should preserve that behavior unless the product requirement explicitly changes.
- **Hydration currently performs a full report scan.** `ArasPipeline._hydrate_report()` loops through `store.list_reports()` to find the inserted report by `file_unique_id`. That is acceptable for `run-once`, but a long-running watcher would repeatedly pay an O(n) lookup cost.
- **Telethon ingress is poll-based, not event-driven.** `TelethonChannelClient.iter_channel_messages()` is a snapshot fetch. The watch loop therefore needs explicit polling cadence, sleep injection, and deadline checks to avoid flaky tests and busy waiting.
- **Summarize failures currently bubble out.** `summarize_report()` has no retry or per-report isolation. The watch loop must catch summarize exceptions outside the fetch layer so download dedupe state remains correct while retries happen immediately.
- **State ownership is split across download and summarize phases.** `last_seen_message_id` and `reports.file_unique_id` should advance only for successful download persistence; summarize retry state should stay in the watch loop and must not mutate crawl state.

## Bounded refactor targets after the feature lands
1. **Extract shared per-report processing from `ArasPipeline.run_once()`.**
   - Add a small helper for `hydrate -> summarize -> collect outputs` so `run-once` and `watch-until` share the same report-processing path without duplicating orchestration.
2. **Add a store lookup by `file_unique_id`.**
   - Replace the `list_reports()` scan used by `_hydrate_report()` with a direct query helper in `storage.py`.
   - Keep this bounded to the touched watch/pipeline flow; no wider storage redesign is needed.
3. **Isolate the watch loop in a new runtime module.**
   - Keep `cli.py` responsible for argument parsing and a single call boundary.
   - Put deadline checks, polling cadence, retry counting, and progress logging in a dedicated watcher module so tests can inject time/sleep dependencies cleanly.
4. **Keep summarize retry policy outside `summarizer.py`.**
   - The retry requirement is operational, not model-contract logic. Preserve `CodexAnalystSummarizer` as a single-attempt primitive and let the watcher own retry/continue semantics.

## Recommended merge checklist for the watch lane
- `watch-until` parser/dispatch test added before implementation
- watcher runtime tests cover deadline reached, unique PDF, duplicate skip, retry-success, and retry-failure-continue
- no duplicate DB rows or duplicate summaries when the same file is replayed
- first-run no-history seeding behavior remains explicit and documented
- README examples match final command spelling and timestamp format exactly
- targeted watch tests plus full `analysts/tests` suite stay green

## Verification after merge
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/pytest tests/test_cli.py tests/test_fetcher.py tests/test_pipeline.py tests/test_watcher.py -q
PYTHONPATH=src ../.venv/bin/pytest tests -q
PYTHONPATH=src ../.venv/bin/python -m compileall src tests
```
