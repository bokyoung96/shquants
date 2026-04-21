# ARAS / analysts

Alpha Research Agent System (ARAS) workspace built only inside `analysts/`.

## What works now
- Telethon-based Telegram crawling for `DOC_POOL`
- PDF-only raw persistence under `data/raw/`
- Token-cheap LLM analyst summaries via local `codex exec`
- Processed artifacts under `data/processed/`
- Saved local Telethon session under `data/state/`
- Watch startup catch-up before switching to live event subscriptions
- Async-safe Telethon refetch fallback for live watcher downloads

## In-flight PDF ingestion contract
The current runtime still uses the lightweight summary extractor, but the active PDF-ingestion lane is targeting a richer artifact contract inside `analysts/` only:

- raw PDFs stay in `data/raw/` as the source of truth
- processed artifacts become inspectable per report under `data/processed/`
- summaries should read chunked PDF content when full extraction succeeds
- future command surface:
  - `ingest-pdf --path ...`
  - `summarize-latest --channel DOC_POOL`
  - `summarize-recent --channel DOC_POOL --limit 10`

Planned processed artifact set:
- `data/processed/*-fulltext.txt`
- `data/processed/*-extraction.json`
- `data/processed/*-images.json`
- `data/processed/*-chunks.json`
- `data/processed/*-summary-input.json`
- `data/processed/*-summary.json`
- `data/processed/*-summary.md`

Until that work lands, the current summarized path still emits `*-raw-text.txt` plus summary input/output artifacts.

## Current workflow
### 1) Authenticate once
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/python -m analysts.cli auth-login
```

### 2) Crawl backlog once from the current `last_seen`
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/python -m analysts.cli run-once --channel DOC_POOL
```

### 3) Summarize the latest downloaded report again
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/python -m analysts.cli summarize-latest --channel DOC_POOL
```

### 4) Watch for new reports until a deadline
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/python -m analysts.cli watch-until \
  --channel DOC_POOL \
  --until 2026-04-15T17:30:00+09:00
```

What the watcher does now:
- runs a **catch-up pass first** using `run-once`
- then switches to Telethon `NewMessage` live subscriptions
- downloads only **actual PDF report posts**
- ignores generic message attachments that do not expose a real PDF filename or PDF mime type

### 4a) Main operator launcher
```bash
cd analysts
./run.sh telegram
```

Telegram behavior:
- reads the default Telegram channel from `config.local.json`
- starts the realtime watcher
- writes watcher logs to `analysts/data/state/telegram.log`

Gmail one-shot:
```bash
cd analysts
./run.sh gmail
```

Gmail behavior:
- syncs recent Gmail once
- summarizes the latest Gmail report once
- appends output to `analysts/data/state/gmail.log`

Useful log commands:
```bash
cd analysts
tail -f data/state/telegram.log
tail -f data/state/gmail.log
```

- deadline must be timezone-aware ISO-8601 when you call `watch-until` directly
- new unique PDF reports are downloaded once and summarized immediately
- startup catch-up reduces the chance of missing posts that arrived before the live watch began
- summarize failures retry immediately without stopping the watch loop
- no new messages are accepted after the deadline; any report already accepted before cutoff is allowed to finish processing
- progress logs stream to stdout and `analysts/data/state/telegram.log`
- heartbeat logs show liveness while the watcher is idle
- new-report detection is event-driven through Telethon `NewMessage` subscriptions, not heartbeat polling

## Gmail source
Gmail is now modeled as a **separate source family** from Telegram rather than as another channel.

Current Gmail assumptions:
- one Gmail message can produce multiple candidate documents
- body text is promoted to a candidate only when rule-based heuristics pass
- ZIP extraction is allowlisted to `.pdf`, `.txt`, and `.html`
- Gmail state lives separately from Telegram state

Current command surface:
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/python -m analysts.cli gmail-auth-login
PYTHONPATH=src ../.venv/bin/python -m analysts.cli gmail-sync-once --limit 20
PYTHONPATH=src ../.venv/bin/python -m analysts.cli gmail-sync-recent --limit 20
PYTHONPATH=src ../.venv/bin/python -m analysts.cli gmail-summarize-latest
PYTHONPATH=src ../.venv/bin/python -m analysts.cli gmail-summarize-recent --limit 10
```

Storage layout for Gmail:
- `data/state/gmail.sqlite3` stores Gmail message records, sync state, and candidate metadata separately from Telegram state.
- `data/raw/gmail/<gmail-message-id>/message.json` stores the fetched Gmail message payload per message container.
- `data/raw/gmail/<gmail-message-id>/body.txt` and `body.html` store extracted plain-text and HTML body artifacts when present.
- `data/raw/gmail/<gmail-message-id>/attachments/manifest.json` records attachment metadata for that message container.
- `data/raw/gmail/<gmail-message-id>/attachments/original/` stores original Gmail attachments such as PDF/TXT/HTML/ZIP payloads.
- `data/raw/gmail/<gmail-message-id>/web/` stores Playwright-captured page HTML/text/screenshot artifacts when a Gmail message is handled through the web-link lane.
- `data/processed/gmail/` holds Gmail-only staged candidate material, such as promoted body-text files and allowlisted ZIP entries.
- `data/processed/report-*.{txt,json,md}` remains the shared analyst artifact surface for both Telegram and Gmail summarization outputs.

Storage layout for Telegram:
- `data/raw/telegram/` stores raw Telegram PDF reports only.
- Telegram is no longer mixed into the `data/raw/` root.

Raw-vs-processed organization notes:
- raw Gmail storage is **message-centric** so one email can keep its body, payload metadata, and attachment manifest together
- raw Telegram storage is **document-centric** because the Telegram source remains a direct PDF feed
- Playwright is used only for the Gmail web-link lane; install browser binaries with `../.venv/bin/python -m playwright install chromium` when setting up a new local environment.
- processed Gmail staging remains **document-centric** so downstream summaries look like the Telegram-style artifact flow
- the Gmail API payload also remains in SQLite via `raw_payload_json` on each stored message record for queryable metadata
- the top-level analyst summaries still point back to the staged source path through `raw_pdf_path`, even when the source is a Gmail body/text candidate rather than a Telegram PDF

## Active pipeline map
The live analysts path is intentionally narrow:

1. `cli.py` — command entrypoints and wiring
2. `telethon_client.py` + `fetcher.py` — Telegram auth/crawl/download
3. `pipeline.py` — top-level orchestration and stale-path recovery
4. `parser.py` + `router.py` — route hints for a report
5. `pdf_ingest.py` — PDF extraction, chunks, page selection, preview artifacts
6. `summarizer.py` — Codex-backed sector/macro summaries
7. `summary_outputs.py` — persisted summary JSON/Markdown artifacts
8. `graphify.py` — optional graphify corpus refresh from processed summaries

Legacy wiki/signal/coordinator modules were removed from the active package surface because they were no longer part of the current production pipeline.

## Local config
Create `analysts/config.local.json` (gitignored):

```json
{
  "telegram": {
    "mode": "telethon",
    "api_id": 123456,
    "api_hash": "replace-me",
    "phone_number": "+821012345678",
    "channel": "DOC_POOL",
    "session_name": "doc-pool",
    "pdf_only": true
  },
  "summary": {
    "provider": "codex_cli",
    "model": "gpt-5.4-mini",
    "reasoning_effort": "low",
    "max_input_chars": 3200,
    "max_key_points": 4,
    "cli_command": "codex"
  }
}
```

Gmail config will live alongside Telegram config under a separate top-level `gmail` key. Expected fields include:
- `account_name`
- `client_secret_path`
- `client_secret_json`
- `token_path`
- `query`
- optional `label_filters`
- `body_candidate_rules`
- `zip_allow_extensions`
- `poll_interval_seconds`

Example Gmail config:

```json
{
  "gmail": {
    "account_name": "reports-primary",
    "client_secret_path": "secrets/google-client.json",
    "token_path": "data/state/gmail-token.json",
    "query": "label:broker-reports newer_than:7d",
    "label_filters": ["Label_Reports"],
    "body_candidate_rules": {
      "min_chars": 200,
      "require_structure": true
    },
    "zip_allow_extensions": [".pdf", ".txt", ".html"],
    "poll_interval_seconds": 300
  }
}
```

## Token-cheap summary design
- raw PDF remains the source of truth
- local extraction builds a compact summary packet first
- two concise analyst lanes run per report:
  - sector analyst
  - macro analyst
- no expensive coordinator by default
- processed outputs are easy to inspect:
  - `data/processed/*-raw-text.txt`
  - `data/processed/*-summary-input.json`
  - `data/processed/*-summary.json`
  - `data/processed/*-summary.md`

## PDF ingestion design references
- `docs/2026-04-15-pdf-ingestion-review.md` — current gap analysis, integration risks, and merge checklist for the ingestion upgrade
- `docs/openclaw-integration.md` — stable command-oriented interface that future OpenClaw automation should call

## Notes
- Generated data is gitignored but still available for inspection locally.
- Telethon session files stay under `data/state/` and are gitignored.
- The local `pyaes` shim exists only to satisfy Telethon import behavior in this environment while `cryptg` handles the actual crypto path.
- `data/raw/`, `data/processed/`, and `data/state/aras.sqlite3` can be safely cleared for a fresh comparison run as long as `data/state/*.session` is kept.

## Verification
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/pytest tests -q
```

Focused Telegram/watch regression slice:
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/pytest \
  tests/test_telethon_client.py \
  tests/test_fetcher.py \
  tests/test_watcher.py \
  tests/test_cli.py \
  tests/test_pipeline.py -q
```

## Graphify wiki/update
```bash
cd analysts
PYTHONPATH=src ../.venv/bin/python -m analysts.cli graphify-update
```

This builds a graphify-ready corpus from processed summary artifacts so a future graphify/OpenClaw layer can update incrementally as new reports are processed.
