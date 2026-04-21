# Telethon DOC_POOL Crawler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Telethon-backed DOC_POOL crawler in `analysts/` that tracks new posts going forward and feeds the existing ARAS pipeline safely.

**Architecture:** Keep parser/router/analysis/output modules unchanged and swap the Telegram ingress path to a Telethon user-session adapter. Persist per-channel `last_seen_message_id` state in SQLite, use a gitignored local config file for secrets, and add CLI support for auth/login plus run-once execution.

**Tech Stack:** Python 3.11, Telethon, SQLite, pytest

---

## File map
- Create: `analysts/src/analysts/telethon_client.py` — Telethon session/config adapter
- Create: `analysts/tests/test_telethon_config.py` — local config loader tests
- Modify: `analysts/.gitignore` — ignore local config and session artifacts
- Modify: `analysts/src/analysts/config.py` — Telethon config loading + state/session paths
- Modify: `analysts/src/analysts/storage.py` — per-channel crawl state read/write helpers
- Modify: `analysts/src/analysts/fetcher.py` — Telethon crawler + first-run seeding semantics
- Modify: `analysts/src/analysts/pipeline.py` — wire Telethon fetch path
- Modify: `analysts/src/analysts/cli.py` — `auth-login`, Telethon-backed `run-once`, config view
- Modify: `analysts/tests/test_fetcher.py` — Telethon crawl behavior tests
- Modify: `analysts/tests/test_pipeline.py` — pipeline behavior tests for new-post semantics
- Modify: `analysts/README.md` — setup and auth instructions
- Modify: `analysts/pyproject.toml` — Telethon dependency

## Tasks
### Task 1: Lock config and secret-handling behavior
- [ ] Add failing tests for local config loading and missing-secret validation.
- [ ] Implement gitignored `config.local.json` support and session-file path defaults.
- [ ] Run targeted config tests.

### Task 2: Lock per-channel crawl state behavior
- [ ] Add failing tests for first-run seeding and new-post-only filtering.
- [ ] Implement per-channel `last_seen_message_id` storage helpers.
- [ ] Run fetcher/storage tests.

### Task 3: Add Telethon client and fetcher integration
- [ ] Add failing tests for PDF-only message handling and no-advance-on-failure semantics.
- [ ] Implement Telethon client adapter and Telethon-backed fetcher flow.
- [ ] Run fetcher tests.

### Task 4: Add CLI auth/run wiring
- [ ] Add failing tests or smoke coverage for config display and auth command plumbing.
- [ ] Implement `auth-login` and Telethon-backed `run-once` wiring.
- [ ] Run CLI/pipeline verification.

### Task 5: Verify documentation and full suite
- [ ] Update README with local setup, login, and run commands.
- [ ] Run `PYTHONPATH=src ../.venv/bin/pytest tests -q` from `analysts/`.
- [ ] Run a minimal config display smoke command.
