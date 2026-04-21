# Token-Cheap LLM Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce useful low-token LLM analyst summaries for downloaded Telegram PDFs in `analysts/`, while keeping raw artifacts inspectable and preserving token discipline.

**Architecture:** Add a summary-ready extraction layer, a route-aware Codex CLI summarizer with strict JSON output, and processed artifact writers. Only run the minimal analyst lanes needed for a report and keep raw PDFs as the truth source.

**Tech Stack:** Python 3.11, Telethon, Codex CLI, SQLite, pytest

---

## File map
- Create: `analysts/src/analysts/extraction.py` — summary-ready extraction packet builder
- Create: `analysts/src/analysts/summarizer.py` — Codex CLI analyst agent wrapper
- Create: `analysts/tests/test_extraction.py` — extraction packet tests
- Create: `analysts/tests/test_summarizer.py` — summarizer command/payload tests
- Modify: `analysts/src/analysts/config.py` — summarizer settings and token caps
- Modify: `analysts/src/analysts/domain.py` — summary output contracts
- Modify: `analysts/src/analysts/pipeline.py` — extraction → summarization → processed artifact flow
- Modify: `analysts/src/analysts/cli.py` — report-specific summarize command / run output
- Modify: `analysts/src/analysts/router.py` — route hints for lane selection if needed
- Modify: `analysts/README.md` — summary workflow docs
- Modify: `analysts/.gitignore` — ignore generated data artifacts cleanly
- Modify: `analysts/tests/test_pipeline.py` — end-to-end summary artifacts

## Tasks
### Task 1: Lock extraction packet behavior
- [ ] Add failing tests for extraction packet creation and text-cap truncation.
- [ ] Implement summary-ready extraction packet builder.
- [ ] Run targeted extraction tests.

### Task 2: Lock analyst summarizer contract
- [ ] Add failing tests for Codex CLI payload generation and lane selection.
- [ ] Implement summarizer wrapper with JSON schema output.
- [ ] Run summarizer tests.

### Task 3: Wire processed summary artifacts
- [ ] Add failing pipeline tests for `processed/*.txt`, `processed/*.json`, and `processed/*.md` outputs.
- [ ] Implement pipeline integration and concise artifact writers.
- [ ] Run pipeline tests.

### Task 4: Verify docs and CLI
- [ ] Update CLI/README usage for low-token analyst summaries.
- [ ] Run `PYTHONPATH=src ../.venv/bin/pytest tests -q` from `analysts/`.
- [ ] Smoke-test a non-live summarization path.
