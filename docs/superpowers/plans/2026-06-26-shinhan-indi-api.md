# Shinhan iIndi API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clear `api/` package for Shinhan iIndi connection, real-time quotes, and stock orders.

**Architecture:** Keep iIndi-specific calls behind a tiny `Control` adapter, then expose one `Indi` client from `make()`. The client owns the pipeline and uses plain data objects for quotes and orders.

**Tech Stack:** Python 3.11, dataclasses, pytest, optional runtime `GiExpertControl`.

---

### Task 1: Lock Client Contract

**Files:**
- Create: `tests/api/test_indi.py`

- [x] Write failing tests for config loading, session reuse, login start, `SC` quote registration, and `SABA101U1` order field mapping.
- [x] Run `uv run python -m pytest tests/api/test_indi.py -q`.
- [x] Confirm failure is caused by missing `api` package.

### Task 2: Implement Package

**Files:**
- Create: `api/__init__.py`
- Create: `api/config.py`
- Create: `api/models.py`
- Create: `api/control.py`
- Create: `api/client.py`
- Create: `api/factory.py`
- Create: `api/config.example.json`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [x] Add `IndiConfig`, `Quote`, `Order`, and `OrderResult`.
- [x] Add `Control` adapter for direct `GiExpertControl` and QAx `dynamicCall`.
- [x] Add `Indi.connect()`, `subscribe_quote()`, `unsubscribe_quote()`, `buy()`, `sell()`, and `order()`.
- [x] Add `make()` and delayed runtime `GiExpertControl` loading.
- [x] Ignore `api/config.json` and include `api*` in package discovery.
- [x] Run `uv run python -m pytest tests/api/test_indi.py -q`.

### Task 3: Add Runnable Examples

**Files:**
- Create: `tests/api/test_scripts.py`
- Create: `api/connect.py`
- Create: `api/quotes.py`
- Create: `api/orders.py`

- [x] Write failing tests for quote formatting and order argument conversion.
- [x] Add `python -m api.connect`.
- [x] Add `python -m api.quotes`.
- [x] Add `python -m api.orders` with dry-run default and `--send` for live orders.
- [x] Run `uv run python -m pytest tests/api -q`.

### Task 4: Document Pipeline

**Files:**
- Create: `api/README.md`
- Create: `docs/superpowers/specs/2026-06-26-shinhan-indi-api-design.md`
- Create: `docs/superpowers/plans/2026-06-26-shinhan-indi-api.md`

- [x] Document config, connect, quote, order, and unsubscribe sequence.
- [x] Record design choices and test coverage.
- [x] Run final targeted tests and package import verification.
- [x] Fix review findings: unsubscribe-before-ack race, order validation, callback fail-fast behavior, and packaged API docs/config template.
