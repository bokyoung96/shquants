# KOSPI200 Signal Event Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a registered KOSPI200 signal-event strategy and a reproducible 500-candidate research runner.

**Architecture:** Reuse the existing composable strategy and RRG helper patterns. Keep the production strategy in `backtesting/strategies/signal_event_rotation.py`; keep broad candidate enumeration and reporting in `scripts/run_signal_event_rotation_grid.py`.

**Tech Stack:** Python, pandas, existing backtesting engine, pytest.

---

### Task 1: Lock Contracts With Tests

**Files:**
- Create: `tests/strategies/test_signal_event_rotation.py`
- Create: `tests/scripts/test_run_signal_event_rotation_grid.py`

- [ ] Write tests for event participation ramp, registry inclusion, dataset requirements, and 500 unique variants.
- [ ] Run targeted pytest and confirm the tests fail because the module does not exist.

### Task 2: Implement Strategy Module

**Files:**
- Create: `backtesting/strategies/signal_event_rotation.py`
- Modify: `backtesting/strategies/registry.py`
- Modify: `backtesting/strategies/README.md`

- [ ] Add signal producer with KOSPI200, price, OP/EPS estimates, benchmark, sector, market-cap, benchmark weights, and flow datasets.
- [ ] Add event masks and fixed participation ramp.
- [ ] Add sector-compressed construction.
- [ ] Register `signal_event_rotation`.

### Task 3: Implement 500-Candidate Runner

**Files:**
- Create: `scripts/run_signal_event_rotation_grid.py`

- [ ] Add deterministic variant grid with exactly 500 candidates.
- [ ] Reuse existing run artifacts when config matches.
- [ ] Write aggregate CSV/JSON and selected summary Markdown.

### Task 4: Verify and Research

**Files:**
- Modify after run: `docs/research/signal-event-rotation-grid-results.md`
- Create after run: `results/signal_event_research/*`

- [ ] Run targeted unit tests.
- [ ] Run the 500-candidate sweep with next-open execution and realistic costs.
- [ ] Select robust candidates by predeclared score.
- [ ] Run relevant broader tests.
- [ ] Commit with Lore protocol and push.
