# 52W 5M Breakout ATR Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a clean, canonical strategy implementation for the 52-week high plus 5-minute confirmed breakout signal with ATR touch stop, then produce a final report from the verified research artifacts.

**Architecture:** Put reusable signal and trade simulation logic in `backtesting/strategies/fifty_two_week_breakout_atr.py`. Put artifact/report generation in `scripts/build_52w_5m_breakout_atr_report.py`, reading the existing `5m_new_high_only` fixed-20 verified result so the final report reflects the already-audited backtest.

**Tech Stack:** Python, pandas, matplotlib, pytest, existing fixed-notional ledger artifacts under `results/flow_filtered_breakout_single/.../variants/5m_new_high_only`.

---

### Task 1: Strategy Signal And ATR Stop

**Files:**
- Create: `backtesting/strategies/fifty_two_week_breakout_atr.py`
- Create: `tests/strategies/test_fifty_two_week_breakout_atr.py`

- [ ] **Step 1: Write failing tests**

Add tests that verify:
- A breakout row is a signal only when the previous 5-minute close was not already above the prior 52-week high.
- Next 5-minute close confirmation is required.
- Entry occurs at the following 5-minute open after confirmation.
- ATR stop exits at the exact stop price after `min_holding_days`.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests\strategies\test_fifty_two_week_breakout_atr.py -q`
Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement minimal strategy**

Add:
- `BreakoutAtrConfig`
- `confirmed_breakout_entries`
- `simulate_atr_continuation`
- `run_breakout_atr_strategy`
- `FiftyTwoWeekBreakoutAtrStrategy`

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests\strategies\test_fifty_two_week_breakout_atr.py -q`
Expected: pass.

### Task 2: Final Report Builder

**Files:**
- Create: `scripts/build_52w_5m_breakout_atr_report.py`
- Create: `tests/scripts/test_build_52w_5m_breakout_atr_report.py`

- [ ] **Step 1: Write failing tests**

Add tests that verify a tiny selected-trades/ledger fixture produces:
- `report.md`
- `performance.png`
- core metrics containing final return, MDD, average trade bps, hit rate, and profit factor.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests\scripts\test_build_52w_5m_breakout_atr_report.py -q`
Expected: import failure because the report script does not exist.

- [ ] **Step 3: Implement report builder**

Read the verified `5m_new_high_only/fixed20` artifacts and write a clean final report under `research/52w_5m_breakout_atr_final`.

- [ ] **Step 4: Run GREEN and final build**

Run:
`uv run pytest tests\strategies\test_fifty_two_week_breakout_atr.py tests\scripts\test_build_52w_5m_breakout_atr_report.py -q`
`uv run python scripts\build_52w_5m_breakout_atr_report.py`

Expected: tests pass and the final report directory is printed.
