# Positivity Event-Driven Alpha Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research runner for a signal-triggered positivity long-alpha event queue with near-high entries, ATR stops, sell signals, and N=1/3/5 queue-size summaries.

**Architecture:** Add a focused `backtesting.strategies.positivity_event` module for pure, testable event simulation. Keep data loading and output writing in `scripts/run_pos_event_alpha.py`, reusing existing positivity scoring and research performance helpers.

**Tech Stack:** Python, pandas, pytest, existing `ParquetStore`, existing `summarize_quintile_returns`, existing `_weighted_next_day_returns`.

---

### Task 1: Pure Event Simulator

**Files:**
- Create: `backtesting/strategies/positivity_event.py`
- Test: `tests/strategies/test_positivity_event.py`

- [ ] **Step 1: Write failing tests for entry, ATR stop, and queue capacity**

```python
from __future__ import annotations

import pandas as pd
import pytest

from backtesting.strategies.positivity_event import (
    EventQueueConfig,
    build_positivity_event_queue_strategy,
    true_range_atr,
)


def test_true_range_atr_uses_high_low_and_previous_close() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    high = pd.DataFrame({"A": [10.0, 13.0, 12.0]}, index=idx)
    low = pd.DataFrame({"A": [9.0, 11.0, 9.0]}, index=idx)
    close = pd.DataFrame({"A": [9.5, 12.0, 10.0]}, index=idx)

    atr = true_range_atr(high=high, low=low, close=close, lookback=2)

    assert pd.isna(atr.loc[idx[0], "A"])
    assert atr.loc[idx[1], "A"] == pytest.approx(2.25)
    assert atr.loc[idx[2], "A"] == pytest.approx(2.75)


def test_event_queue_enters_near_high_and_exits_on_atr_stop() -> None:
    idx = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    close = pd.DataFrame({"A": [100.0, 102.0, 104.0, 103.0, 96.0]}, index=idx)
    high = close.add(1.0)
    low = close.sub(1.0)
    membership = pd.DataFrame(True, index=idx, columns=close.columns)

    result = build_positivity_event_queue_strategy(
        close=close,
        high=high,
        low=low,
        membership=membership,
        config=EventQueueConfig(
            max_positions=1,
            positivity_lookback=2,
            min_periods=2,
            high_lookback=2,
            atr_lookback=2,
            atr_multiplier=2.0,
            relative_signal_groups=1,
            entry_high_ratio=0.95,
            exit_high_ratio=0.90,
        ),
    )

    assert result.weights.loc[idx[2], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[3], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[4], "A"] == pytest.approx(0.0)
    assert result.trades["exit_reason"].tolist() == ["atr_stop"]


def test_event_queue_replaces_weakest_active_name_when_score_margin_is_met() -> None:
    idx = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    )
    close = pd.DataFrame(
        {
            "A": [100.0, 101.0, 103.0, 103.0, 103.0],
            "B": [100.0, 101.0, 100.0, 103.0, 104.0],
        },
        index=idx,
    )
    high = close.add(1.0)
    low = close.sub(1.0)
    membership = pd.DataFrame(True, index=idx, columns=close.columns)
    bonus = pd.DataFrame(0.0, index=idx, columns=close.columns)
    bonus.loc[idx[3], "B"] = 5.0

    result = build_positivity_event_queue_strategy(
        close=close,
        high=high,
        low=low,
        membership=membership,
        score_bonus=bonus,
        config=EventQueueConfig(
            max_positions=1,
            positivity_lookback=2,
            min_periods=2,
            high_lookback=2,
            atr_lookback=2,
            atr_multiplier=3.0,
            relative_signal_groups=1,
            replacement_margin=1.0,
        ),
    )

    assert result.weights.loc[idx[2], "A"] == pytest.approx(1.0)
    assert result.weights.loc[idx[3], "A"] == pytest.approx(0.0)
    assert result.weights.loc[idx[3], "B"] == pytest.approx(1.0)
    assert result.trades["exit_reason"].tolist() == ["replacement"]
```

- [ ] **Step 2: Run the tests and confirm they fail because the module is missing**

Run: `uv run pytest tests/strategies/test_positivity_event.py -q`

Expected: FAIL with `ModuleNotFoundError` or missing symbols.

- [ ] **Step 3: Implement `positivity_event.py`**

Implement:

- `EventQueueConfig`
- `PositivityEventQueueResult`
- `true_range_atr`
- `build_positivity_event_queue_strategy`

The simulator must:

- use prior 252-day high analogs through `close.shift(1).rolling(...)`,
- allow near-high entries by `close / prior_high >= entry_high_ratio`,
- compute positivity via existing `positivity_score`,
- rank candidates by positivity percentile plus near-high ratio percentile plus optional score bonus,
- cap active names by `max_positions`,
- replace weakest active only when score margin is met,
- exit on ATR stop or near-high failure,
- emit equal-notional research weights, trade logs, and entry event logs.

- [ ] **Step 4: Run the new test file**

Run: `uv run pytest tests/strategies/test_positivity_event.py -q`

Expected: PASS.

### Task 2: Research Runner

**Files:**
- Create: `scripts/run_pos_event_alpha.py`
- Test: `tests/scripts/test_run_pos_event_alpha.py`

- [ ] **Step 1: Write failing tests for spec generation and summary ranking**

Test that:

- `build_event_strategy_specs()` includes queue sizes `1, 3, 5`,
- stop multipliers `2.0, 2.5, 3.0`,
- entry variants `near_high` and `breakout`,
- `rank_event_summary()` prefers viable positive-alpha rows with stronger robust score.

- [ ] **Step 2: Run the script tests and confirm failure**

Run: `uv run pytest tests/scripts/test_run_pos_event_alpha.py -q`

Expected: FAIL because script module is missing.

- [ ] **Step 3: Implement the runner**

Implement:

- `EventStrategySpec`
- `build_event_strategy_specs`
- `rank_event_summary`
- `run_event_alpha_grid`
- CLI `main`

The runner should load:

- `qw_adj_c`
- `qw_adj_h`
- `qw_adj_l`
- `qw_k200_yn`
- `qw_BM`

For each spec, run the pure event simulator, compute next-day research sleeve
returns, benchmark-relative active returns, split validation metrics, event
counts, holding days, turnover counts, and write:

- `event_alpha_summary.csv`
- `top10_event_alpha_summary.csv`
- `selected_event_alpha_strategy.json`
- `event_alpha_config.json`

- [ ] **Step 4: Run script tests**

Run: `uv run pytest tests/scripts/test_run_pos_event_alpha.py -q`

Expected: PASS.

### Task 3: End-to-End Verification

**Files:**
- Modify only if needed: documentation or imports.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
uv run pytest tests/strategies/test_positivity_event.py tests/scripts/test_run_pos_event_alpha.py -q
```

Expected: PASS.

- [ ] **Step 2: Run existing positivity regression tests**

Run:

```powershell
uv run pytest tests/strategies/test_positivity.py -q
```

Expected: PASS.

- [ ] **Step 3: Run a smoke research grid on a short real-data window**

Run:

```powershell
python scripts/run_pos_event_alpha.py --start 2024-01-01 --end 2024-12-31 --output results/pos_research/event_alpha_smoke_20260624
```

Expected: command exits 0 and writes the configured output files.

- [ ] **Step 4: Commit**

Commit with Lore trailers summarizing implementation scope, verification, and
known gaps.
