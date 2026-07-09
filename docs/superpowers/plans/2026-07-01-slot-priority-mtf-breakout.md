# Slot Priority MTF Breakout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a conservative slot-priority comparison for the flow-confirmed 52-week high breakout strategy without adding fitted score weights.

**Architecture:** Add one research script that loads existing confirmed-breakout trade artifacts, attaches already-computed multi-timeframe features, selects trades with either chronological cap or simple tier priority, and writes metrics, ledgers, chart, and markdown report. Keep accounting at fixed 20-slot notional so max-15 variants leave cash instead of increasing per-position weight.

**Tech Stack:** Python, pandas, matplotlib, existing `scripts.verified_flow_backtest` portfolio accounting helpers, pytest.

---

### Task 1: Lock Slot Priority Selection Behavior

**Files:**
- Create: `tests/scripts/test_run_slot_priority_mtf_breakout.py`
- Create: `scripts/run_slot_priority_mtf_breakout.py`

- [ ] **Step 1: Write failing tests**

```python
def test_mtf_tier_priority_prefers_sector_and_vol_without_scores() -> None:
    trades = pd.DataFrame(
        {
            "ticker": ["T4", "T2", "T3", "T1"],
            "signal_time": pd.to_datetime(["2024-01-03 10:00"] * 4),
            "entry_time": pd.to_datetime(["2024-01-03 10:05"] * 4),
            "exit_time": pd.to_datetime(["2024-01-05 15:30"] * 4),
            "weekly_sector_rs_ok": [False, True, False, True],
            "daily_vol_compression_ok": [False, False, True, True],
            "net_return": [0.04, 0.02, 0.03, 0.01],
        }
    )

    selected, skipped = select_mtf_priority_fixed_slot_trades(trades, max_positions=2)

    assert selected["ticker"].tolist() == ["T1", "T2"]
    assert skipped["ticker"].tolist() == ["T3", "T4"]
```

- [ ] **Step 2: Run test and confirm RED**

Run: `uv run pytest tests\scripts\test_run_slot_priority_mtf_breakout.py -q`
Expected: import error because the script does not exist.

- [ ] **Step 3: Implement minimal selector**

Create `mtf_tier()` and `select_mtf_priority_fixed_slot_trades()` using only booleans and chronological tie-breaks.

- [ ] **Step 4: Run selector tests and confirm GREEN**

Run: `uv run pytest tests\scripts\test_run_slot_priority_mtf_breakout.py -q`
Expected: pass.

### Task 2: Lock Fixed 5% Accounting For Max-15

**Files:**
- Modify: `tests/scripts/test_run_slot_priority_mtf_breakout.py`
- Modify: `scripts/run_slot_priority_mtf_breakout.py`

- [ ] **Step 1: Write failing test**

```python
def test_selection_audit_uses_accounting_slots_separate_from_position_cap() -> None:
    close = pd.DataFrame({"A": [100.0, 110.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    trades = pd.DataFrame(
        {
            "ticker": ["A"],
            "signal_time": pd.to_datetime(["2024-01-02 10:00"]),
            "entry_time": pd.to_datetime(["2024-01-02 10:05"]),
            "exit_time": pd.to_datetime(["2024-01-03 15:30"]),
            "entry_price": [100.0],
            "net_return": [0.10],
        }
    )

    audit, selected, _skipped, fixed, _rebalanced = slot_selection_audit(
        trades,
        close,
        max_positions=15,
        accounting_slots=20,
        priority=False,
    )

    assert audit.slot_weight == 0.05
    assert selected["ticker"].tolist() == ["A"]
    assert fixed["equity"].iloc[-1] == 1.005
```

- [ ] **Step 2: Run test and confirm RED**

Run: `uv run pytest tests\scripts\test_run_slot_priority_mtf_breakout.py -q`
Expected: function missing.

- [ ] **Step 3: Implement audit wrapper**

Use `select_fixed_slot_trades()` or the new MTF priority selector for selection, but call `fixed_notional_mtm_ledger(..., slots=20)` and `rebalanced_mtm_ledger(..., slots=20)`.

- [ ] **Step 4: Run test and confirm GREEN**

Run: `uv run pytest tests\scripts\test_run_slot_priority_mtf_breakout.py -q`
Expected: pass.

### Task 3: Generate Comparison Artifacts

**Files:**
- Modify: `scripts/run_slot_priority_mtf_breakout.py`
- Modify: `tests/scripts/test_run_slot_priority_mtf_breakout.py`

- [ ] **Step 1: Write test for strategy set**

Assert `default_variants()` returns exactly `5m_only_max20`, `flow_confirmed_max20`, `flow_confirmed_max15`, `slot_priority_mtf_max15`, and `hard_filter_weekly_sector_daily_vol`.

- [ ] **Step 2: Implement experiment runner**

Load variants from `results/.../research/variants`, load `current_trades_with_mtf_features.csv`, attach MTF columns to flow-only trades, and write `slot_priority_mtf_metrics.csv`, ledgers, `slot_priority_mtf_comparison.png`, and `slot_priority_mtf_report.md`.

- [ ] **Step 3: Verify**

Run:
`uv run pytest tests\scripts\test_run_slot_priority_mtf_breakout.py -q`
`uv run python scripts\run_slot_priority_mtf_breakout.py`

Expected: tests pass and the output directory path is printed.
