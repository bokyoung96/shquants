# Confirmed Breakout Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a baseline versus confirmed-breakout entry-compression comparison for the 5-minute KOSPI200 flow-filtered breakout strategy.

**Architecture:** Keep the existing baseline reproducible. Add a narrow entry-confirmation mode to the flow-filtered single-strategy runner, then add score-free comparison metrics for entry reduction, profit factor, concurrency, yearly entries, and right-tail preservation.

**Tech Stack:** Python, pandas, pytest, existing `scripts/run_flow_filtered_breakout_single.py` and `scripts/compare_flow_filtered_breakout_performance.py`.

---

### Task 1: Confirmed Breakout Entry Selection

**Files:**
- Modify: `tests/scripts/test_run_flow_filtered_breakout_single.py`
- Modify: `scripts/run_flow_filtered_breakout_single.py`
- Modify: `scripts/run_tech_gamma_long_only.py`

- [ ] **Step 1: Write failing tests**

Add tests that prove `_entry_candidates()` keeps baseline behavior for `first_close`, waits one extra 5-minute close for `next_close_confirmed`, and suppresses repeated same-ticker entries after `compress_breakout_episodes()`.

- [ ] **Step 2: Run tests to verify red**

Run: `uv run pytest tests/scripts/test_run_flow_filtered_breakout_single.py -q`

Expected: fail because `entry_confirmation` and episode compression helpers do not exist.

- [ ] **Step 3: Implement minimal behavior**

Add `entry_confirmation: str = "first_close"` and `episode_compression: bool = False` to `TechGammaConfig`. In `_entry_candidates()`, if `entry_confirmation == "next_close_confirmed"`, require the next 5-minute close to remain above the same prior 52-week high and enter at the following 5-minute open.

- [ ] **Step 4: Run tests to verify green**

Run: `uv run pytest tests/scripts/test_run_flow_filtered_breakout_single.py -q`

Expected: pass.

### Task 2: Comparison Metrics

**Files:**
- Modify: `tests/scripts/test_compare_flow_filtered_breakout_performance.py`
- Modify: `scripts/compare_flow_filtered_breakout_performance.py`

- [ ] **Step 1: Write failing comparison-metric test**

Add expectations for `entry_reduction_vs_baseline`, `profit_factor`, holding days, concurrent positions, yearly entry counts, and right-tail metrics.

- [ ] **Step 2: Run tests to verify red**

Run: `uv run pytest tests/scripts/test_compare_flow_filtered_breakout_performance.py -q`

Expected: fail because the extra metrics are not emitted.

- [ ] **Step 3: Implement minimal metrics**

Extend `write_comparison_outputs()` to write `comparison_metrics.csv`, `yearly_entries.csv`, and `right_tail_preservation.csv` with the required fields.

- [ ] **Step 4: Run tests to verify green**

Run: `uv run pytest tests/scripts/test_compare_flow_filtered_breakout_performance.py -q`

Expected: pass.

### Task 3: Run Backtests

**Files:**
- No production file changes expected after Task 2
- Outputs under `results/flow_filtered_breakout_single/`

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/scripts/test_run_flow_filtered_breakout_single.py tests/scripts/test_compare_flow_filtered_breakout_performance.py -q`

Expected: pass.

- [ ] **Step 2: Run confirmed breakout backtest**

Run the existing baseline config with `entry_confirmation = "next_close_confirmed"` and output to `results/flow_filtered_breakout_single/sector_pos90_margin002_flow_or_60d_2019start_confirmed`.

- [ ] **Step 3: Run confirmed breakout with episode compression**

Run the same config with `episode_compression = true` and output to `results/flow_filtered_breakout_single/sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode`.

- [ ] **Step 4: Write comparison outputs**

Compare baseline, confirmed, and confirmed_episode into `results/flow_filtered_breakout_single/comparison_2019_baseline_vs_confirmed`.

- [ ] **Step 5: Report results**

Summarize entry reduction, average net bps, profit factor, MDD, final equity, concurrency, and right-tail preservation.

