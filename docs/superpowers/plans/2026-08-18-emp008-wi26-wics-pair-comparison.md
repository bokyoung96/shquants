# EMP008 WI26-WICS Pair Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible paired WI26-WICS cumulative excess-gap subplots and remove the redundant unsuffixed WI26 artifact directory safely.

**Architecture:** A standalone report module reads the two completed grid-search return panels and pairs common candidate columns. Pure transformation functions build full-period and annual-reset gap frames; plotting and CLI orchestration remain thin wrappers.

**Tech Stack:** Python, pandas, matplotlib, pytest, JSON/CSV artifacts.

---

### Task 1: Lock the paired-gap calculation

**Files:**
- Create: `tests/strategies/test_emp008_factor_weight_grid_comparison.py`
- Create: `backtesting/strategies/emp008/reports/factor_weight_grid_comparison.py`

- [ ] Write tests proving that common candidates are paired, positive gaps favor WICS, annual frames reset to zero, and unequal benchmark series are rejected.
- [ ] Run `pytest tests/strategies/test_emp008_factor_weight_grid_comparison.py -q` and confirm the tests fail because the report module does not exist.
- [ ] Implement loaders and pure frame-building functions with explicit input validation.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Render and export the comparison

**Files:**
- Modify: `backtesting/strategies/emp008/reports/factor_weight_grid_comparison.py`

- [ ] Add the 3x3 candidate-pair plot, 4x2 annual-reset plot, year-end CSV, and JSON manifest.
- [ ] Run the report with the saved WI26 and WICS `daily_returns.csv` files.
- [ ] Assert from the CSV that all nine 2023 year-end gaps are negative.
- [ ] Inspect both PNGs for readable labels, zero reference lines, year separators, and complete panels.

### Task 3: Retire the redundant unsuffixed artifact directory

**Files:**
- Modify: text artifacts under `backtesting/strategies/emp008/tests/factor_weight_grid_search_wi26/`
- Remove: `backtesting/strategies/emp008/tests/factor_weight_grid_search/`

- [ ] Verify every file in the unsuffixed directory has an identical relative-path/hash counterpart in the WI26 directory.
- [ ] Replace stale embedded unsuffixed paths in WI26 text artifacts with `factor_weight_grid_search_wi26`.
- [ ] Verify no WI26 text artifact still references the unsuffixed directory.
- [ ] Remove only the resolved unsuffixed directory and verify WI26/WICS/report outputs remain present.
- [ ] Run the focused tests and report generation again.
