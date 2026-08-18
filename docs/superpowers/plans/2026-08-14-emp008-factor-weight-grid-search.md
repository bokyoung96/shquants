# EMP008 Factor Weight Grid Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backtest positive, sum-to-100% weights for Size, 12-month momentum, operating-profit consensus momentum, and FCF/TEV, then save ranked and auditable results.

**Architecture:** Add one exact four-factor registry entry and a standalone experiment runner. The runner builds the prepared factor bundle once, executes each weight candidate with candidate-specific cache metadata, and writes full-period, yearly, subperiod, and daily-return outputs.

**Tech Stack:** Python, pandas, scipy optimizer, existing EMP008 backtest runner, openpyxl.

---

### Task 1: Register the four-factor portfolio

**Files:**
- Modify: `backtesting/strategies/emp008/factor_registry.py`
- Modify: `tests/strategies/test_emp008_factor_registry.py`

- [ ] Add `size_momentum_earnings_value` with `ln_market_cap`, `momentum_12m`, `earnings_momentum`, and `value`.
- [ ] Keep the existing direction constraint enabled.
- [ ] Run `uv run pytest tests/strategies/test_emp008_factor_registry.py -q` and confirm it passes.

### Task 2: Add the resumable grid-search runner

**Files:**
- Create: `backtesting/strategies/emp008/experiments/factor_weight_grid_search.py`
- Create: `tests/strategies/test_emp008_factor_weight_grid_search.py`

- [ ] Generate eight 10-point tilted combinations plus the 25/25/25/25 baseline.
- [ ] Reject non-positive weights and any combination whose total is not exactly 100%.
- [ ] Convert display percentages to relative EMP008 multipliers whose equal-weight baseline is 1.0.
- [ ] Include normalized weights in every candidate cache signature.
- [ ] Save candidate weights, diagnostics, returns, and metadata before continuing to the next candidate.
- [ ] Run the focused test file and confirm all cases pass.

### Task 3: Run and report the experiment

**Files:**
- Create: `backtesting/strategies/emp008/tests/factor_weight_grid_search/*`

- [ ] Run all candidates from 2019-12-30 through the latest common dataset date using annual TE 0.7%, fee 2 bp, sell tax 15 bp, and slippage 5 bp.
- [ ] Rank by annualized excess return, then information ratio.
- [ ] Save full-period summary, yearly returns, yearly relative excess returns, top-five subperiod robustness, and daily returns as CSV.
- [ ] Save the same result tables in one formatted Excel workbook.
- [ ] Verify all weights are positive, every row sums to 100%, all candidates have common return dates, and the workbook opens without formula errors.
