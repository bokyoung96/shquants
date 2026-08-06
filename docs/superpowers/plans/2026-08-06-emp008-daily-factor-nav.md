# EMP008 Daily Factor NAV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plot every monthly-rebalanced factor quintile with its realized daily adjusted-close NAV path.

**Architecture:** Preserve all existing monthly diagnostic tables and summaries. Add one daily path builder that values the already-recorded monthly quantile weights as fixed-share holdings, validates that every month-end endpoint matches the monthly cumulative table, and feeds the existing PNG renderer.

**Tech Stack:** Python, pandas, NumPy, Matplotlib, pytest, PyArrow

---

### Task 1: Daily fixed-share NAV contract

**Files:**
- Modify: `tests/strategies/test_emp008_factor_quantiles.py`
- Modify: `backtesting/strategies/emp008/factor_quantiles.py`

- [ ] **Step 1: Write the failing daily-path test**

Add a fixture with one signal date, at least one intermediate trading date, and one return date. Assert that `evaluate_factor_quantiles(...).daily_cumulative_returns` contains the intermediate date and that each Q portfolio equals the weighted price-relative NAV, not an interpolated monthly value.

- [ ] **Step 2: Run the focused test to verify RED**

Run: `pytest tests/strategies/test_emp008_factor_quantiles.py -k "daily_cumulative" -q`

Expected: FAIL because `Emp008FactorQuantileResult` has no `daily_cumulative_returns` field.

- [ ] **Step 3: Implement the minimal daily path builder**

Add `_build_daily_cumulative_returns(close, monthly_returns, portfolio_weights, directions, q)` and a `daily_cumulative_returns` result field. For every `(signal_date, return_date, factor, weighting, quantile)` weight vector, value fixed shares with:

```python
period_nav = close.loc[signal_date:return_date, tickers].ffill().divide(
    close.loc[signal_date, tickers]
).mul(weights, axis="columns").sum(axis="columns")
```

Chain Q1-Q5 across months. Derive `high_minus_low` from `high_nav - low_nav` and reverse it for low-direction `preferred_minus_avoided`. Emit the initial signal-date baseline once and subsequent actual close dates through each return date.

- [ ] **Step 4: Add endpoint and edge-case tests**

Assert:

- every daily month-end cumulative value equals `_build_cumulative_returns(monthly_returns)`;
- signal dates are not duplicated between adjacent holding periods;
- intermediate missing prices are forward-filled only after a valid signal price;
- daily rows have unique `(date, factor, weighting, portfolio)` keys and finite cumulative returns.

- [ ] **Step 5: Run focused tests to verify GREEN**

Run: `pytest tests/strategies/test_emp008_factor_quantiles.py -k "daily_cumulative or evaluate_factor_quantiles" -q`

Expected: PASS.

### Task 2: Daily artifacts and plot wiring

**Files:**
- Modify: `tests/strategies/test_emp008_factor_quantiles.py`
- Modify: `tests/scripts/test_run_emp008_factor_quantiles.py`
- Modify: `backtesting/strategies/emp008/factor_quantiles.py`
- Modify: `backtesting/strategies/emp008/README.md`

- [ ] **Step 1: Write failing output tests**

Require `daily_cumulative_returns.csv` and `daily_cumulative_returns.parquet` in the payload, manifest, atomic artifact list, and real CLI output. Assert PNG plotting receives the daily `date` series and therefore contains more observations than the monthly cumulative table in a multi-date fixture.

- [ ] **Step 2: Run output tests to verify RED**

Run: `pytest tests/strategies/test_emp008_factor_quantiles.py tests/scripts/test_run_emp008_factor_quantiles.py -q`

Expected: FAIL on missing daily artifact keys/files.

- [ ] **Step 3: Wire daily output and plotting**

Write both daily files in the existing staging directory, include them in atomic publication, payload, validation, and manifest, and render both existing PNG filenames from `daily_cumulative_returns` using `date` as the x-axis. Add manifest fields:

```python
"rebalance_frequency": "monthly",
"nav_frequency": "daily",
```

Document the daily NAV semantics and artifact names in the README without changing the monthly summary description.

- [ ] **Step 4: Run tests and lint to verify GREEN**

Run: `pytest tests/strategies/test_emp008_factor_quantiles.py tests/scripts/test_run_emp008_factor_quantiles.py tests/scripts/test_run_emp008_full.py -q`

Run: `ruff check backtesting/strategies/emp008/factor_quantiles.py tests/strategies/test_emp008_factor_quantiles.py tests/scripts/test_run_emp008_factor_quantiles.py backtesting/strategies/emp008/README.md`

Expected: all tests and Python lint checks pass; if Ruff does not accept Markdown, rerun it on the three Python files only.

### Task 3: Full-data regeneration and integration

**Files:**
- Regenerate ignored artifacts under: `results/emp008_factor_quantiles/all_factors/`

- [ ] **Step 1: Run the full daily diagnostic**

Run:

```powershell
python -m backtesting.strategies.emp008.run_factor_quantiles `
  --factor-set all_factors --quantiles 5 `
  --start 2020-01-31 --end 2026-06-30 `
  --parquet-dir C:\Users\CHECK\Documents\GitHub\shquants\parquet `
  --output-dir results\emp008_factor_quantiles\all_factors
```

Expected: daily artifact paths and a positive `daily_cumulative_returns_rows` count in the JSON payload.

- [ ] **Step 2: Verify endpoint identity and inspect images**

Compare daily rows on each `return_date` against `cumulative_returns.csv` for every factor, weighting, and portfolio with numerical tolerance `1e-10`. Open both PNGs and confirm ten populated subplots, Q1-Q5 plus the direction-aware spread, daily line density, and no title/legend collision.

- [ ] **Step 3: Run final focused verification**

Run: `pytest tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_factor_quantiles.py tests/scripts/test_run_emp008_full.py tests/scripts/test_run_emp008_factor_quantiles.py -q`

Run Ruff on all changed Python files and `git diff --check`.

- [ ] **Step 4: Commit, push, and integrate**

Commit with Lore trailers, push the feature branch and fast-forward remote `main` when possible, then merge remote `main` into the dirty local `main` without modifying unrelated user files.
