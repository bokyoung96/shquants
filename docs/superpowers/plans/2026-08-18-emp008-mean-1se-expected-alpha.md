# EMP008 Mean Minus One Standard Error Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `mean_1se` factor expected-alpha estimator and compare it with the existing 36-month arithmetic mean for equal-weight WI26 and WICS runs.

**Architecture:** Reuse the same trailing `risk_window=36` monthly factor returns. Compute each factor's sample mean and standard error `std(ddof=1) / sqrt(count)`, then shrink the mean toward zero by one standard error while preserving its sign. Keep factor construction, factor weights, covariance, tracking error, sector neutrality, optimizer, and costs unchanged.

**Tech Stack:** Python, pandas, pytest, Ruff, existing EMP008 backtest and reporting pipeline.

---

### Task 1: Lock the estimator behavior

**Files:**
- Modify: `tests/strategies/test_emp008_factor_weights.py`
- Modify: `tests/scripts/test_run_emp008_full.py`

- [ ] **Step 1: Add failing calculation tests**

Add a test using 36 observations and assert:

```python
mean = factor_returns.tail(36).mean()
standard_error = factor_returns.tail(36).std(ddof=1) / (36**0.5)
expected = mean.sign() * (mean.abs() - standard_error).clip(lower=0.0)
actual = compute_expected_alpha(
    factor_returns,
    alpha_factor_names=["size", "momentum"],
    sector_factor_names=["sector_tech"],
    window=36,
    estimator="mean_1se",
)
```

Verify positive and negative means both shrink toward zero, a mean smaller than its standard error becomes zero, and sector expected alpha remains zero.

- [ ] **Step 2: Add failing config and CLI tests**

Assert `mean_1se` is accepted by `Emp008Config`, `build_emp008_config`, `run_weights`, and `run_full`, while the default remains `mean`.

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/strategies/test_emp008_factor_weights.py tests/scripts/test_run_emp008_full.py -q -k "expected_alpha"
```

Expected: failures because `mean_1se` is not yet accepted.

### Task 2: Implement the estimator

**Files:**
- Modify: `backtesting/strategies/emp008/data.py`
- Modify: `backtesting/strategies/emp008/risk.py`
- Modify: `backtesting/strategies/emp008/run_weights.py`
- Modify: `backtesting/strategies/emp008/run_full.py`
- Modify: `backtesting/strategies/emp008/README.md`

- [ ] **Step 1: Extend allowed estimator values**

Add `mean_1se` alongside `mean` and `ewma36` in configuration validation and both CLI choices.

- [ ] **Step 2: Add direction-neutral shrinkage toward zero**

Implement:

```python
mean = recent.mean(axis=0).astype(float)
standard_error = recent.std(axis=0, ddof=1).div(recent.count().pow(0.5))
alpha = mean.sign().mul(mean.abs().sub(standard_error).clip(lower=0.0))
```

Continue forcing sector expected alpha to zero. The existing direction policy is applied afterward without change.

- [ ] **Step 3: Document exact semantics**

Document that `mean_1se` uses the sample standard deviation of the same trailing 36 monthly factor returns and subtracts the standard error of their mean, not one monthly standard deviation.

- [ ] **Step 4: Verify GREEN**

Run the focused test files and Ruff on touched production and test files.

### Task 3: Run and save the requested comparison

**Files:**
- Create outputs under: `backtesting/strategies/emp008/tests/expected_alpha_mean_1se_equal_weight/`

- [ ] **Step 1: Run WI26 and WICS in parallel**

Use `equal_25`, `factor_timing=none`, annual TE `0.007`, `factor_idio`, close fill, fee `0.0002`, sell tax `0.0015`, and slippage `0.0005` from `2019-12-30` through `2026-06-30`.

- [ ] **Step 2: Compare with the stored mean baseline**

Save CAGR, cumulative return, cumulative excess return, IR, MDD, turnover, yearly returns, daily returns, and cumulative paths. Cumulative excess return is only strategy cumulative return minus benchmark cumulative return.

- [ ] **Step 3: Save a Korean workbook and verify**

Create summary, yearly, daily, estimator-detail, and checks sheets; render every sheet; scan formula errors; verify workbook integrity and calculation identities.
