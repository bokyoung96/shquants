# EMP008 EWMA(36) Expected Alpha Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the existing 36-month simple-mean factor expected alpha with an optional EWMA estimator using `span=36`, without changing factor weights, risk estimation, or optimizer constraints.

**Architecture:** Keep monthly cross-sectional factor-return estimation unchanged. Add an estimator switch at `compute_expected_alpha`, pass it from `Emp008Config`, and expose only `mean` and `ewma36` through the existing EMP008 CLIs. The default remains `mean`, so existing runs are byte-for-byte behavior compatible.

**Tech Stack:** Python, pandas, pytest, Ruff, existing EMP008 backtest/reporting pipeline.

---

### Task 1: Lock expected-alpha behavior with tests

**Files:**
- Modify: `tests/strategies/test_emp008_factor_weights.py`
- Modify: `tests/scripts/test_run_emp008_full.py`

- [ ] **Step 1: Add failing estimator tests**

Add tests that assert:

```python
expected = factor_returns.tail(36).ewm(span=36, adjust=True).mean().iloc[-1]
actual = compute_expected_alpha(
    factor_returns,
    alpha_factor_names=["size", "momentum"],
    sector_factor_names=["sector_tech"],
    window=36,
    estimator="ewma36",
)
pd.testing.assert_series_equal(actual.loc[["size", "momentum"]], expected.loc[["size", "momentum"]])
assert actual.loc["sector_tech"] == 0.0
```

Also assert that the default estimator still equals the arithmetic mean, invalid estimator values fail, both parsers default to `mean`, and `--expected-alpha-estimator ewma36` maps into `Emp008Config`.

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
python -m pytest tests/strategies/test_emp008_factor_weights.py tests/scripts/test_run_emp008_full.py -q
```

Expected: failures because `estimator` and the CLI option do not exist yet.

### Task 2: Implement the optional EWMA(36) estimator

**Files:**
- Modify: `backtesting/strategies/emp008/data.py`
- Modify: `backtesting/strategies/emp008/risk.py`
- Modify: `backtesting/strategies/emp008/strategy.py`
- Modify: `backtesting/strategies/emp008/run_weights.py`
- Modify: `backtesting/strategies/emp008/run_full.py`
- Modify: `backtesting/strategies/emp008/README.md`

- [ ] **Step 1: Add validated configuration**

Add `expected_alpha_estimator: str = "mean"` to `Emp008Config` and validate membership in `{"mean", "ewma36"}` in `__post_init__`.

- [ ] **Step 2: Add the estimator branch**

Update `compute_expected_alpha` so the existing branch remains:

```python
alpha = recent.mean(axis=0).astype(float)
```

and the optional branch is:

```python
alpha = recent.ewm(span=36, adjust=True).mean().iloc[-1].astype(float)
```

Use only `tail(window)` observations and continue forcing sector expected alpha to zero.

- [ ] **Step 3: Wire config and CLI**

Pass `config.expected_alpha_estimator` from `strategy.py`. Add shared CLI option:

```python
parser.add_argument(
    "--expected-alpha-estimator",
    choices=("mean", "ewma36"),
    default="mean",
)
```

Record the selected estimator in run summaries and logs. Do not modify `factor_timing`, covariance estimation, tracking error, or costs.

- [ ] **Step 4: Document exact semantics**

Document that `ewma36` means pandas `ewm(span=36, adjust=True)` applied to the same trailing 36 monthly factor returns used by the baseline.

- [ ] **Step 5: Run focused verification**

Run:

```powershell
python -m pytest tests/strategies/test_emp008_factor_weights.py tests/scripts/test_run_emp008_full.py -q
python -m ruff check backtesting/strategies/emp008/data.py backtesting/strategies/emp008/risk.py backtesting/strategies/emp008/strategy.py backtesting/strategies/emp008/run_weights.py backtesting/strategies/emp008/run_full.py tests/strategies/test_emp008_factor_weights.py tests/scripts/test_run_emp008_full.py
```

Expected: all tests pass and Ruff reports no errors.

### Task 3: Run the single requested comparison

**Files:**
- Create outputs under: `backtesting/strategies/emp008/tests/expected_alpha_ewma36_equal_weight/`

- [ ] **Step 1: Generate EWMA weights**

Use the same period, factor set, equal factor weights, annual TE, WI26/WICS neutrality datasets, and data cutoff as the existing equal-weight baseline. Set factor timing to `none` and expected-alpha estimator to `ewma36`.

- [ ] **Step 2: Run costed WI26 and WICS backtests**

Use fee `0.0002`, sell tax `0.0015`, slippage `0.0005`, and close fill, matching the stored baseline.

- [ ] **Step 3: Save the comparison**

Save performance metrics, yearly returns, yearly excess returns, daily returns, cumulative comparison figures, and a Korean Excel summary. Calculate cumulative excess return only as strategy cumulative return minus benchmark cumulative return.

- [ ] **Step 4: Verify artifacts and decide**

Verify date alignment, row counts, return identities, and workbook formulas. Report whether EWMA improves CAGR, cumulative excess return, IR, and MDD for WI26 and WICS, and whether any improvement is concentrated in one year.
