# MFBT EMP008 Origin Directional Warmup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Origin-style directional expected-alpha guards to all six MFBT factors and produce a 2019 year-end portfolio from 36 months of prior history.

**Architecture:** Keep the existing factor, regression, risk, optimizer, and backtest pipeline. Extend the existing `origin_small_cap` expected-alpha policy to all economically directed MFBT factors, and complete only the pre-official portion of the benchmark-weight panel with normalized float-market-cap proxy weights.

**Tech Stack:** Python, pandas, NumPy, SciPy SLSQP, pytest, parquet-backed EMP008 runners.

---

### Task 1: Lock the Origin directional alpha policy

**Files:**
- Modify: `tests/scripts/test_run_mfbt_emp008_full.py`
- Modify: `backtesting/strategies/emp008/mfbt_emp008.py`

- [ ] **Step 1: Replace the small-cap-only assertion with a failing directional-policy test**

```python
def test_mfbt_origin_small_cap_policy_preserves_origin_factor_directions() -> None:
    expected_alpha = pd.Series(
        {
            "price_momentum": -0.01,
            "earnings_momentum": -0.02,
            "dividend_yield": -0.03,
            "retail_flow": -0.04,
            "value": -0.05,
            "ln_market_cap": 0.06,
            "sector": 0.0,
        }
    )
    result = _apply_expected_alpha_policy(
        expected_alpha,
        MfbtEmp008Config(expected_alpha_policy="origin_small_cap"),
    )
    assert result.eq(0.0).all()
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `uv run pytest tests/scripts/test_run_mfbt_emp008_full.py::test_mfbt_origin_small_cap_policy_preserves_origin_factor_directions -q`

Expected: FAIL because the current policy retains negative values for the five positive-direction factors.

- [ ] **Step 3: Implement the minimal policy**

In `_apply_expected_alpha_policy`, copy the series, set negative values for
`price_momentum`, `earnings_momentum`, `dividend_yield`, `retail_flow`, and
`value` to zero, and set positive `ln_market_cap` to zero. Leave unknown and
sector entries untouched.

- [ ] **Step 4: Run the policy tests and verify GREEN**

Run: `uv run pytest tests/scripts/test_run_mfbt_emp008_full.py -q`

Expected: all EMP008 script tests pass.

### Task 2: Supply pre-2020 benchmark history without changing official rows

**Files:**
- Modify: `tests/scripts/test_run_mfbt_emp008_full.py`
- Modify: `backtesting/strategies/emp008/mfbt_emp008.py`

- [ ] **Step 1: Add a failing benchmark-history test**

```python
def test_complete_benchmark_history_uses_float_cap_only_before_official_weights() -> None:
    dates = pd.to_datetime(["2019-11-29", "2019-12-30", "2020-01-02"])
    bm = pd.DataFrame({"A": [np.nan, np.nan, 0.7], "B": [np.nan, np.nan, 0.3]}, index=dates)
    float_cap = pd.DataFrame({"A": [60.0, 80.0, 90.0], "B": [40.0, 20.0, 10.0]}, index=dates)
    universe = pd.DataFrame(True, index=dates, columns=["A", "B"])
    result = _complete_benchmark_history(bm, float_cap, universe)
    assert result.loc["2019-11-29"].tolist() == pytest.approx([0.6, 0.4])
    assert result.loc["2019-12-30"].tolist() == pytest.approx([0.8, 0.2])
    assert result.loc["2020-01-02"].tolist() == pytest.approx([0.7, 0.3])
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/scripts/test_run_mfbt_emp008_full.py::test_complete_benchmark_history_uses_float_cap_only_before_official_weights -q`

Expected: FAIL because `_complete_benchmark_history` does not exist.

- [ ] **Step 3: Implement benchmark completion and wire it into the runner**

Create `_complete_benchmark_history(bm_weights, float_mktcap, universe)` in
`mfbt_emp008.py`. Find the first row with a positive official total; fill only
earlier rows from `_normalized_weights`-equivalent float-cap normalization. Call
it after loading `bm_weights` and before preprocessing or monthly regression.

- [ ] **Step 4: Run the benchmark and EMP008 tests**

Run: `uv run pytest tests/scripts/test_run_mfbt_emp008_full.py tests/strategies/test_mfbt_emp008_experiments.py -q`

Expected: all tests pass.

### Task 3: Document and run the full experiment

**Files:**
- Modify: `backtesting/strategies/emp008/README.md`
- Generate: `results/emp008_runs/mfbt_emp008_wics_origin_directional_20191230_20260728/`

- [ ] **Step 1: Document the sign policy and warmup benchmark proxy**

Update the README expected-alpha and input sections to state that the five
positive-direction MFBT factors are floored at zero, size is capped at zero,
and pre-official benchmark history uses normalized KOSPI200 float-market-cap
weights.

- [ ] **Step 2: Run the full weights/backtest/report/comparison/attribution pipeline**

```powershell
uv run python scripts/run_mfbt_emp008_full.py `
  --name mfbt_emp008_wics_origin_directional_20191230_20260728 `
  --start 2019-12-01 `
  --end 2026-06-30 `
  --tracking-error-annual 0.007 `
  --risk-model factor_idio `
  --factor-set mfbt_origin_smallcap `
  --sector-neutral-dataset wics `
  --fill-mode close
```

Expected: `run_summary.json` plus weights, gross/costed backtests, report,
comparison, and factor-attribution artifacts.

- [ ] **Step 3: Audit the generated run**

Verify from `run_summary.json` and `weights/diagnostics.parquet` that the first
target date is 2019-12-30, the first backtest return after signal is 2020-01-02,
all solver rows succeeded, final weights sum to one, minimum weights are
nonnegative, sector residuals are within tolerance, and ex-ante TE does not
exceed the configured monthly limit.

- [ ] **Step 4: Run final verification**

```powershell
uv run pytest tests/scripts/test_run_mfbt_emp008_full.py tests/strategies/test_mfbt_emp008_experiments.py -q
uv run ruff check backtesting/strategies/emp008 tests/scripts/test_run_mfbt_emp008_full.py
```

Expected: both commands exit zero with no failures or lint errors introduced by
the change.
