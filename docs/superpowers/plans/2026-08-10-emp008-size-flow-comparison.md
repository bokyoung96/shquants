# EMP008 Size and Flow Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Flow comparison profile that runs Size-only against Size + existing Retail Flow and produces the same verified report package as Value and Momentum.

**Architecture:** Extend the existing factor registry with one two-factor set and extend the existing comparison runner with one profile. Reuse the shared Size-only run, historical EMP008 series, report writers, metrics, and plotting functions; no new calculation pipeline or flow definition is introduced.

**Tech Stack:** Python, pandas, matplotlib, pytest, Ruff, the existing EMP008 optimizer and backtest runner.

---

### Task 1: Register Size + Retail Flow

**Files:**
- Modify: `backtesting/strategies/emp008/factor_registry.py`
- Test: `tests/strategies/test_emp008_factor_registry.py`

- [ ] **Step 1: Write the failing registry tests**

Add `size_retail_flow` to the expected enum list and assert that its definition contains exactly `ln_market_cap` and `retail_flow`, constrains expected alpha to direction, and resolves the `qw_retail` dataset.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/strategies/test_emp008_factor_registry.py -q`

Expected: FAIL because `FactorSetId.SIZE_RETAIL_FLOW` does not exist.

- [ ] **Step 3: Implement the factor set**

Add `SIZE_RETAIL_FLOW = "size_retail_flow"` to `FactorSetId` and this definition:

```python
FactorSetId.SIZE_RETAIL_FLOW: FactorSetDefinition(
    id=FactorSetId.SIZE_RETAIL_FLOW,
    factors=(FactorId.LN_MARKET_CAP, FactorId.RETAIL_FLOW),
    constrain_expected_alpha_to_direction=True,
),
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `uv run pytest tests/strategies/test_emp008_factor_registry.py -q`

Expected: all registry tests pass.

### Task 2: Add the Flow comparison profile

**Files:**
- Modify: `backtesting/strategies/emp008/experiments/size_value_measure_comparison.py`
- Test: `tests/strategies/test_emp008_size_value_measure_comparison.py`

- [ ] **Step 1: Write failing profile tests**

Assert that `FLOW_VARIANTS` equals `("size_only", "size_retail_flow")`, validation resolves both enum members, the label/color maps contain the new variant, and `--comparison-profile flow` selects the Flow output directory.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py -q`

Expected: FAIL because the Flow profile and constants do not exist.

- [ ] **Step 3: Implement the profile**

Add `FLOW_VARIANTS`, `DEFAULT_FLOW_OUTPUT_DIR`, display labels, colors, parser choice, profile dispatch, manifest/title handling, and export. Keep Value and Momentum selections unchanged. Reuse `DEFAULT_OUTPUT_DIR/size_only` when its metadata is compatible.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `uv run pytest tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_size_value_measure_comparison.py -q`

Expected: all focused tests pass.

### Task 3: Generate and verify the complete Flow report

**Files:**
- Create: `backtesting/strategies/emp008/tests/size_flow_measure_comparison/*`

- [ ] **Step 1: Run the Flow comparison**

Run:

```powershell
uv run python -m backtesting.strategies.emp008.experiments.size_value_measure_comparison --comparison-profile flow --end 2026-06-30
```

Expected: output contains `size_only` and `size_retail_flow`, starts at the 2019-12-30 zero baseline, realizes the first return on 2020-01-02, and ends on 2026-06-30.

- [ ] **Step 2: Audit artifacts and numerical contracts**

Verify 1,594 aligned daily rows, exact 20/15/5bp costs, six performance-table rows, years 2020 through 2026, and inclusion of existing/first/second EMP008 plus KOSPI200 BM. Verify the annual excess endpoint directly from compounded strategy and benchmark returns.

- [ ] **Step 3: Inspect all generated figures**

Open `cumulative_returns.png`, `cumulative_excess_returns.png`, `yearly_excess_returns.png`, and `performance_dashboard.png`. Confirm complete legends, readable labels, no clipping, and annual zero baselines immediately before each first trading day.

- [ ] **Step 4: Run final verification**

Run:

```powershell
uv run pytest tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_size_value_measure_comparison.py -q
uv run ruff check backtesting/strategies/emp008/factor_registry.py backtesting/strategies/emp008/experiments/size_value_measure_comparison.py tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_size_value_measure_comparison.py
```

Expected: all tests and Ruff checks pass.
