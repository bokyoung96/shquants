# EMP008 Size + Earnings Momentum Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent Size + operating-profit consensus momentum portfolio to the existing momentum comparison and regenerate all charts and Korean tables through 2026-06-30.

**Architecture:** Register one new `FactorSetId` that composes the existing `LN_MARKET_CAP` and `EARNINGS_MOMENTUM` definitions. Extend the existing momentum comparison profile, labels, and colors; reuse validated caches for all existing variants so only the new portfolio is computed.

**Tech Stack:** Python 3.12, pandas, matplotlib, openpyxl, pytest, Ruff, existing EMP008 backtest runner.

---

## File Structure

- Modify `backtesting/strategies/emp008/factor_registry.py`: register the new factor-set identity and exact factor membership.
- Modify `backtesting/strategies/emp008/experiments/size_value_measure_comparison.py`: add the new variant to the momentum profile, labels, and chart colors.
- Modify `tests/strategies/test_emp008_factor_registry.py`: lock factor membership, dataset dependency, ordering, and parser values.
- Modify `tests/strategies/test_emp008_size_value_measure_comparison.py`: lock momentum-profile ordering and output support.
- Regenerate `backtesting/strategies/emp008/tests/size_momentum_measure_comparison/`: update cached result package and all report artifacts.

### Task 1: Register the Size + earnings-momentum factor set

**Files:**
- Modify: `tests/strategies/test_emp008_factor_registry.py`
- Modify: `backtesting/strategies/emp008/factor_registry.py`

- [ ] **Step 1: Write the failing registry assertions**

Add `size_earnings_momentum` after `size_momentum_high` in enum/value-order assertions and add this case to the exact factor-set contract test:

```python
pytest.param(
    FactorSetId.SIZE_EARNINGS_MOMENTUM,
    (FactorId.LN_MARKET_CAP, FactorId.EARNINGS_MOMENTUM),
    (),
    (),
    0,
    (DatasetId.QW_OP_FWD_12M,),
    id="size-earnings-momentum",
)
```

- [ ] **Step 2: Run the registry test and confirm the red state**

Run: `uv run pytest tests/strategies/test_emp008_factor_registry.py -q`

Expected: failure because `FactorSetId.SIZE_EARNINGS_MOMENTUM` is not defined.

- [ ] **Step 3: Add the minimal registry definition**

Add:

```python
SIZE_EARNINGS_MOMENTUM = "size_earnings_momentum"
```

and:

```python
FactorSetId.SIZE_EARNINGS_MOMENTUM: FactorSetDefinition(
    id=FactorSetId.SIZE_EARNINGS_MOMENTUM,
    factors=(FactorId.LN_MARKET_CAP, FactorId.EARNINGS_MOMENTUM),
    constrain_expected_alpha_to_direction=True,
),
```

- [ ] **Step 4: Run the registry test and confirm green**

Run: `uv run pytest tests/strategies/test_emp008_factor_registry.py -q`

Expected: all registry tests pass.

### Task 2: Extend the momentum comparison profile

**Files:**
- Modify: `tests/strategies/test_emp008_size_value_measure_comparison.py`
- Modify: `backtesting/strategies/emp008/experiments/size_value_measure_comparison.py`

- [ ] **Step 1: Write failing profile assertions**

Extend the expected `MOMENTUM_VARIANTS` resolution with `FactorSetId.SIZE_EARNINGS_MOMENTUM` and assert the new display title path remains the momentum comparison.

- [ ] **Step 2: Run the comparison test and confirm the red state**

Run: `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py -q`

Expected: failure because the new variant is missing from `MOMENTUM_VARIANTS`.

- [ ] **Step 3: Add the variant, labels, and color**

Append `size_earnings_momentum` to `MOMENTUM_VARIANTS`. Add English label `Size + OP Consensus Momentum`, Korean label `사이즈 + 영업이익 컨센서스`, and distinct color `#DB2777`.

- [ ] **Step 4: Run comparison and combined tests**

Run:

```powershell
uv run pytest tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_size_value_measure_comparison.py -q
```

Expected: all selected tests pass.

### Task 3: Regenerate and verify the complete comparison package

**Files:**
- Regenerate: `backtesting/strategies/emp008/tests/size_momentum_measure_comparison/`

- [ ] **Step 1: Run the momentum comparison through the fixed end date**

Run:

```powershell
uv run python -m backtesting.strategies.emp008.experiments.size_value_measure_comparison --comparison-profile momentum --end 2026-06-30
```

Expected: existing four portfolio caches are reused; only `size_earnings_momentum` creates new weights and backtest metadata.

- [ ] **Step 2: Audit manifest, dates, costs, and tables**

Verify:

```python
assert manifest["end"] == "2026-06-30"
assert manifest["costs"] == {"fee": 0.0002, "sell_tax": 0.0015, "slippage": 0.0005}
assert manifest["variants"][-1] == {
    "factor_set": "size_earnings_momentum",
    "factors": ["ln_market_cap", "earnings_momentum"],
    "datasets": ["qw_op_fwd_12m"],
}
```

Confirm daily returns span 2020-01-31 through 2026-06-30 without missing values and all Korean tables contain the new row/column.

- [ ] **Step 3: Inspect all four images**

Open `cumulative_returns.png`, `cumulative_excess_returns.png`, `performance_dashboard.png`, and `yearly_excess_returns.png`. Confirm the magenta operating-profit consensus line, readable legends, correct title, and complete 2020-2026 panels.

- [ ] **Step 4: Run final verification**

Run:

```powershell
uv run pytest tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_size_value_measure_comparison.py -q
uv run ruff check backtesting/strategies/emp008/factor_registry.py backtesting/strategies/emp008/experiments/size_value_measure_comparison.py tests/strategies/test_emp008_factor_registry.py tests/strategies/test_emp008_size_value_measure_comparison.py
uv run python -m compileall -q backtesting/strategies/emp008/factor_registry.py backtesting/strategies/emp008/experiments/size_value_measure_comparison.py
```

Expected: tests and static checks exit successfully with no errors.
