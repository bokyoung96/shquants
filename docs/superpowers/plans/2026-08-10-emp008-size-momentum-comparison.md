# EMP008 Size and Momentum Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three Size-plus-momentum EMP008 portfolios and generate a separate result package identical to the Size–Value comparison.

**Architecture:** Register three focused factor sets, then parameterize the existing comparison runner with Value and Momentum profiles. Reuse the same backtest, KOSPI200-relative calculations, plots, Korean yearly subplots, and annual tables while keeping output directories isolated.

**Tech Stack:** Python, pandas, matplotlib, pytest, Ruff, existing EMP008 runner.

---

### Task 1: Register three Size-plus-momentum factor sets

**Files:**
- Modify: `backtesting/strategies/emp008/factor_registry.py`
- Modify: `tests/strategies/test_emp008_factor_registry.py`

- [ ] Add failing tests for `size_momentum_12m`, `size_momentum_12_1m`, and `size_momentum_high` with exact factor tuples `(ln_market_cap, momentum)` and direction-constrained alpha.
- [ ] Run `uv run pytest tests/strategies/test_emp008_factor_registry.py -q` and confirm failure for missing enum members.
- [ ] Add the three enum members and immutable `FactorSetDefinition` entries:

```python
FactorSetDefinition(
    id=FactorSetId.SIZE_MOMENTUM_12M,
    factors=(FactorId.LN_MARKET_CAP, FactorId.MOMENTUM_12M),
    constrain_expected_alpha_to_direction=True,
)
```

Repeat with `MOMENTUM_12_1M` and `PRICE_TO_252D_HIGH`.
- [ ] Run the registry tests and confirm all pass.

### Task 2: Parameterize the comparison runner

**Files:**
- Modify: `backtesting/strategies/emp008/experiments/size_value_measure_comparison.py`
- Modify: `tests/strategies/test_emp008_size_value_measure_comparison.py`

- [ ] Add failing tests proving the Value profile remains the default and the Momentum profile resolves:

```python
MOMENTUM_VARIANTS = (
    "size_only",
    "size_momentum_12m",
    "size_momentum_12_1m",
    "size_momentum_high",
)
```

- [ ] Extend supported labels, colors, Korean yearly labels, validation, titles, and CLI choices without changing existing Value outputs.
- [ ] Add `--comparison-profile momentum`; default its output directory to `backtesting/strategies/emp008/tests/size_momentum_measure_comparison` and select the four Momentum variants.
- [ ] Ensure the manifest and all generated tables/images use the selected profile.
- [ ] Run `uv run pytest tests/strategies/test_emp008_size_value_measure_comparison.py -q` and confirm all pass.

### Task 3: Run and verify the Momentum result package

**Files:**
- Generate: `backtesting/strategies/emp008/tests/size_momentum_measure_comparison/**`

- [ ] Run:

```powershell
uv run python -m backtesting.strategies.emp008.experiments.size_value_measure_comparison --comparison-profile momentum --end 2026-06-30
```

- [ ] Verify the manifest records four Momentum-profile portfolios, KOSPI200, end date `2026-06-30`, fee `0.0002`, sell tax `0.0015`, and slippage `0.0005`.
- [ ] Run focused registry/comparison tests and Ruff.
- [ ] Inspect cumulative returns, cumulative excess, dashboard, and Korean yearly subplot images.
- [ ] Verify annual CSV/Excel tables cover 2020–2026 and match subplot endpoints.
