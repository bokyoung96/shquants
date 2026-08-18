# EMP008 Optional Factor Timing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, lag-safe factor-momentum policy to EMP008 without changing default untimed behavior.

**Architecture:** A new `factor_timing.py` module owns timing configuration and decisions. `strategy.py` optionally calls it with factor returns strictly earlier than the rebalance date, applies returned weights through the existing factor-weight path, and writes long-form diagnostics. CLI runners expose only a `none|momentum` switch.

**Tech Stack:** Python 3.12, pandas, dataclasses, argparse, pytest

---

### Task 1: Factor-timing policy module

**Files:**
- Create: `backtesting/strategies/emp008/factor_timing.py`
- Create: `tests/strategies/test_emp008_factor_timing.py`

- [ ] **Step 1: Write failing unit tests**

Cover disabled passthrough, strong/neutral/weak states, LOW-direction reversal, insufficient-history fallback, normalization, and validation. The central test shape is:

```python
decision = decide_factor_timing(
    factor_returns=returns,
    factor_directions={"size": FactorDirection.LOW, "momentum": FactorDirection.HIGH},
    base_weights=pd.Series({"size": 0.4, "momentum": 0.6}),
    rebalance_date=pd.Timestamp("2024-01-31"),
    config=FactorTimingConfig(policy="momentum", fast_lookback=2, slow_lookback=3),
)
assert decision.weights.sum() == pytest.approx(1.0)
assert set(decision.diagnostics["state"]) == {"strong", "weak"}
assert decision.diagnostics["last_signal_date"].max() < pd.Timestamp("2024-01-31")
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/strategies/test_emp008_factor_timing.py -q`

Expected: import failure because `factor_timing.py` does not exist.

- [ ] **Step 3: Implement the minimal policy module**

Define immutable `FactorTimingConfig` and `FactorTimingDecision`, strict config/input validation, directional compounded returns, state mapping, normalized timed weights, and empty diagnostics for `config=None`.

```python
@dataclass(frozen=True, slots=True)
class FactorTimingConfig:
    policy: str = "momentum"
    fast_lookback: int = 6
    slow_lookback: int = 12
    strong_multiplier: float = 1.25
    neutral_multiplier: float = 1.0
    weak_multiplier: float = 0.75

@dataclass(frozen=True, slots=True)
class FactorTimingDecision:
    weights: pd.Series
    diagnostics: pd.DataFrame
```

- [ ] **Step 4: Run unit tests and confirm GREEN**

Run: `uv run pytest tests/strategies/test_emp008_factor_timing.py -q`

Expected: all tests pass.

### Task 2: Optional strategy integration and outputs

**Files:**
- Modify: `backtesting/strategies/emp008/data.py`
- Modify: `backtesting/strategies/emp008/strategy.py`
- Modify: `tests/strategies/test_emp008_factor_pipeline.py`

- [ ] **Step 1: Write failing integration tests**

Add tests proving `Emp008Config().factor_timing is None`, disabled runs preserve current result shape, enabled runs accumulate timing diagnostics, and `Emp008Result.write_outputs()` writes timing CSV/parquet only when diagnostics are non-empty.

- [ ] **Step 2: Run focused tests and confirm RED**

Run: `uv run pytest tests/strategies/test_emp008_factor_pipeline.py -q`

Expected: failures for missing config/result fields.

- [ ] **Step 3: Integrate timing without altering the disabled path**

Add `factor_timing: FactorTimingConfig | None = None` to `Emp008Config`. Extend `Emp008Result` with `factor_timing: pd.DataFrame` using a default empty frame for constructor compatibility. In `_optimize_month`, call the timing module only when enabled, use observations dated before `return_date`, and apply its weights through `apply_factor_weights`. Accumulate diagnostics in `run_emp008` and write timing files only when non-empty.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run: `uv run pytest tests/strategies/test_emp008_factor_pipeline.py tests/strategies/test_emp008_factor_weights.py tests/strategies/test_emp008_factor_timing.py -q`

Expected: all tests pass.

### Task 3: CLI and manifest contract

**Files:**
- Modify: `backtesting/strategies/emp008/run_weights.py`
- Modify: `backtesting/strategies/emp008/run_full.py`
- Modify: `tests/scripts/test_run_emp008_full.py`

- [ ] **Step 1: Write failing CLI/config tests**

Assert `build_emp008_config(factor_timing=None)` keeps timing disabled, `factor_timing="momentum"` creates default timing config, unknown policies fail, and parser defaults do not activate timing.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/scripts/test_run_emp008_full.py -q`

Expected: failures for the unsupported `factor_timing` argument.

- [ ] **Step 3: Add the optional switch and summary fields**

Add `--factor-timing {none,momentum}` to both runners, pass it to `build_emp008_config`, and record `factor_timing: "none"|"momentum"` plus timing artifact paths/counts in weight/run summaries.

- [ ] **Step 4: Run tests and confirm GREEN**

Run: `uv run pytest tests/scripts/test_run_emp008_full.py -q`

Expected: all tests pass.

### Task 4: Documentation and regression verification

**Files:**
- Modify: `backtesting/strategies/emp008/README.md`

- [ ] **Step 1: Document disabled and enabled examples**

Describe the optional stage, 6/12-month rule, factor-direction handling, lag contract, output schema, and CLI example:

```powershell
uv run python scripts/run_mfbt_emp008_weights.py --factor-set size_momentum_earnings_value --factor-timing momentum
```

- [ ] **Step 2: Run formatting and focused regression tests**

Run:

```powershell
uv run ruff check backtesting/strategies/emp008/factor_timing.py backtesting/strategies/emp008/data.py backtesting/strategies/emp008/strategy.py backtesting/strategies/emp008/run_weights.py backtesting/strategies/emp008/run_full.py tests/strategies/test_emp008_factor_timing.py tests/strategies/test_emp008_factor_pipeline.py tests/scripts/test_run_emp008_full.py
uv run pytest tests/strategies/test_emp008_factor_timing.py tests/strategies/test_emp008_factor_pipeline.py tests/strategies/test_emp008_factor_weights.py tests/scripts/test_run_emp008_full.py -q
```

Expected: Ruff reports no errors and all focused tests pass.

- [ ] **Step 3: Inspect the final diff**

Run: `git diff --check` and `git diff --stat`.

Expected: no whitespace errors; changes remain limited to the timing module, its integration, tests, and documentation.
