# RRG Sector Rotation Concentration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add quantile, minimum-revision, and max-name concentration controls to `rrg_sector_rotation` while preserving current default behavior.

**Architecture:** Keep the existing signal producer unchanged. Add concentration parameters to `RrgSectorRotation` and `_RrgLongShortRankProportionalWeight`, then filter per-date candidate scores inside the construction rule before existing rank-proportional weighting. This keeps RRG state and OP confirmation semantics intact.

**Tech Stack:** Python dataclasses, pandas, pytest, existing `backtesting.run` CLI/spec machinery.

---

## File Structure

- Modify `backtesting/strategies/rrg_sector_rotation.py`: add parameters, validation helpers, and score filtering.
- Modify `tests/strategies/test_registry.py`: add regression tests for defaults, quantile/min filters, max caps, and validation.
- Modify `backtesting/strategies/README.md`: document the concentration controls and suggested personal-account experiment parameters.

---

### Task 1: Lock Current Default Behavior

**Files:**
- Test: `tests/strategies/test_registry.py`

- [ ] **Step 1: Write the failing/passing regression test**

Add this test near the existing RRG tests:

```python
def test_rrg_sector_rotation_default_parameters_keep_all_confirmed_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backtesting.strategies import rrg_sector_rotation as module

    index = pd.date_range("2024-01-02", periods=60, freq="D")
    columns = ["L1", "L2", "L3", "S1", "S2"]
    close = pd.DataFrame({symbol: np.linspace(100.0, 120.0, len(index)) for symbol in columns}, index=index)
    k200 = pd.DataFrame(True, index=index, columns=columns)
    sector = pd.DataFrame(
        {"L1": "Long", "L2": "Long", "L3": "Long", "S1": "Short", "S2": "Short"},
        index=index,
    )
    market_cap = pd.DataFrame(100.0, index=index, columns=columns)
    benchmark = pd.DataFrame({"IKS200": np.linspace(100.0, 115.0, len(index))}, index=index)
    empty = pd.DataFrame(np.nan, index=index, columns=columns)
    market = MarketData(
        frames={
            "close": close,
            "k200_yn": k200,
            "sector_big": sector,
            "market_cap": market_cap,
            "benchmark": benchmark,
            "op_fwd_q1": empty,
            "op_fwd_q2": empty,
            "op_fwd": empty,
        },
        universe=None,
        benchmark=None,
    )

    state = pd.DataFrame({"Long": "Leading", "Short": "Lagging"}, index=index)
    stock_op = pd.DataFrame({"L1": 0.40, "L2": 0.20, "L3": 0.01, "S1": -0.40, "S2": -0.01}, index=index)
    sector_op = pd.DataFrame({"Long": 0.25, "Short": -0.25}, index=index)

    monkeypatch.setattr(module, "_build_rrg_context", lambda **_: (state, state.eq("Leading"), state.eq("Lagging")))
    monkeypatch.setattr(module, "_build_stock_op_revision", lambda **_: stock_op)
    monkeypatch.setattr(module, "_build_sector_op_revision", lambda **_: sector_op)

    strategy = build_strategy("rrg_sector_rotation")
    last = strategy.build_weights(market).iloc[-1]

    assert set(last[last.gt(0.0)].index) == {"L1", "L2", "L3"}
    assert set(last[last.lt(0.0)].index) == {"S1", "S2"}
```

- [ ] **Step 2: Run test**

Run: `uv run pytest tests/strategies/test_registry.py::test_rrg_sector_rotation_default_parameters_keep_all_confirmed_candidates -q`

Expected: PASS before implementation, proving default behavior is already broad and must remain so.

---

### Task 2: Add Concentration Filter Tests

**Files:**
- Test: `tests/strategies/test_registry.py`

- [ ] **Step 1: Add quantile/min/cap behavior test**

Add this test after the default-behavior test:

```python
def test_rrg_sector_rotation_quantile_thresholds_and_caps_reduce_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backtesting.strategies import rrg_sector_rotation as module

    index = pd.date_range("2024-01-02", periods=60, freq="D")
    columns = ["L1", "L2", "L3", "L4", "S1", "S2", "S3"]
    close = pd.DataFrame({symbol: np.linspace(100.0, 120.0, len(index)) for symbol in columns}, index=index)
    k200 = pd.DataFrame(True, index=index, columns=columns)
    sector = pd.DataFrame(
        {
            "L1": "Long",
            "L2": "Long",
            "L3": "Long",
            "L4": "Long",
            "S1": "Short",
            "S2": "Short",
            "S3": "Short",
        },
        index=index,
    )
    market_cap = pd.DataFrame(100.0, index=index, columns=columns)
    benchmark = pd.DataFrame({"IKS200": np.linspace(100.0, 115.0, len(index))}, index=index)
    empty = pd.DataFrame(np.nan, index=index, columns=columns)
    market = MarketData(
        frames={
            "close": close,
            "k200_yn": k200,
            "sector_big": sector,
            "market_cap": market_cap,
            "benchmark": benchmark,
            "op_fwd_q1": empty,
            "op_fwd_q2": empty,
            "op_fwd": empty,
        },
        universe=None,
        benchmark=None,
    )

    state = pd.DataFrame({"Long": "Leading", "Short": "Lagging"}, index=index)
    stock_op = pd.DataFrame(
        {"L1": 0.50, "L2": 0.30, "L3": 0.10, "L4": 0.02, "S1": -0.60, "S2": -0.25, "S3": -0.04},
        index=index,
    )
    sector_op = pd.DataFrame({"Long": 0.25, "Short": -0.25}, index=index)

    monkeypatch.setattr(module, "_build_rrg_context", lambda **_: (state, state.eq("Leading"), state.eq("Lagging")))
    monkeypatch.setattr(module, "_build_stock_op_revision", lambda **_: stock_op)
    monkeypatch.setattr(module, "_build_sector_op_revision", lambda **_: sector_op)

    strategy = build_strategy(
        "rrg_sector_rotation",
        long_quantile=0.50,
        short_quantile=0.50,
        min_long_revision=0.05,
        min_short_revision=0.05,
        max_long_names=2,
        max_short_names=1,
    )
    last = strategy.build_weights(market).iloc[-1]

    assert set(last[last.gt(0.0)].index) == {"L1", "L2"}
    assert set(last[last.lt(0.0)].index) == {"S1"}
    assert last[last.gt(0.0)].sum() == pytest.approx(1.0)
    assert last[last.lt(0.0)].sum() == pytest.approx(-0.5)
```

- [ ] **Step 2: Add validation test**

Add:

```python
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"long_quantile": -0.1}, "long_quantile must be between 0 and 1"),
        ({"long_quantile": 1.1}, "long_quantile must be between 0 and 1"),
        ({"short_quantile": -0.1}, "short_quantile must be between 0 and 1"),
        ({"short_quantile": 1.1}, "short_quantile must be between 0 and 1"),
        ({"min_long_revision": -0.01}, "min_long_revision must be non-negative"),
        ({"min_short_revision": -0.01}, "min_short_revision must be non-negative"),
        ({"max_long_names": 0}, "max_long_names must be positive"),
        ({"max_short_names": 0}, "max_short_names must be positive"),
    ],
)
def test_rrg_sector_rotation_validates_concentration_parameters(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_strategy("rrg_sector_rotation", **kwargs)
```

- [ ] **Step 3: Run tests to verify they fail where expected**

Run: `uv run pytest tests/strategies/test_registry.py::test_rrg_sector_rotation_quantile_thresholds_and_caps_reduce_candidates tests/strategies/test_registry.py::test_rrg_sector_rotation_validates_concentration_parameters -q`

Expected before implementation: FAIL because constructor does not accept/apply the new parameters.

---

### Task 3: Implement Concentration Controls

**Files:**
- Modify: `backtesting/strategies/rrg_sector_rotation.py`

- [ ] **Step 1: Add dataclass fields**

Add these fields to both `RrgSectorRotation` and `_RrgLongShortRankProportionalWeight`:

```python
long_quantile: float | None = None
short_quantile: float | None = None
min_long_revision: float = 0.0
min_short_revision: float = 0.0
max_long_names: int | None = None
max_short_names: int | None = None
```

- [ ] **Step 2: Add validation helpers**

Add functions:

```python
def _validate_optional_quantile(name: str, value: float | None) -> None:
    if value is None:
        return
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")


def _validate_optional_positive_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if value <= 0:
        raise ValueError(f"{name} must be positive")
```

- [ ] **Step 3: Validate and pass constructor values**

In `RrgSectorRotation.__post_init__`, validate the new parameters and pass them into `_RrgLongShortRankProportionalWeight`.

- [ ] **Step 4: Add candidate filtering helper**

Add:

```python
def _filter_candidate_scores(
    scores: pd.Series,
    *,
    min_score: float,
    quantile: float | None,
    max_names: int | None,
) -> pd.Series:
    filtered = scores[scores.ge(float(min_score))]
    if filtered.empty:
        return filtered
    if quantile is not None:
        cutoff = float(filtered.quantile(float(quantile)))
        filtered = filtered[filtered.ge(cutoff)]
    if max_names is not None and len(filtered) > max_names:
        filtered = filtered.sort_values(ascending=False, kind="stable").head(max_names)
    return filtered
```

- [ ] **Step 5: Apply helper before weighting**

In `_RrgLongShortRankProportionalWeight.build`, replace raw `long_scores` and `short_scores` with filtered scores before calling `_proportional_rank_weights`.

- [ ] **Step 6: Run focused tests**

Run: `uv run pytest tests/strategies/test_registry.py::test_rrg_sector_rotation_default_parameters_keep_all_confirmed_candidates tests/strategies/test_registry.py::test_rrg_sector_rotation_quantile_thresholds_and_caps_reduce_candidates tests/strategies/test_registry.py::test_rrg_sector_rotation_validates_concentration_parameters -q`

Expected: PASS.

---

### Task 4: Update Strategy Documentation

**Files:**
- Modify: `backtesting/strategies/README.md`

- [ ] **Step 1: Update RRG section**

Add under `rrg_sector_rotation`:

```markdown
- `concentration`: optional quantile/min-revision/max-name controls can reduce
  holdings for personal-account execution without forcing a fixed top-N
  portfolio. Suggested experiment: `long_quantile=0.70`,
  `short_quantile=0.70`, `min_long_revision=0.03`,
  `min_short_revision=0.03`, `max_long_names=20`, `max_short_names=5`.
```

- [ ] **Step 2: Run RRG tests**

Run: `uv run pytest tests/strategies/test_registry.py -q`

Expected: PASS.

---

### Task 5: Backtest Comparison

**Files:**
- No repository file changes required.

- [ ] **Step 1: Run comparison script**

Run an inline Python script that executes these configurations with `shorting.enabled=true`, `weekly`, `next_open`, `fee=0.0002`, `sell_tax=0.0015`, `slippage=0.0005`, and `warmup_days=365`:

```python
variants = {
    "baseline": {},
    "Q70_min3_max20x5": {
        "long_quantile": 0.70,
        "short_quantile": 0.70,
        "min_long_revision": 0.03,
        "min_short_revision": 0.03,
        "max_long_names": 20,
        "max_short_names": 5,
    },
    "Q80_min3_max15x5": {
        "long_quantile": 0.80,
        "short_quantile": 0.80,
        "min_long_revision": 0.03,
        "min_short_revision": 0.03,
        "max_long_names": 15,
        "max_short_names": 5,
    },
    "Q80_min5_max10x3": {
        "long_quantile": 0.80,
        "short_quantile": 0.80,
        "min_long_revision": 0.05,
        "min_short_revision": 0.05,
        "max_long_names": 10,
        "max_short_names": 3,
    },
}
```

- [ ] **Step 2: Report metrics**

Report total return, CAGR, MDD, Sharpe, average turnover, active days, average long count, average short count, last active long count, last active short count, and output directory for each variant.
