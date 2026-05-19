# RRG Sector Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a registered `rrg_sector_rotation` Backtesting strategy that builds a KOSPI200 dollar-neutral sector rotation long-short book from RRG sector regimes, forward revision, and investor flow imbalance.

**Architecture:** Follow the existing `ComposableStrategy` pattern. Put sector-rotation signed-weight construction in `backtesting/construction/sector_rotation.py`; put RRG, forward-revision, flow scoring, diagnostics, and the registered strategy class in `backtesting/strategies/rrg_sector_rotation.py`.

**Tech Stack:** Python, pandas, NumPy, pytest, existing `backtesting` strategy/construction/data contracts.

---

## File Structure

- Create `backtesting/construction/sector_rotation.py`
  - Owns `SectorRotationLongShort`, a construction rule that receives `SignalBundle.alpha` plus sector/leg context and emits signed target weights.
  - Does not calculate RRG, fwd revision, or flow signals.
- Create `backtesting/strategies/rrg_sector_rotation.py`
  - Owns `RrgSectorRotation`, `_RrgSectorRotationSignal`, and focused helper functions for RRG state, bounded fwd deltas, sector-internal ranks, and flow scores.
  - Declares QuantWise dataset requirements.
- Modify `backtesting/strategies/registry.py`
  - Imports and registers `RrgSectorRotation` as `rrg_sector_rotation`.
- Modify `backtesting/strategies/README.md`
  - Documents the strategy id, data, signal, construction, and intended use.
- Create `tests/construction/test_sector_rotation.py`
  - Tests signed sector-rotation construction independent of signal math.
- Create `tests/strategies/test_rrg_sector_rotation.py`
  - Tests signal helpers, dataset contract, registration, and minimal end-to-end plan building.
- Modify `tests/strategies/test_registry.py`
  - Adds the new strategy to expected registry names and export checks.

---

### Task 1: Add Sector Rotation Construction Tests

**Files:**
- Create: `tests/construction/test_sector_rotation.py`
- Later create: `backtesting/construction/sector_rotation.py`

- [ ] **Step 1: Write failing construction tests**

Create `tests/construction/test_sector_rotation.py` with:

```python
import pandas as pd
import pytest

from backtesting.construction.sector_rotation import SectorRotationLongShort
from backtesting.signals.base import SignalBundle


def test_sector_rotation_long_short_builds_dollar_neutral_book() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame(
        {
            "A": [10.0],
            "B": [8.0],
            "C": [2.0],
            "D": [1.0],
            "E": [9.0],
            "F": [0.0],
        },
        index=index,
    )
    sector = pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Tech"],
            "C": ["Finance"],
            "D": ["Finance"],
            "E": ["Energy"],
            "F": ["Energy"],
        },
        index=index,
    )
    long_sector = pd.DataFrame({"Tech": [True], "Finance": [False], "Energy": [False]}, index=index)
    short_sector = pd.DataFrame({"Tech": [False], "Finance": [True], "Energy": [False]}, index=index)
    sector_weight_basis = pd.DataFrame(
        {"A": [70.0], "B": [30.0], "C": [30.0], "D": [70.0], "E": [100.0], "F": [100.0]},
        index=index,
    )
    bundle = SignalBundle(
        alpha=alpha,
        context={
            "sector": sector,
            "long_sector": long_sector,
            "short_sector": short_sector,
            "sector_weight_basis": sector_weight_basis,
            "tradable": alpha.notna(),
        },
    )

    result = SectorRotationLongShort(long_count=2, short_count=2).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(0.5)
    assert weights["C"] == pytest.approx(-0.5)
    assert weights["D"] == pytest.approx(-0.5)
    assert weights["E"] == pytest.approx(0.0)
    assert weights["F"] == pytest.approx(0.0)
    assert weights.sum() == pytest.approx(0.0)
    assert weights.clip(lower=0.0).sum() == pytest.approx(1.0)
    assert (-weights.clip(upper=0.0)).sum() == pytest.approx(1.0)
    assert bool(result.meta["selected_long"].loc[index[0], "A"])
    assert bool(result.meta["selected_short"].loc[index[0], "D"])


def test_sector_rotation_budgets_multiple_sectors_by_kospi200_weight_basis() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame(
        {"A": [10.0], "B": [1.0], "C": [9.0], "D": [0.0], "E": [3.0], "F": [2.0]},
        index=index,
    )
    sector = pd.DataFrame(
        {
            "A": ["Tech"],
            "B": ["Tech"],
            "C": ["Energy"],
            "D": ["Energy"],
            "E": ["Finance"],
            "F": ["Finance"],
        },
        index=index,
    )
    long_sector = pd.DataFrame({"Tech": [True], "Energy": [True], "Finance": [False]}, index=index)
    short_sector = pd.DataFrame({"Tech": [False], "Energy": [False], "Finance": [True]}, index=index)
    sector_weight_basis = pd.DataFrame(
        {"A": [25.0], "B": [75.0], "C": [150.0], "D": [150.0], "E": [80.0], "F": [20.0]},
        index=index,
    )
    bundle = SignalBundle(
        alpha=alpha,
        context={
            "sector": sector,
            "long_sector": long_sector,
            "short_sector": short_sector,
            "sector_weight_basis": sector_weight_basis,
        },
    )

    result = SectorRotationLongShort(long_count=2, short_count=1).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == pytest.approx(0.25)
    assert weights["C"] == pytest.approx(0.75)
    assert weights["F"] == pytest.approx(-1.0)
    assert weights.sum() == pytest.approx(0.0)
    assert result.group_long_budget.loc[index[0], "Tech"] == pytest.approx(0.25)
    assert result.group_long_budget.loc[index[0], "Energy"] == pytest.approx(0.75)
    assert result.group_short_budget.loc[index[0], "Finance"] == pytest.approx(1.0)


def test_sector_rotation_reduces_side_exposure_when_no_qualified_sector_exists() -> None:
    index = pd.to_datetime(["2024-01-02"])
    alpha = pd.DataFrame({"A": [5.0], "B": [1.0]}, index=index)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"]}, index=index)
    long_sector = pd.DataFrame({"Tech": [True]}, index=index)
    short_sector = pd.DataFrame({"Tech": [False]}, index=index)
    bundle = SignalBundle(alpha=alpha, context={"sector": sector, "long_sector": long_sector, "short_sector": short_sector})

    result = SectorRotationLongShort(long_count=1, short_count=1).build(bundle)

    weights = result.base_target_weights.loc[index[0]]
    assert weights["A"] == pytest.approx(1.0)
    assert weights["B"] == pytest.approx(0.0)
    assert weights.sum() == pytest.approx(1.0)
    assert result.meta["side_exposure"].loc[index[0], "long"] == pytest.approx(1.0)
    assert result.meta["side_exposure"].loc[index[0], "short"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify missing module failure**

Run:

```bash
uv run python -m pytest tests/construction/test_sector_rotation.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backtesting.construction.sector_rotation'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/construction/test_sector_rotation.py
git commit -m "Specify sector rotation long-short construction"
```

---

### Task 2: Implement Sector Rotation Construction

**Files:**
- Create: `backtesting/construction/sector_rotation.py`
- Test: `tests/construction/test_sector_rotation.py`

- [ ] **Step 1: Add construction implementation**

Create `backtesting/construction/sector_rotation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.signals.base import SignalBundle
from backtesting.strategy.base import validate_positive

from .base import ConstructionResult


@dataclass(slots=True)
class SectorRotationLongShort:
    long_count: int
    short_count: int
    gross_long: float = 1.0
    gross_short: float = 1.0
    weighting: str = "equal"

    def __post_init__(self) -> None:
        validate_positive("long_count", self.long_count)
        validate_positive("short_count", self.short_count)
        if self.gross_long < 0.0:
            raise ValueError("gross_long must be non-negative")
        if self.gross_short < 0.0:
            raise ValueError("gross_short must be non-negative")
        if self.weighting != "equal":
            raise ValueError(f"unsupported weighting: {self.weighting}")

    def build(self, bundle: SignalBundle) -> ConstructionResult:
        alpha = bundle.alpha.astype(float)
        sector = _required_frame(bundle, "sector").reindex(index=alpha.index, columns=alpha.columns)
        tradable = _optional_frame(bundle, "tradable", default=alpha.notna()).reindex(index=alpha.index, columns=alpha.columns)
        tradable = tradable.astype("boolean").fillna(False).astype(bool)
        long_sector = _required_frame(bundle, "long_sector").reindex(index=alpha.index).fillna(False).astype(bool)
        short_sector = _required_frame(bundle, "short_sector").reindex(index=alpha.index).fillna(False).astype(bool)
        basis = _optional_frame(bundle, "sector_weight_basis", default=alpha.notna().astype(float)).reindex(
            index=alpha.index,
            columns=alpha.columns,
        )
        basis = basis.fillna(0.0).astype(float).clip(lower=0.0)

        weights = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns, dtype=float)
        selected_long = pd.DataFrame(False, index=alpha.index, columns=alpha.columns, dtype=bool)
        selected_short = pd.DataFrame(False, index=alpha.index, columns=alpha.columns, dtype=bool)
        long_budget_rows: dict[pd.Timestamp, dict[object, float]] = {}
        short_budget_rows: dict[pd.Timestamp, dict[object, float]] = {}
        side_exposure_rows: dict[pd.Timestamp, dict[str, float]] = {}

        for ts in alpha.index:
            row_alpha = alpha.loc[ts]
            row_sector = sector.loc[ts]
            row_tradable = tradable.loc[ts] & row_alpha.notna() & row_sector.notna()
            row_basis = basis.loc[ts].where(row_tradable, 0.0)

            long_names, long_budgets = _select_side(
                row_alpha=row_alpha,
                row_sector=row_sector,
                row_tradable=row_tradable,
                row_basis=row_basis,
                active_sectors=_active_sectors(long_sector.loc[ts]),
                count=self.long_count,
                ascending=False,
            )
            short_names, short_budgets = _select_side(
                row_alpha=row_alpha,
                row_sector=row_sector,
                row_tradable=row_tradable,
                row_basis=row_basis,
                active_sectors=_active_sectors(short_sector.loc[ts]),
                count=self.short_count,
                ascending=True,
            )

            for sector_name, names in long_names.items():
                if not names:
                    continue
                sector_budget = long_budgets[sector_name] * self.gross_long
                weights.loc[ts, names] = sector_budget / len(names)
                selected_long.loc[ts, names] = True
            for sector_name, names in short_names.items():
                if not names:
                    continue
                sector_budget = short_budgets[sector_name] * self.gross_short
                weights.loc[ts, names] = -sector_budget / len(names)
                selected_short.loc[ts, names] = True

            long_budget_rows[ts] = long_budgets
            short_budget_rows[ts] = short_budgets
            side_exposure_rows[ts] = {
                "long": float(weights.loc[ts].clip(lower=0.0).sum()),
                "short": float((-weights.loc[ts].clip(upper=0.0)).sum()),
            }

        group_long_budget = pd.DataFrame.from_dict(long_budget_rows, orient="index").reindex(index=alpha.index).fillna(0.0)
        group_short_budget = pd.DataFrame.from_dict(short_budget_rows, orient="index").reindex(index=alpha.index).fillna(0.0)
        side_exposure = pd.DataFrame.from_dict(side_exposure_rows, orient="index").reindex(index=alpha.index).fillna(0.0)
        return ConstructionResult(
            base_target_weights=weights.fillna(0.0).astype(float),
            selection_mask=weights.ne(0.0),
            group_long_budget=group_long_budget.astype(float),
            group_short_budget=group_short_budget.astype(float),
            meta={
                "selected_long": selected_long,
                "selected_short": selected_short,
                "group_id": sector,
                "group_long_budget": group_long_budget.astype(float),
                "group_short_budget": group_short_budget.astype(float),
                "side_exposure": side_exposure.astype(float),
            },
        )


def _required_frame(bundle: SignalBundle, key: str) -> pd.DataFrame:
    value = bundle.context.get(key)
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"sector rotation construction requires {key} context")
    return value


def _optional_frame(bundle: SignalBundle, key: str, *, default: pd.DataFrame) -> pd.DataFrame:
    value = bundle.context.get(key)
    if isinstance(value, pd.DataFrame):
        return value
    return default


def _active_sectors(row: pd.Series) -> set[object]:
    return {sector_name for sector_name, active in row.items() if bool(active)}


def _select_side(
    *,
    row_alpha: pd.Series,
    row_sector: pd.Series,
    row_tradable: pd.Series,
    row_basis: pd.Series,
    active_sectors: set[object],
    count: int,
    ascending: bool,
) -> tuple[dict[object, list[object]], dict[object, float]]:
    candidates: dict[object, pd.Index] = {}
    basis_by_sector: dict[object, float] = {}
    for sector_name in active_sectors:
        mask = row_tradable & row_sector.eq(sector_name)
        names = row_alpha.index[mask]
        if names.empty:
            continue
        candidates[sector_name] = names
        sector_basis = float(row_basis.reindex(names).sum())
        basis_by_sector[sector_name] = sector_basis if sector_basis > 0.0 else float(len(names))

    total_basis = float(sum(basis_by_sector.values()))
    if total_basis <= 0.0:
        return {}, {}

    budgets = {sector_name: basis / total_basis for sector_name, basis in basis_by_sector.items()}
    selected: dict[object, list[object]] = {}
    for sector_name, names in candidates.items():
        target = max(1, round(count * budgets[sector_name]))
        target = min(target, len(names))
        ranked = row_alpha.reindex(names).sort_values(ascending=ascending, kind="mergesort")
        selected[sector_name] = list(ranked.head(target).index)
    return selected, budgets
```

- [ ] **Step 2: Run construction tests**

Run:

```bash
uv run python -m pytest tests/construction/test_sector_rotation.py -v
```

Expected: PASS.

- [ ] **Step 3: Run nearby construction tests**

Run:

```bash
uv run python -m pytest tests/construction/test_rules.py tests/construction/test_sector_rotation.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit implementation**

```bash
git add backtesting/construction/sector_rotation.py tests/construction/test_sector_rotation.py
git commit -m "Add sector rotation long-short construction"
```

---

### Task 3: Add RRG Strategy Helper Tests

**Files:**
- Create: `tests/strategies/test_rrg_sector_rotation.py`
- Later create: `backtesting/strategies/rrg_sector_rotation.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/strategies/test_rrg_sector_rotation.py` with:

```python
import pandas as pd
import pytest

from backtesting.strategies.rrg_sector_rotation import (
    _bounded_delta,
    _classify_rrg_states,
    _sector_rank,
)


def test_bounded_delta_clips_negative_to_positive_explosion() -> None:
    current = pd.DataFrame({"A": [10.0], "B": [12.0]}, index=pd.to_datetime(["2024-01-02"]))
    prior = pd.DataFrame({"A": [-90.0], "B": [10.0]}, index=current.index)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"]}, index=current.index)

    out = _bounded_delta(current=current, prior=prior, sector=sector)

    assert out.loc[current.index[0], "A"] == pytest.approx(1.0)
    assert out.loc[current.index[0], "B"] == pytest.approx(2.0 / 12.0)


def test_sector_rank_scores_within_each_sector() -> None:
    index = pd.to_datetime(["2024-01-02"])
    values = pd.DataFrame({"A": [9.0], "B": [1.0], "C": [8.0], "D": [0.0]}, index=index)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"], "C": ["Energy"], "D": ["Energy"]}, index=index)

    ranked = _sector_rank(values, sector=sector, ascending=True)

    assert ranked.loc[index[0], "A"] == pytest.approx(1.0)
    assert ranked.loc[index[0], "B"] == pytest.approx(0.5)
    assert ranked.loc[index[0], "C"] == pytest.approx(1.0)
    assert ranked.loc[index[0], "D"] == pytest.approx(0.5)


def test_classify_rrg_states_maps_quadrants_to_sector_legs() -> None:
    index = pd.to_datetime(["2024-01-31"])
    medium_ratio = pd.DataFrame({"Lead": [0.2], "Improve": [-0.1], "Weak": [0.2], "Lag": [-0.2]}, index=index)
    medium_momentum = pd.DataFrame({"Lead": [0.1], "Improve": [0.2], "Weak": [-0.2], "Lag": [-0.1]}, index=index)
    short_momentum = pd.DataFrame({"Lead": [-0.1], "Improve": [0.1], "Weak": [-0.1], "Lag": [0.1]}, index=index)

    states, long_sector, short_sector = _classify_rrg_states(
        medium_ratio=medium_ratio,
        medium_momentum=medium_momentum,
        short_momentum=short_momentum,
    )

    assert states.loc[index[0], "Lead"] == "leading"
    assert states.loc[index[0], "Improve"] == "improving"
    assert states.loc[index[0], "Weak"] == "weakening"
    assert states.loc[index[0], "Lag"] == "lagging"
    assert bool(long_sector.loc[index[0], "Lead"])
    assert bool(long_sector.loc[index[0], "Improve"])
    assert bool(short_sector.loc[index[0], "Weak"])
    assert bool(short_sector.loc[index[0], "Lag"])
```

- [ ] **Step 2: Run helper tests to verify missing module failure**

Run:

```bash
uv run python -m pytest tests/strategies/test_rrg_sector_rotation.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backtesting.strategies.rrg_sector_rotation'`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/strategies/test_rrg_sector_rotation.py
git commit -m "Specify RRG sector rotation signal helpers"
```

---

### Task 4: Implement Strategy Helpers And Dataset Contract

**Files:**
- Create: `backtesting/strategies/rrg_sector_rotation.py`
- Test: `tests/strategies/test_rrg_sector_rotation.py`

- [ ] **Step 1: Add initial strategy module with helpers**

Create `backtesting/strategies/rrg_sector_rotation.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.construction.sector_rotation import SectorRotationLongShort
from backtesting.data import MarketData
from backtesting.signals.base import SignalBundle

from .composable import ComposableStrategy


@dataclass(slots=True)
class RrgSectorRotation(ComposableStrategy):
    top_n: int = 25
    bottom_n: int = 25
    lookback: int = 20
    flow_lookback: int = 20
    flow_impulse_lookback: int = 5
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    gross_long: float = 1.0
    gross_short: float = 1.0
    fwd_partial_confidence: float = 0.7
    weighting: str = "equal"

    def __post_init__(self) -> None:
        self.signal_producer = _RrgSectorRotationSignal(
            lookback=self.lookback,
            flow_lookback=self.flow_lookback,
            flow_impulse_lookback=self.flow_impulse_lookback,
            rrg_medium_lookback=self.rrg_medium_lookback,
            rrg_momentum_lookback=self.rrg_momentum_lookback,
            rrg_short_lookback=self.rrg_short_lookback,
            fwd_partial_confidence=self.fwd_partial_confidence,
        )
        self.construction_rule = SectorRotationLongShort(
            long_count=self.top_n,
            short_count=self.bottom_n,
            gross_long=self.gross_long,
            gross_short=self.gross_short,
            weighting=self.weighting,
        )


@dataclass(slots=True)
class _RrgSectorRotationSignal:
    lookback: int = 20
    flow_lookback: int = 20
    flow_impulse_lookback: int = 5
    rrg_medium_lookback: int = 126
    rrg_momentum_lookback: int = 21
    rrg_short_lookback: int = 42
    fwd_partial_confidence: float = 0.7

    @property
    def datasets(self) -> tuple[DatasetId, ...]:
        return (
            DatasetId.QW_ADJ_C,
            DatasetId.QW_BM,
            DatasetId.QW_K200_YN,
            DatasetId.QW_WICS_SEC_BIG,
            DatasetId.QW_MKTCAP,
            DatasetId.QW_V,
            DatasetId.QW_EPS_NFQ1,
            DatasetId.QW_EPS_NFQ2,
            DatasetId.QW_EPS_NFY1,
            DatasetId.QW_OP_NFQ1,
            DatasetId.QW_OP_NFQ2,
            DatasetId.QW_OP_NFY1,
            DatasetId.QW_FOREIGN,
            DatasetId.QW_INSTITUTION,
            DatasetId.QW_RETAIL,
        )

    def build(self, market: MarketData) -> SignalBundle:
        raise NotImplementedError("signal build is implemented in Task 5")


def _bounded_delta(*, current: pd.DataFrame, prior: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    current = current.astype(float)
    prior = prior.reindex(index=current.index, columns=current.columns).astype(float)
    sector = sector.reindex(index=current.index, columns=current.columns)
    raw = current - prior
    abs_estimate = pd.concat(
        [
            current.abs().stack(dropna=False).rename("current"),
            prior.abs().stack(dropna=False).rename("prior"),
            sector.stack(dropna=False).rename("sector"),
        ],
        axis=1,
    )
    abs_estimate.index = abs_estimate.index.set_names(["date", "symbol"])
    sector_median = abs_estimate.groupby([abs_estimate.index.get_level_values("date"), "sector"])[["current", "prior"]].transform("median")
    sector_scale = sector_median.max(axis=1).unstack("symbol").reindex(index=current.index, columns=current.columns)
    scale = pd.concat(
        [
            current.abs().stack(dropna=False),
            prior.abs().stack(dropna=False),
            sector_scale.stack(dropna=False),
        ],
        axis=1,
    ).max(axis=1).unstack().reindex(index=current.index, columns=current.columns)
    scale = scale.where(scale.gt(1e-9), np.nan)
    return raw.divide(scale).clip(lower=-1.0, upper=1.0)


def _sector_rank(values: pd.DataFrame, *, sector: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    values = values.astype(float)
    sector = sector.reindex(index=values.index, columns=values.columns)
    stacked = values.where(sector.notna()).stack()
    if stacked.empty:
        return pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    stacked.index = stacked.index.set_names(["date", "symbol"])
    sector_stacked = sector.stack().reindex(stacked.index)
    ranks = stacked.groupby([stacked.index.get_level_values("date"), sector_stacked]).rank(
        method="average",
        pct=True,
        ascending=ascending,
    )
    return ranks.unstack("symbol").reindex(index=values.index, columns=values.columns)


def _classify_rrg_states(
    *,
    medium_ratio: pd.DataFrame,
    medium_momentum: pd.DataFrame,
    short_momentum: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    medium_ratio = medium_ratio.astype(float)
    medium_momentum = medium_momentum.reindex_like(medium_ratio).astype(float)
    short_momentum = short_momentum.reindex_like(medium_ratio).astype(float)
    states = pd.DataFrame("lagging", index=medium_ratio.index, columns=medium_ratio.columns, dtype=object)
    states = states.mask(medium_ratio.gt(0.0) & medium_momentum.ge(0.0), "leading")
    states = states.mask(medium_ratio.le(0.0) & medium_momentum.gt(0.0), "improving")
    states = states.mask(medium_ratio.gt(0.0) & medium_momentum.lt(0.0), "weakening")
    long_sector = states.eq("leading") | (states.eq("improving") & short_momentum.gt(0.0))
    short_sector = states.eq("lagging") | (states.eq("weakening") & short_momentum.lt(0.0))
    return states, long_sector.astype(bool), short_sector.astype(bool)
```

- [ ] **Step 2: Run helper tests**

Run:

```bash
uv run python -m pytest tests/strategies/test_rrg_sector_rotation.py -v
```

Expected: PASS for helper tests; if a test reaches `_RrgSectorRotationSignal.build`, it should not exist yet in this file.

- [ ] **Step 3: Add dataset contract test**

Append to `tests/strategies/test_rrg_sector_rotation.py`:

```python
from backtesting.strategies.rrg_sector_rotation import RrgSectorRotation


def test_rrg_sector_rotation_declares_required_datasets() -> None:
    dataset_values = {dataset.value for dataset in RrgSectorRotation().datasets}

    assert "qw_adj_c" in dataset_values
    assert "qw_BM" in dataset_values
    assert "qw_k200_yn" in dataset_values
    assert "qw_wics_sec_big" in dataset_values
    assert "qw_mktcap" in dataset_values
    assert "qw_v" in dataset_values
    assert "qw_eps_nfq1" in dataset_values
    assert "qw_eps_nfq2" in dataset_values
    assert "qw_eps_nfy1" in dataset_values
    assert "qw_op_nfq1" in dataset_values
    assert "qw_op_nfq2" in dataset_values
    assert "qw_op_nfy1" in dataset_values
    assert "qw_foreign" in dataset_values
    assert "qw_institution" in dataset_values
    assert "qw_retail" in dataset_values
```

- [ ] **Step 4: Run strategy helper tests again**

Run:

```bash
uv run python -m pytest tests/strategies/test_rrg_sector_rotation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit helpers**

```bash
git add backtesting/strategies/rrg_sector_rotation.py tests/strategies/test_rrg_sector_rotation.py
git commit -m "Add RRG sector rotation signal helpers"
```

---

### Task 5: Implement RRG Signal Build And End-To-End Strategy Test

**Files:**
- Modify: `backtesting/strategies/rrg_sector_rotation.py`
- Modify: `tests/strategies/test_rrg_sector_rotation.py`

- [ ] **Step 1: Add end-to-end strategy test**

Append this test and helper to `tests/strategies/test_rrg_sector_rotation.py`:

```python
from backtesting.data import MarketData


def _rrg_market() -> MarketData:
    index = pd.date_range("2024-01-02", periods=180, freq="D")
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(index=index, columns=cols, dtype=float)
    close["A"] = [100.0 + i * 0.40 for i in range(len(index))]
    close["B"] = [100.0 + i * 0.30 for i in range(len(index))]
    close["C"] = [120.0 - i * 0.25 for i in range(len(index))]
    close["D"] = [120.0 - i * 0.30 for i in range(len(index))]
    benchmark = pd.DataFrame({"IKS200": [100.0 + i * 0.03 for i in range(len(index))]}, index=index)
    k200 = pd.DataFrame(True, index=index, columns=cols)
    sector = pd.DataFrame({"A": ["Tech"], "B": ["Tech"], "C": ["Finance"], "D": ["Finance"]}, index=index).ffill()
    market_cap = pd.DataFrame({"A": [60.0], "B": [40.0], "C": [55.0], "D": [45.0]}, index=index).ffill()
    volume = pd.DataFrame(1000.0, index=index, columns=cols)
    eps_q1 = pd.DataFrame(index=index, columns=cols, dtype=float)
    eps_q1["A"] = [10.0 + i * 0.05 for i in range(len(index))]
    eps_q1["B"] = [9.0 + i * 0.04 for i in range(len(index))]
    eps_q1["C"] = [9.0 - i * 0.04 for i in range(len(index))]
    eps_q1["D"] = [10.0 - i * 0.05 for i in range(len(index))]
    eps_q2 = eps_q1 + 1.0
    eps_y1 = eps_q1 + 2.0
    op_q1 = eps_q1 * 2.0
    op_q2 = eps_q2 * 2.0
    op_y1 = eps_y1 * 2.0
    foreign = pd.DataFrame({"A": [5.0], "B": [4.0], "C": [-4.0], "D": [-5.0]}, index=index).ffill()
    inst = foreign.copy()
    retail = -foreign.copy()
    return MarketData(
        frames={
            "close": close,
            "benchmark": benchmark,
            "k200_yn": k200,
            "sector_big": sector,
            "market_cap": market_cap,
            "volume": volume,
            "eps_fwd_q1": eps_q1,
            "eps_fwd_q2": eps_q2,
            "eps_fwd": eps_y1,
            "op_fwd_q1": op_q1,
            "op_fwd_q2": op_q2,
            "op_fwd": op_y1,
            "foreign_flow": foreign,
            "inst_flow": inst,
            "retail_flow": retail,
        },
        universe=k200,
        benchmark=None,
    )


def test_rrg_sector_rotation_builds_signed_weights_from_market_data() -> None:
    strategy = RrgSectorRotation(
        top_n=2,
        bottom_n=2,
        lookback=20,
        rrg_medium_lookback=60,
        rrg_momentum_lookback=10,
        rrg_short_lookback=20,
    )

    plan = strategy.build_plan(_rrg_market())
    last = plan.target_weights.iloc[-1]

    assert last["A"] > 0.0
    assert last["B"] > 0.0
    assert last["C"] < 0.0
    assert last["D"] < 0.0
    assert last.clip(lower=0.0).sum() == pytest.approx(1.0)
    assert (-last.clip(upper=0.0)).sum() == pytest.approx(1.0)
    assert last.sum() == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify `NotImplementedError`**

Run:

```bash
uv run python -m pytest tests/strategies/test_rrg_sector_rotation.py::test_rrg_sector_rotation_builds_signed_weights_from_market_data -v
```

Expected: FAIL with `NotImplementedError: signal build is implemented in Task 5`.

- [ ] **Step 3: Implement signal build and helper functions**

Replace `_RrgSectorRotationSignal.build` in `backtesting/strategies/rrg_sector_rotation.py` and add the referenced helpers:

```python
    def build(self, market: MarketData) -> SignalBundle:
        close = market.frames["close"].astype(float)
        k200 = market.frames["k200_yn"].reindex_like(close).fillna(False).astype(bool)
        active_columns = k200.columns[k200.any(axis=0)]
        close = close.loc[:, active_columns]
        k200 = k200.loc[:, active_columns]
        sector = market.frames["sector_big"].reindex(index=close.index, columns=close.columns).ffill()
        market_cap = market.frames["market_cap"].reindex(index=close.index, columns=close.columns).ffill()
        benchmark_frame = market.frames["benchmark"]
        benchmark = benchmark_frame["IKS200"] if "IKS200" in benchmark_frame.columns else benchmark_frame.iloc[:, 0]
        benchmark = benchmark.reindex(close.index).ffill()

        rrg_states, long_sector, short_sector = _build_rrg_context(
            close=close,
            benchmark=benchmark,
            sector=sector,
            membership=k200,
            market_cap=market_cap,
            medium_lookback=self.rrg_medium_lookback,
            momentum_lookback=self.rrg_momentum_lookback,
            short_lookback=self.rrg_short_lookback,
        )
        fwd_score, fwd_confidence, fwd_coverage = _build_forward_score(
            frames=market.frames,
            index=close.index,
            columns=close.columns,
            sector=sector,
            lookback=self.lookback,
            partial_confidence=self.fwd_partial_confidence,
        )
        flow_score_20d, flow_score_5d = _build_flow_scores(
            frames=market.frames,
            close=close,
            sector=sector,
            flow_lookback=self.flow_lookback,
            impulse_lookback=self.flow_impulse_lookback,
        )
        alpha = (0.5 * fwd_score.mul(fwd_confidence) + 0.5 * flow_score_20d).where(k200)
        tradable = k200 & alpha.notna()
        return SignalBundle(
            alpha=alpha.where(tradable),
            context={
                "tradable": tradable,
                "sector": sector,
                "long_sector": long_sector,
                "short_sector": short_sector,
                "sector_weight_basis": market_cap.where(k200),
            },
            meta={
                "rrg_state": rrg_states,
                "fwd_score": fwd_score,
                "fwd_confidence": fwd_confidence,
                "fwd_coverage": fwd_coverage,
                "flow_score_20d": flow_score_20d,
                "flow_score_5d": flow_score_5d,
            },
        )
```

Add these helpers below `_classify_rrg_states`:

```python
def _build_rrg_context(
    *,
    close: pd.DataFrame,
    benchmark: pd.Series,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    market_cap: pd.DataFrame,
    medium_lookback: int,
    momentum_lookback: int,
    short_lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = close.pct_change(fill_method=None)
    benchmark_returns = benchmark.pct_change(fill_method=None)
    sector_returns = _sector_weighted_returns(returns=returns, sector=sector, membership=membership, weights=market_cap)
    sector_index = (1.0 + sector_returns.fillna(0.0)).cumprod()
    benchmark_index = (1.0 + benchmark_returns.fillna(0.0)).cumprod()
    relative = sector_index.divide(benchmark_index, axis=0)
    medium_base = relative.rolling(medium_lookback, min_periods=max(5, medium_lookback // 3)).mean()
    short_base = relative.rolling(short_lookback, min_periods=max(5, short_lookback // 3)).mean()
    medium_ratio = relative.divide(medium_base.replace(0.0, np.nan)) - 1.0
    short_ratio = relative.divide(short_base.replace(0.0, np.nan)) - 1.0
    medium_momentum = medium_ratio - medium_ratio.shift(momentum_lookback)
    short_momentum = short_ratio - short_ratio.shift(max(1, momentum_lookback // 2))
    states, long_sector, short_sector = _classify_rrg_states(
        medium_ratio=medium_ratio.fillna(0.0),
        medium_momentum=medium_momentum.fillna(0.0),
        short_momentum=short_momentum.fillna(0.0),
    )
    return states, long_sector, short_sector


def _sector_weighted_returns(
    *,
    returns: pd.DataFrame,
    sector: pd.DataFrame,
    membership: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    rows: dict[pd.Timestamp, dict[object, float]] = {}
    for ts in returns.index:
        valid = membership.loc[ts].astype(bool) & returns.loc[ts].notna() & sector.loc[ts].notna()
        row: dict[object, float] = {}
        for sector_name in pd.unique(sector.loc[ts, valid]):
            names = returns.columns[valid & sector.loc[ts].eq(sector_name)]
            raw_weight = weights.loc[ts, names].fillna(0.0).clip(lower=0.0)
            if float(raw_weight.sum()) <= 0.0:
                raw_weight = pd.Series(1.0, index=names)
            norm_weight = raw_weight / float(raw_weight.sum())
            row[sector_name] = float((returns.loc[ts, names] * norm_weight).sum())
        rows[ts] = row
    return pd.DataFrame.from_dict(rows, orient="index").reindex(index=returns.index)


def _build_forward_score(
    *,
    frames: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
    partial_confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eps = _estimate_composite(
        frames=frames,
        keys=("eps_fwd_q1", "eps_fwd_q2", "eps_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    op = _estimate_composite(
        frames=frames,
        keys=("op_fwd_q1", "op_fwd_q2", "op_fwd"),
        index=index,
        columns=columns,
        sector=sector,
        lookback=lookback,
    )
    count = eps.notna().astype(int) + op.notna().astype(int)
    score = (eps.fillna(0.0) + op.fillna(0.0)).divide(count.replace(0, np.nan))
    confidence = count.replace({0: np.nan, 1: partial_confidence, 2: 1.0}).astype(float)
    coverage = pd.DataFrame(
        {
            "eps_available": eps.notna().sum(axis=1),
            "op_available": op.notna().sum(axis=1),
            "either_available": count.gt(0).sum(axis=1),
            "both_missing": count.eq(0).sum(axis=1),
        },
        index=index,
    )
    return score, confidence, coverage


def _estimate_composite(
    *,
    frames: dict[str, pd.DataFrame],
    keys: tuple[str, ...],
    index: pd.Index,
    columns: pd.Index,
    sector: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    ranks: list[pd.DataFrame] = []
    for key in keys:
        estimate = frames[key].reindex(index=index, columns=columns).ffill()
        prior = estimate.shift(lookback)
        bounded = _bounded_delta(current=estimate, prior=prior, sector=sector)
        ranks.append(_sector_rank(bounded, sector=sector, ascending=True))
    stacked = pd.concat(ranks, keys=range(len(ranks)), names=["horizon", "date"])
    return stacked.groupby(level="date").mean().reindex(index=index, columns=columns)


def _build_flow_scores(
    *,
    frames: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    sector: pd.DataFrame,
    flow_lookback: int,
    impulse_lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    foreign = frames["foreign_flow"].reindex_like(close).fillna(0.0)
    inst = frames["inst_flow"].reindex_like(close).fillna(0.0)
    retail = frames["retail_flow"].reindex_like(close).fillna(0.0)
    volume = frames["volume"].reindex_like(close).fillna(0.0)
    trading_value = close.mul(volume).replace(0.0, np.nan)
    pressure = foreign.add(inst, fill_value=0.0).sub(retail, fill_value=0.0).divide(trading_value)
    score_20d = _rolling_zscore(pressure.rolling(flow_lookback, min_periods=max(2, flow_lookback // 2)).mean(), flow_lookback)
    score_5d = _rolling_zscore(pressure.rolling(impulse_lookback, min_periods=max(2, impulse_lookback // 2)).mean(), impulse_lookback)
    return _sector_rank(score_20d, sector=sector, ascending=True), _sector_rank(score_5d, sector=sector, ascending=True)


def _rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=max(2, window // 2)).mean()
    std = frame.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    return frame.sub(mean).divide(std.replace(0.0, np.nan))
```

- [ ] **Step 4: Run end-to-end strategy test**

Run:

```bash
uv run python -m pytest tests/strategies/test_rrg_sector_rotation.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit signal build**

```bash
git add backtesting/strategies/rrg_sector_rotation.py tests/strategies/test_rrg_sector_rotation.py
git commit -m "Build RRG sector rotation strategy signals"
```

---

### Task 6: Register Strategy And Update Strategy Documentation

**Files:**
- Modify: `backtesting/strategies/registry.py`
- Modify: `tests/strategies/test_registry.py`
- Modify: `backtesting/strategies/README.md`

- [ ] **Step 1: Update registry tests**

Modify `tests/strategies/test_registry.py`:

```python
def test_strategy_modules_export_simple_class_names() -> None:
    from backtesting.strategies.benchmark_overlay import BenchmarkOverlay
    from backtesting.strategies.benchmark_tilt import BenchmarkTilt
    from backtesting.strategies.earnings_revision import EarningsRevision
    from backtesting.strategies.revision_signal import RevisionSignal
    from backtesting.strategies.rrg_sector_rotation import RrgSectorRotation

    assert BenchmarkOverlay.__name__ == "BenchmarkOverlay"
    assert BenchmarkTilt.__name__ == "BenchmarkTilt"
    assert EarningsRevision.__name__ == "EarningsRevision"
    assert RevisionSignal.__name__ == "RevisionSignal"
    assert RrgSectorRotation.__name__ == "RrgSectorRotation"
```

In `test_registry_lists_default_strategies`, add:

```python
    assert "rrg_sector_rotation" in strategies
```

In `test_registry_lists_screened_strategy_names_only`, change the exact set to:

```python
    assert strategies == {
        "trend_rank",
        "earnings_revision",
        "revision_signal",
        "benchmark_overlay",
        "benchmark_tilt",
        "rrg_sector_rotation",
    }
```

- [ ] **Step 2: Run registry tests to verify failure**

Run:

```bash
uv run python -m pytest tests/strategies/test_registry.py -v
```

Expected: FAIL because `rrg_sector_rotation` is not registered.

- [ ] **Step 3: Register the strategy**

Modify `backtesting/strategies/registry.py`:

```python
from .rrg_sector_rotation import RrgSectorRotation
```

Add registration:

```python
register_strategy("rrg_sector_rotation", RrgSectorRotation)
```

- [ ] **Step 4: Add README entry**

Add this section to `backtesting/strategies/README.md` under Active Strategies:

```markdown
### RRG Sector Rotation

- `id`: `rrg_sector_rotation`
- `file`: `rrg_sector_rotation.py`
- `class`: `RrgSectorRotation`
- `profile`: alpha or long-short research.
- `data`: `close`, `benchmark`, `k200_yn`, `sector_big`, `market_cap`, `volume`, EPS/OP forward estimate horizons, and foreign/institution/retail flow.
- `signal`: RRG-style WICS sector regimes against KOSPI200, plus sector-internal forward revision and investor flow imbalance ranks.
- `construction`: dollar-neutral sector rotation long-short. Leading/confirmed Improving sectors feed the long leg; Lagging/confirmed Weakening sectors feed the short leg.
- `use`: research whether sector RRG direction adds value beyond fwd revision and daily investor flow imbalance in KOSPI200 names.
```

- [ ] **Step 5: Run registry tests**

Run:

```bash
uv run python -m pytest tests/strategies/test_registry.py tests/strategies/test_rrg_sector_rotation.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit registration and docs**

```bash
git add backtesting/strategies/registry.py tests/strategies/test_registry.py backtesting/strategies/README.md
git commit -m "Register RRG sector rotation strategy"
```

---

### Task 7: Add Shorting Assumption Smoke Spec Path

**Files:**
- Modify: `tests/strategies/test_rrg_sector_rotation.py`

- [ ] **Step 1: Add explicit shorting reminder test**

Append this test to `tests/strategies/test_rrg_sector_rotation.py` so future maintainers remember that the strategy emits negative target weights:

```python
def test_rrg_sector_rotation_emits_negative_weights_for_short_leg() -> None:
    strategy = RrgSectorRotation(
        top_n=2,
        bottom_n=2,
        lookback=20,
        rrg_medium_lookback=60,
        rrg_momentum_lookback=10,
        rrg_short_lookback=20,
    )

    weights = strategy.build_weights(_rrg_market())

    assert weights.min().min() < 0.0
    assert weights.max().max() > 0.0
```

- [ ] **Step 2: Run focused strategy tests**

Run:

```bash
uv run python -m pytest tests/strategies/test_rrg_sector_rotation.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit smoke coverage**

```bash
git add tests/strategies/test_rrg_sector_rotation.py
git commit -m "Cover RRG sector rotation signed weights"
```

---

### Task 8: Run Full Verification

**Files:**
- No new file changes expected.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run python -m pytest tests/construction/test_sector_rotation.py tests/strategies/test_rrg_sector_rotation.py tests/strategies/test_registry.py -v
```

Expected: PASS.

- [ ] **Step 2: Run relevant broader suites**

Run:

```bash
uv run python -m pytest tests/construction tests/strategies tests/run -v
```

Expected: PASS.

- [ ] **Step 3: Run full root test suite**

Run:

```bash
uv run python -m pytest
```

Expected: PASS. If unrelated failures appear, capture exact failing test names and error messages before deciding whether the failure is in scope.

- [ ] **Step 4: Inspect git diff**

Run:

```bash
git diff --stat
git diff -- backtesting/construction/sector_rotation.py backtesting/strategies/rrg_sector_rotation.py tests/construction/test_sector_rotation.py tests/strategies/test_rrg_sector_rotation.py tests/strategies/test_registry.py backtesting/strategies/README.md
```

Expected: diff only contains the RRG strategy, construction rule, tests, and README entry.

- [ ] **Step 5: Final implementation commit**

If all prior task commits already exist, skip this step. If there are uncommitted verification fixes, commit them:

```bash
git add backtesting/construction/sector_rotation.py backtesting/strategies/rrg_sector_rotation.py tests/construction/test_sector_rotation.py tests/strategies/test_rrg_sector_rotation.py tests/strategies/test_registry.py backtesting/strategies/README.md
git commit -m "Verify RRG sector rotation integration"
```

---

## Self-Review

Spec coverage:

- Registered strategy under `backtesting/strategies`: Task 6.
- Existing Backtesting integration through `ComposableStrategy`: Tasks 4-6.
- Dollar-neutral sector rotation long-short construction: Tasks 1-2.
- KOSPI200 universe and WICS sectors: Task 5.
- RRG strong-sector long and weak-sector short legs: Tasks 3 and 5.
- Forward estimate horizons with bounded deltas: Tasks 3-5.
- Flow imbalance proxy from foreign/institution/retail over trading value: Task 5.
- Missing forward estimate policy: Task 5 builds confidence and excludes no-forward alpha through `alpha.notna`; add more granular tests in implementation if the first test exposes ambiguity.
- Weighting variants beyond `equal`: not implemented in the first code path because the current construction test and implementation intentionally reject unsupported weighting. Add `score` and `market_cap_tilt` only after the base equal-weight strategy passes and produces interpretable results.
- Diagnostics: Task 5 stores RRG state, fwd score, fwd confidence, fwd coverage, and flow scores; Task 2 stores side exposure and sector budgets.

Placeholder scan:

- No open placeholder markers or omitted implementation steps are present.
- Unsupported future variants are explicitly rejected or scoped outside the first implementation path.

Type consistency:

- `RrgSectorRotation` uses `top_n` and `bottom_n` because `RunConfig` already exposes `top_n`; `bottom_n` is accepted by `build_strategy` when passed through specs or direct construction.
- `SectorRotationLongShort` consumes `SignalBundle.context` keys produced by `_RrgSectorRotationSignal.build`: `sector`, `long_sector`, `short_sector`, `sector_weight_basis`, and `tradable`.
- Dataset frame keys match `DataLoader.FRAME_KEYS`: `eps_fwd_q1`, `eps_fwd_q2`, `eps_fwd`, `op_fwd_q1`, `op_fwd_q2`, `op_fwd`, `foreign_flow`, `inst_flow`, and `retail_flow`.
