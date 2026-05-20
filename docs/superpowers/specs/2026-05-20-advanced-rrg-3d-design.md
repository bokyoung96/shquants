# Advanced 3D RRG Framework Design

## Goal

Build a reusable `rrg/` package for KOSPI200 WICS sector rotation analysis. The first version is not a standard two-dimensional RRG clone. It is a three-dimensional relative-rotation phase-space tool where sectors move through:

- relative trend
- relative momentum
- relative acceleration
- turning point states
- persistence states
- multi-horizon states

The framework is a visualization and interpretation system, not a trading signal.

## Scope

In scope:

- KOSPI200 WICS large-sector data from the existing backtesting data stack.
- Sector portfolio construction from KOSPI200 constituents and market-cap weights.
- Log relative strength calculation.
- Multi-horizon metrics for short, medium, and long horizons.
- Relative momentum as first difference of log relative strength.
- Relative acceleration as second difference of log relative strength.
- Optional smoothing hooks.
- Acceleration normalization.
- Turning-point and persistence labels.
- Plotly 3D RRG visualization.
- HTML export example.
- Documentation and interpretation guide.

Out of scope for the first version:

- ETF input adapters.
- Trading strategy registration.
- Existing dashboard frontend integration.
- Portfolio construction or backtest execution.
- Proprietary JdK RS-Ratio replication.

## References

Traditional RRGs plot relative trend and relative momentum in a four-quadrant plane. StockCharts describes RRGs as a way to visualize relative performance trends against a common benchmark and emphasizes quadrant movement and trails. RRG Research describes the two main normalized inputs as JdK RS-Ratio and JdK RS-Momentum.

This project keeps the interpretive structure but uses explicit, transparent log-relative-strength derivatives instead of attempting to reproduce proprietary JdK formulas.

- StockCharts RRG overview: https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-types/relative-rotation-graphs-rrg-charts
- RRG Research building blocks: https://relativerotationgraphs.com/educational/the-building-blocks-for-rrg/

## Data Model

The data adapter will use the existing backtesting data infrastructure:

- `backtesting.catalog.DatasetId`
- `backtesting.data.DataLoader`
- `backtesting.data.ParquetStore`
- `backtesting.universe.UniverseRegistry`

Required datasets:

- `QW_ADJ_C` as adjusted close
- `QW_BM` as benchmark
- `QW_K200_YN` as KOSPI200 membership
- `QW_WICS_SEC_BIG` as WICS large sector
- `QW_MKTCAP` as market-cap weight basis

The adapter will produce:

- `sector_prices`: sector-level synthetic price/index series
- `benchmark`: KOSPI200 benchmark series
- `sector_membership_count`: number of valid constituents by date and sector
- `sector_weight_sum`: market-cap basis used by date and sector

Sector returns are computed by date:

1. Filter to KOSPI200 members with valid return and valid `sector_big`.
2. Group by WICS large sector.
3. Use nonnegative market cap as within-sector weights.
4. Fall back to equal weight inside a sector if market-cap weight sum is zero.
5. Compound returns into sector index series.

## Core Metrics

For each sector and horizon `k`:

```text
RS_t = sector_index_t / benchmark_index_t
LRS_t = log(RS_t)
MOM_t = LRS_t - LRS_{t-k}
ACC_t = MOM_t - MOM_{t-k}
      = LRS_t - 2 * LRS_{t-k} + LRS_{t-2k}
```

The default horizons are:

| Name | k |
| --- | ---: |
| short | 5 |
| medium | 20 |
| long | 60 |

The framework will return one tidy table with:

- `date`
- `sector`
- `horizon`
- `rs`
- `log_rs`
- `rs_centered`
- `mom`
- `acc`
- `acc_z`
- `state`
- `turning_label`
- `persistence`
- `confidence`

`rs_centered` is the x-axis default:

```text
rs_centered = log_rs - rolling_mean(log_rs, trend_window)
```

This keeps the 3D plot centered around a stable origin while preserving the raw `rs` and `log_rs`.

## State Classification

Traditional quadrant labels use trend and momentum:

| State | Condition |
| --- | --- |
| Leading | `rs_centered >= 0` and `mom >= 0` |
| Improving | `rs_centered < 0` and `mom >= 0` |
| Lagging | `rs_centered < 0` and `mom < 0` |
| Weakening | `rs_centered >= 0` and `mom < 0` |
| Unclassified | required values are missing |

Acceleration enriches interpretation:

| Turning label | Condition |
| --- | --- |
| Trend strengthening | `mom > 0` and `acc_z > threshold` |
| Exhaustion risk | `mom > 0` and `acc_z < -threshold` |
| Recovery candidate | `mom < 0` and `acc_z > threshold` |
| Breakdown pressure | `mom < 0` and `acc_z < -threshold` |
| Neutral | acceleration is inside threshold |

Persistence is measured as consecutive periods in the same state, capped for display.

## Smoothing And Normalization

The first version will support deterministic smoothing:

- `none`
- exponential moving average
- rolling mean

Kalman filtering will be designed as an optional interface but not required for the first implementation unless SciPy-free implementation is small and testable.

Normalized acceleration:

```text
acc_z = acc / rolling_std(acc, window)
```

If rolling standard deviation is zero or missing, `acc_z` is missing.

## 3D Visualization

The primary plot is a Plotly 3D phase-space RRG:

- x-axis: `rs_centered`
- y-axis: `mom`
- z-axis: `acc_z` by default, raw `acc` optionally
- line trails: latest `trail_length` observations per sector
- marker: latest point per sector
- color: acceleration, using red to green continuous scale
- marker size: `confidence`, defaulting to constituent-count confidence
- hover: sector, horizon, state, turning label, persistence, RS, MOM, ACC

The plot should include origin planes or reference lines:

- `x = 0`
- `y = 0`
- `z = 0`

Multi-horizon behavior is handled by:

- one figure per horizon, or
- a Plotly dropdown that switches horizon traces.

The MVP default will be a single HTML file with a dropdown for `short`, `medium`, and `long`.

## Module Layout

```text
rrg/
  __init__.py
  core.py
  data.py
  filters.py
  plot.py
  dashboard.py
  examples.py
  README.md
tests/
  rrg/
    test_core.py
    test_data.py
    test_plot.py
```

Responsibilities:

- `rrg/core.py`: formulas, horizon computation, state labels, persistence.
- `rrg/data.py`: KOSPI200 WICS sector data adapter using existing backtesting loaders.
- `rrg/filters.py`: smoothing and acceleration scaling helpers.
- `rrg/plot.py`: Plotly 3D figure construction.
- `rrg/dashboard.py`: multi-horizon HTML export wrapper.
- `rrg/examples.py`: runnable example entry points.
- `rrg/README.md`: formulas, interpretation, and usage.

## API Sketch

```python
from rrg.data import load_kospi200_wics_sector_rrg_input
from rrg.core import RrgConfig, compute_multi_horizon_rrg
from rrg.dashboard import export_multi_horizon_rrg

input_data = load_kospi200_wics_sector_rrg_input(
    start="2020-01-02",
    end="2026-03-25",
)

result = compute_multi_horizon_rrg(
    sector_prices=input_data.sector_prices,
    benchmark=input_data.benchmark,
    confidence=input_data.confidence,
    config=RrgConfig(),
)

export_multi_horizon_rrg(
    result,
    output_path="results/rrg/advanced_rrg_3d.html",
)
```

## Testing Strategy

Tests will be written before implementation.

Core tests:

- `log_rs` equals `log(sector / benchmark)`.
- `mom` equals `LRS_t - LRS_{t-k}`.
- `acc` equals `LRS_t - 2LRS_{t-k} + LRS_{t-2k}`.
- quadrant labels match all four sign combinations.
- turning labels distinguish exhaustion and recovery.
- persistence counts consecutive state runs.
- multi-horizon output has expected tidy columns.

Data tests:

- sector returns use market-cap weights inside each sector.
- sector returns fall back to equal weight if market-cap sum is zero.
- missing sector labels are excluded.
- benchmark is aligned to sector index dates.

Plot tests:

- 3D figure contains `scatter3d` traces.
- exported HTML is created.
- horizon dropdown contains the configured horizons.

## Interpretation Guide

The 3D view should be read as phase-space, not as a buy/sell signal:

- High x, high y, high z: leadership still strengthening.
- High x, high y, low z: leadership may be exhausting.
- Low x, low y, high z: recovery may be forming.
- Low x, low y, low z: relative weakness is still worsening.
- Long persistence in Leading with fading acceleration suggests mature leadership.
- Improving with rising acceleration is an early transition candidate.

## Deliverables

- `rrg/` package.
- Unit tests for formulas, data adapter, and Plotly construction.
- KOSPI200 WICS example.
- Plotly 3D HTML export.
- Formula and interpretation documentation.

## Spec Self-Review

- No ETF support is included in this phase.
- The design does not depend on proprietary RRG formulas.
- The implementation remains separate from backtesting strategy registration.
- The first visualization is explicitly 3D, not a 2D dashboard.
- All required formulas have deterministic tests.
