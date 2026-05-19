# RRG Sector Rotation Long-Short Strategy Design

## Goal

Add a registered Backtesting strategy under `backtesting/strategies` that tests whether WICS sector rotation, measured by RRG-style relative strength against KOSPI200, can improve a KOSPI200 stock-selection long-short portfolio.

The strategy is a research strategy first, not a benchmark-overlay implementation portfolio. It should produce a dollar-neutral sector rotation long-short book:

- Long gross: 1.0
- Short gross: 1.0
- Net exposure: near 0.0
- Gross exposure: near 2.0
- Universe: KOSPI200 constituents only

The production-style benchmark tilt version can be designed later if this research strategy shows value.

## Strategic Intent

The strategy has two decision layers.

1. Sector layer: use RRG-style sector regimes to decide which sectors belong in the long leg and which sectors belong in the short leg.
2. Stock layer: inside those sectors, use forward estimate revision and investor flow imbalance to select the best long candidates and weakest short candidates.

This keeps the interpretation clean:

- RRG answers: which sectors are strong or weak against KOSPI200?
- Forward revision and flow answer: which stocks within those sectors should be long or short?

## Strategy Identity

Recommended registered strategy id:

```text
rrg_sector_rotation
```

Recommended module:

```text
backtesting/strategies/rrg_sector_rotation.py
```

The strategy should follow the existing `ComposableStrategy` pattern used by `benchmark_tilt` and `benchmark_overlay`:

- A signal producer loads market frames and builds alpha, masks, diagnostics, and sector-leg context.
- A construction rule turns that signal bundle into signed target weights.
- `registry.py` registers the strategy for CLI, specs, presets, and dashboard launch compatibility.

## Data Requirements

The initial version should use existing QuantWise-backed dataset ids and semantic frame keys:

| Purpose | Dataset ids | Frame keys |
| --- | --- | --- |
| Prices | `QW_ADJ_C`, optionally `QW_ADJ_O` for fills outside this strategy | `close` |
| Benchmark | `QW_BM` | `benchmark` |
| KOSPI200 membership | `QW_K200_YN` | `k200_yn` |
| Sector taxonomy | `QW_WICS_SEC_BIG` | `sector_big` |
| Market cap | `QW_MKTCAP` or `QW_MKTCAP_FLT` | `market_cap` / `float_market_cap` |
| Volume | `QW_V` | `volume` |
| EPS estimates | `QW_EPS_NFQ1`, `QW_EPS_NFQ2`, `QW_EPS_NFY1` | `eps_fwd_q1`, `eps_fwd_q2`, `eps_fwd` |
| OP estimates | `QW_OP_NFQ1`, `QW_OP_NFQ2`, `QW_OP_NFY1` | `op_fwd_q1`, `op_fwd_q2`, `op_fwd` |
| Investor flow | `QW_FOREIGN`, `QW_INSTITUTION`, `QW_RETAIL` | `foreign_flow`, `inst_flow`, `retail_flow` |

`DataLoader.FRAME_KEYS` already maps these estimate and flow datasets into semantic frame keys. The new strategy should declare these datasets directly from its signal producer.

## RRG Sector Signal

### Sector Return Series

Build WICS large-sector return series from KOSPI200 constituents:

1. Restrict stocks to `k200_yn`.
2. Use `sector_big` as the sector group.
3. Use market cap or float market cap as the sector return weight basis.
4. Compute daily sector returns from constituent close returns.
5. Compare each sector against KOSPI200 benchmark returns.

The first implementation should prefer `market_cap` because it is already used by current benchmark-aware strategies. `float_market_cap` can be a later option if data quality is stable for the full test window.

### RRG Regimes

The strategy should calculate two RRG-style views:

- Medium-term RRG: main sector-selection signal, roughly 12-26 weeks.
- Short-term RRG: confirmation signal, roughly 4-12 weeks.

Exact default windows can be implementation parameters, but the design intent is:

- Medium-term signal decides the main regime.
- Short-term signal confirms Improving entries and Weakening exits.

Initial regime mapping:

| Medium-term state | Short-term confirmation | Action |
| --- | --- | --- |
| Leading | Not required | Long-sector candidate |
| Improving | Improving/positive confirmation required | Long-sector candidate |
| Weakening | Weakening/negative confirmation required | Short-sector candidate |
| Lagging | Not required | Short-sector candidate |

If there are too few long or short sectors on a date, the strategy should reduce exposure for that side rather than forcing weak classifications into the book.

## Forward Revision Signal

Forward revision should use estimate changes, not estimate levels.

### Horizons

Use all available forward horizons without arbitrary horizon weights:

- EPS: `NFQ1`, `NFQ2`, `NFY1`
- OP: `NFQ1`, `NFQ2`, `NFY1`

For each estimate frame and each date:

```text
raw_delta = current_forward_estimate - prior_forward_estimate
```

where `prior_forward_estimate` is shifted by `lookback` trading days after the monthly frames have been expanded by the data loader.

### Bounded Delta

Raw delta can be distorted when an estimate moves from negative to positive or from a large loss to a smaller loss. Use a bounded delta before ranking:

```text
scale = max(abs(current), abs(prior), sector_median_abs_estimate, epsilon)
bounded_delta = clip((current - prior) / scale, -1, 1)
```

This preserves upgrade and downgrade direction while preventing sign-change explosions from dominating the cross-section.

### Sector-Internal Ranking

Convert bounded deltas into rank scores inside each sector and date. The stock layer should rank within sector because RRG already handles sector selection. Whole-KOSPI200 rank can be saved as diagnostic data, but should not drive the base strategy.

Composite rules:

- EPS composite: simple average of available EPS horizon rank scores.
- OP composite: simple average of available OP horizon rank scores.
- Forward score: simple average of available EPS and OP composites.
- If both EPS and OP composites exist, confidence is full.
- If only one of EPS or OP exists, apply a confidence penalty.
- If neither exists, exclude the stock from the base strategy.

The first version should keep the confidence penalty parameterized. The design assumption is that partial estimate coverage should be allowed, but should not be treated as equal-quality evidence.

## Flow Imbalance Signal

Use investor flow imbalance as a daily proxy for order imbalance. The code and documentation should call it `flow_imbalance`, not pure order imbalance, because the source is daily investor-category flow rather than order-book imbalance.

Raw pressure:

```text
flow_pressure = (foreign_flow + inst_flow - retail_flow) / trading_value
trading_value = close * volume
```

Signal shape:

- Main flow signal: 20-day accumulation pressure.
- Timing signal: 5-day impulse pressure.
- Standardization: rolling z-score after normalizing by trading value.
- Ranking: sector-internal rank on each date.

The base stock score should use the 20-day flow score. The 5-day flow score should be retained as a timing or weight-adjustment input, not as a full equal leg in the initial score.

## Stock Score

Base stock score:

```text
stock_score = 0.5 * fwd_score * fwd_confidence + 0.5 * flow_score_20d
```

This intentionally keeps forward revision and flow equally weighted in the first version. State-conditioned weights can be a later experiment, but should not be introduced before the base strategy is measurable.

Long candidates are high `stock_score` names inside long sectors. Short candidates are low `stock_score` names inside short sectors.

## Portfolio Construction

### Exposure

The base strategy is dollar-neutral:

```text
gross_long = 1.0
gross_short = 1.0
```

The construction rule should not be sector-neutral. RRG needs sector direction to matter:

- Strong sectors populate the long leg.
- Weak sectors populate the short leg.

### Sector Budgets

Within each leg, selected sector budgets should be based on KOSPI200 sector market-cap weights and then renormalized inside the leg.

Example:

```text
long_sector_budget[s] = k200_sector_weight[s] / sum(k200_sector_weight for selected long sectors)
short_sector_budget[s] = k200_sector_weight[s] / sum(k200_sector_weight for selected short sectors)
```

This keeps sector sizing linked to the benchmark structure without making the portfolio a benchmark overlay.

### Name Counts

Target name count should be concentrated:

- Long side target: 20-30 names.
- Short side target: 20-30 names.

Per-sector target counts should be proportional to sector budget and capped by eligible names:

```text
raw_target = round(side_target_names * sector_budget)
target = min(raw_target, eligible_names, max_names_per_sector)
```

If a selected sector has too few eligible names, the unused budget should be redistributed across other selected sectors on the same side when possible. If redistribution is impossible, reduce exposure for that side and record the exposure shortfall in diagnostics.

### Weighting Variants

The base implementation should support three weighting modes:

1. `equal`: selected names are equally weighted inside each sector budget.
2. `score`: selected names are weighted by transformed stock-score strength.
3. `market_cap_tilt`: selected names start from sector-internal KOSPI200 market-cap weights and receive a bounded score tilt.

The first backtest interpretation should use `equal` as the core result because it tests signal selection with the fewest assumptions. `score` and `market_cap_tilt` are sensitivity tests.

## Rebalancing

Base cadence:

- Sector selection: monthly.
- Stock selection: weekly.

Implementation should fit the repository's existing schedule model as much as possible. The first practical implementation may build target weights for all dates while only changing sector regimes on monthly evaluation dates and stock membership on weekly evaluation dates.

If turnover is too high, a later version should consider the existing `signal_dates` schedule semantics so trades occur only when target weights change materially.

## Backtesting Integration

The implementation should integrate with existing Backtesting surfaces:

- `backtesting/strategies/rrg_sector_rotation.py`: new strategy module.
- `backtesting/strategies/registry.py`: register `rrg_sector_rotation`.
- `backtesting/strategies/README.md`: document strategy id, data, signal, construction, and use case.
- Tests under `tests/strategies/`: verify strategy contract and registration.
- Tests under `tests/construction/` or `tests/strategies/`: verify sector rotation long-short construction edge cases.

The strategy should expose constructor parameters compatible with `build_strategy`, which filters CLI/spec kwargs by signature:

- `lookback`
- `flow_lookback`
- `top_n` or side-specific target counts
- RRG windows
- weighting mode
- gross long/short exposure
- confidence penalty

Because `RunConfig` currently has limited generic kwargs, the first CLI path can use existing fields where sensible and add a preset later if needed. A dedicated preset can carry stable defaults after the first strategy tests pass.

## Diagnostics And Reporting

The strategy should save enough meta information in `SignalBundle.meta` or construction meta for reports/tests to inspect the research claim.

Required diagnostics:

- RRG state by sector and date.
- Long-sector and short-sector selection masks.
- Sector long and short budgets.
- Selected long and short stock masks.
- Forward coverage by sector.
- Count of names with EPS+OP, EPS-only, OP-only, and no forward signal.
- Count of excluded no-forward names.
- Bounded delta profit-state diagnostics:
  - positive to positive
  - negative to positive
  - negative to less negative
  - positive to negative
  - negative to more negative
- Long-leg return and short-leg return contribution where reporting support allows it.

The base report should evaluate:

- CAGR
- MDD
- Sharpe
- turnover
- long/short spread return
- gross and net exposure
- market beta
- sector contribution
- RRG-regime contribution
- fwd-score and flow-score bucket performance

## Comparison Experiments

The initial implementation should make these comparisons easy:

1. Core equal-weight RRG sector rotation long-short.
2. Score-weighted variant.
3. Market-cap tilt variant.
4. Flow fallback sensitivity: limited flow-only names when forward estimates are absent.
5. No-RRG control: same fwd/flow stock ranking without sector regime gating.

The no-RRG control is important. It tests whether RRG adds value beyond the forward revision and flow stock signal.

## Edge Cases

The implementation should handle:

- No long sectors or no short sectors on a date.
- Selected sector has fewer eligible names than its budget implies.
- Missing sector labels.
- Missing benchmark column name; follow existing convention of using `IKS200` when present, otherwise first benchmark column.
- Missing or zero trading value.
- Forward estimates crossing zero.
- Monthly estimate frames expanded to daily dates.
- Short target weights requiring shorting assumptions in specs/runs.

For no-sector or no-leg cases, prefer reduced exposure and explicit diagnostics over forced positions.

## Out Of Scope

- Full benchmark-overlay implementation portfolio.
- Intraday or order-book order imbalance.
- Broker locate, recall, liquidation, or margin waterfall modeling.
- New data dependencies.
- New external packages.
- Frontend/dashboard UI changes.

## Acceptance Criteria

- `rrg_sector_rotation` can be built through `backtesting.strategies.build_strategy`.
- The strategy declares all required datasets through its signal producer.
- A focused synthetic test verifies long sectors produce positive weights and short sectors produce negative weights.
- A focused synthetic test verifies bounded forward delta does not let negative-to-positive transitions explode unbounded.
- A focused synthetic test verifies missing forward estimates apply the intended coverage policy.
- Existing strategy contract and registry tests pass.
- A smoke backtest can run through `backtesting.run` once real parquet data is available for the required frames.

