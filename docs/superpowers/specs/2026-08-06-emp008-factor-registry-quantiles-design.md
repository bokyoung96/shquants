# EMP008 Factor Registry and Quantile Pipeline Design

Date: 2026-08-06
Status: Approved for planning
Scope: `backtesting/strategies/emp008`

## Context

EMP008 currently chooses factors through repeated string comparisons. Factor-set membership lives in `mfbt_emp008_factors.py`, required datasets are repeated in `mfbt_emp008_data.py`, CLI choices are repeated in two entry points, and factor-direction rules are embedded in expected-alpha policy branches. Adding a factor therefore requires synchronized edits in several files, and a missed edit can silently load the wrong data or evaluate the factor with the wrong economic direction.

EMP008 also has no first-class single-factor portfolio evaluation. The optimizer and attribution use preprocessed model exposures, while the repository's generic `quantile_returns` helper only calculates equal-weight bucket returns and does not preserve portfolio weights, factor direction, monthly metrics, or market-cap-weighted variants.

## Goals

1. Give every EMP008 factor and factor set a unique typed identifier.
2. Make one registry authoritative for factor construction, required data, preprocessing, direction, and factor-set membership.
3. Evaluate every factor in the selected factor set independently as monthly Q1-Q5 portfolios.
4. Produce both equal-weighted and market-cap-weighted results from identical bucket membership.
5. Ensure optimization, attribution, and quantile evaluation consume the same preprocessed model exposures.
6. Integrate the evaluation into `run_full` while also exposing a standalone EMP008 quantile command.
7. Preserve the current optimizer, Origin/MFBT variants, public string values, and existing output names.

## Non-goals

- Automatically optimize factor weights from quantile results.
- Add sector-neutral quantile portfolios in this iteration.
- Add transaction costs to diagnostic quantile returns.
- Replace the production EMP008 optimizer with a quantile portfolio.
- Introduce a plugin discovery framework or a new dependency.

The outputs are deliberately diagnostic. They provide evidence for a later, separately controlled change to factor selection or optimization policy.

## Chosen Approach

Use an explicit enum-backed registry, not scattered conditionals and not auto-discovered plugin classes.

### Identifiers

Define string enums decorated with `@unique`:

- `FactorId`: current model-facing factor names.
- `FactorSetId`: `mfbt`, `mfbt_pos`, `mfbt_origin_smallcap`, `origin`, and `origin_new_dividend`.
- `FactorDirection`: `high` or `low`.
- `QuantileWeighting`: `equal_weight` or `market_cap_weight`.

The enum values remain strings so existing CLI arguments, diagnostics, Excel outputs, and expected-alpha indexes remain backward compatible. Origin names such as `LnMktcap`, `Momentum_12M`, and `DY` remain distinct model factors because their preprocessing or source semantics differ from the MFBT variants even when a calculation helper is shared.

### Factor definitions

Each `FactorDefinition` is immutable and contains:

- unique `FactorId`;
- raw-frame builder callable;
- factor-specific `DatasetId` dependencies;
- expected economic direction;
- preprocessing metadata such as rank transformation;
- optional configuration attributes for winsorization and z-score caps;
- optional large-benchmark-weight neutralization behavior.

Each `FactorSetDefinition` is immutable and contains:

- unique `FactorSetId`;
- ordered tuple of factor IDs;
- expected-alpha policy;
- monthly snapshot forward allowance.

Registry construction validates uniqueness, complete factor-set references, and non-empty factor sets at import time. Public accessors return definitions rather than exposing mutable dictionaries.

### Single source of truth

The registry drives:

- `build_raw_mfbt_factors` order and builders;
- `required_datasets` factor-specific union;
- preprocessing options;
- large-benchmark neutralization membership;
- expected direction for the diagnostic spread;
- CLI `choices` for factor sets;
- variant defaults in `build_emp008_config`.

Common execution datasets remain explicit in the data module because they are strategy infrastructure rather than factor dependencies: adjusted close, benchmark weights, total market cap, float market cap, KOSPI200 membership, and the selected sector taxonomy.

## Shared Preparation Boundary

Add a preparation boundary that loads market data once and returns an immutable `PreparedEmp008Factors` bundle containing:

- active configuration and factor-set definition;
- aligned market frames needed downstream;
- ordered raw factors;
- ordered, fully preprocessed model factors;
- sector exposures and completed benchmark weights;
- common monthly factor dates.

The preprocessing sequence remains identical to today's optimizer:

1. restrict to the KOSPI200 universe;
2. apply registered raw winsorization when configured;
3. fill missing values with float-market-cap-weighted cross-sectional means;
4. apply registered rank transforms;
5. float-market-cap center and standardize;
6. apply configured z-score caps;
7. apply registered large-benchmark-weight exposure neutralization.

The optimizer and quantile evaluator both receive the same final model-factor frames from this bundle. Attribution will call the same preparation function rather than rebuilding its own parallel preprocessing pipeline. Existing public runners continue to accept their current arguments; preparation is an internal shared boundary, not an API break.

## Quantile Portfolio Contract

### Timing and universe

For each pair of consecutive common month-end observations:

- `signal_date` is month end `t`;
- `return_date` is the next available month end `t+1`;
- membership is the KOSPI200 membership known at `signal_date`;
- the signal is the final EMP008 model exposure at `signal_date`;
- total market capitalization for weighting is taken at `signal_date`;
- the realized return is adjusted close at `return_date` divided by adjusted close at `signal_date`, minus one.

A ticker is eligible only when membership is true, signal and both prices are finite, signal-date price is positive, and signal-date total market cap is finite and positive. This common eligibility rule guarantees identical bucket membership for both weighting modes. Nothing from `return_date` other than the ending price is used to construct a portfolio.

### Buckets

For each factor and signal date, eligible tickers are deterministically ranked by `(signal, ticker)` and split as evenly as possible into Q1 through Q5. Q1 always contains the lowest exposures and Q5 the highest exposures.

Rank-based splitting is used instead of raw `qcut` boundaries so duplicate signals cannot collapse buckets and every eligible name is assigned exactly once. If fewer than five names are eligible, the available ordered buckets are populated without duplicating names and unavailable buckets remain empty. The portfolio result records constituent count so sparse periods are visible.

Both weighting modes use exactly the same bucket membership:

- `equal_weight`: every ticker in the bucket receives `1 / n`.
- `market_cap_weight`: positive total market caps within the bucket are normalized to one.

Every non-empty long-only bucket sums to one. Market-cap weighting never falls back silently to equal weighting; a bucket with no positive eligible market cap is marked unavailable.

### Returns and directional spreads

Each Q1-Q5 return is the weighted average constituent return for the holding month. No fees, tax, slippage, or intra-month rebalancing are applied.

Two spreads are retained:

- `high_minus_low`: Q5 minus Q1, preserving the statistical exposure convention.
- `preferred_minus_avoided`: Q5 minus Q1 for `FactorDirection.HIGH`, and Q1 minus Q5 for `FactorDirection.LOW`.

Market-cap factors use `LOW`, representing the small-cap preference. All other current factors use `HIGH`; the retail-flow builder already negates retail buying relative to its sector, so a higher exposure retains the intended direction.

Rank IC is the monthly Spearman correlation between the model exposure and the next-month stock return. Both raw rank IC and direction-adjusted rank IC are saved. Rank IC is independent of portfolio weighting and therefore calculated once per factor/date.

## Results and Metrics

The quantile runner returns an immutable result object with normalized long-form frames:

- monthly portfolio returns;
- non-zero portfolio weights;
- rank IC observations;
- cumulative returns;
- factor/weighting summary metrics.

Summary metrics include:

- observation count;
- annualized return using 12 monthly periods;
- annualized volatility using 12 monthly periods;
- Sharpe ratio using 12 monthly periods and zero risk-free rate;
- maximum drawdown;
- positive-month rate;
- mean monthly return;
- average constituent count;
- average one-way turnover where weights are available;
- mean rank IC, direction-adjusted mean rank IC, IC information ratio, and IC positive rate;
- quantile monotonicity measured from average Q1-Q5 returns in the registered direction.

Metrics are calculated for every factor, weighting mode, Q1-Q5 portfolio, `high_minus_low`, and `preferred_minus_avoided`. Missing periods remain explicit rather than being converted to zero returns.

## Output Layout

`run_full` writes the following under `<run_root>/factor_quantiles/` by default:

```text
factor_quantiles/
  monthly_returns.csv
  monthly_returns.parquet
  portfolio_weights.parquet
  rank_ic.csv
  rank_ic.parquet
  cumulative_returns.csv
  summary.csv
  summary.json
  manifest.json
```

Long-form files contain explicit `factor`, `weighting`, `quantile`, `signal_date`, and `return_date` fields rather than encoded multi-index columns. This keeps downstream factor comparison and parameter searches simple.

The manifest records factor set, factor order, directions, weighting modes, quantile count, date range, timing convention, market-cap field, and artifact paths. The parent `run_summary.json` stores a compact factor-quantile payload with paths and row counts.

`run_full` evaluates quantiles unless `--no-factor-quantiles` is passed. A standalone `run_factor_quantiles` module accepts the common date, parquet, factor-set, and sector taxonomy arguments and writes the same artifact contract without running portfolio optimization or the production backtest.

## Error Handling

- Unknown factor and factor-set strings fail at the enum conversion boundary with supported values in the message.
- Duplicate registry entries and unresolved factor-set members fail during registry initialization.
- Invalid quantile counts and unsupported weighting modes fail before data work begins.
- A completely empty evaluation raises a descriptive error naming the factor set and requested dates.
- Individual sparse factor-months remain in the outputs with empty buckets and counts; they do not abort all other factors.
- Output files are written only after the in-memory result passes schema and portfolio-weight invariants.

## Compatibility and Migration

- Existing factor-set string values and factor output names remain unchanged.
- `MfbtEmp008Config.factor_set` accepts the enum or a compatible string at public boundaries and is normalized once.
- Existing `rank_transform_factors` and `large_bm_neutral_factor_names` configuration fields are removed only after their behavior is represented in registry metadata and regression tests prove parity.
- Existing optimizer output files keep their names and schemas.
- Existing `run_weights` behavior is unchanged; factor-quantile evaluation is attached to `run_full` and the new standalone command.

## Testing Strategy

Implementation follows red-green-refactor tests in this order:

1. Registry uniqueness, factor-set coverage, ordering, and dataset-union tests.
2. Regression tests proving every current factor set builds the same named factors and variant defaults as before.
3. Preprocessing parity tests for rank, value winsor/cap, and large-benchmark neutralization.
4. Deterministic bucket assignment tests covering ties, sparse universes, and invalid returns.
5. Equal-weight and market-cap-weight invariants, including identical bucket membership and signal-date market-cap usage.
6. HIGH/LOW spread-direction and rank-IC direction tests.
7. Monthly metric annualization, drawdown, turnover, monotonicity, and missing-period tests.
8. Artifact schema and manifest tests.
9. `run_full` default integration and `--no-factor-quantiles` opt-out tests.
10. Standalone CLI parser and orchestration tests.
11. Existing EMP008 regression suite, full repository tests, Ruff, and compile checks.

## Acceptance Criteria

The work is complete when:

- every current EMP008 factor is represented exactly once in the typed registry;
- factor-set choices, factor construction, required datasets, preprocessing, and direction derive from registry metadata;
- each selected factor produces Q1-Q5, high-minus-low, preferred-minus-avoided, and rank-IC results;
- equal-weighted and total-market-cap-weighted portfolios are both produced for every selected factor;
- both weighting modes share the same deterministic bucket membership;
- the evaluator uses the exact model exposures consumed by optimization and attribution;
- `run_full` and the standalone command write the documented artifacts;
- existing EMP008 variants and outputs remain compatible;
- automated tests cover the contracts above and all verification commands pass.
