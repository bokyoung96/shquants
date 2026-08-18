# EMP008 Size + Value Measure Comparison Design

## Objective

Compare three EMP008 portfolios while holding the Size factor definition and
portfolio construction settings constant:

1. `ln_market_cap` + `value` (`FCF / TEV`)
2. `ln_market_cap` + `dividend_yield_fy0`
3. `ln_market_cap` + `dividend_yield_ttm`

No momentum, retail-flow, or earnings factor is included. This isolates the
incremental portfolio behavior of each value measure.

## Considered Approaches

### 1. Registered factor sets plus a dedicated comparison runner (selected)

Add three explicit two-factor sets to the EMP008 registry, then run the normal
EMP008 optimizer and backtest pipeline for each set. This reuses the production
data loading, preprocessing, risk model, benchmark-relative optimization, and
performance calculations. The registry entries make every run auditable and
reproducible.

### 2. Build ad hoc factor frames inside a one-off script

This keeps the registry smaller, but duplicates factor selection and dataset
loading rules. It is easier for the experiment to drift from EMP008 behavior,
so it is rejected.

### 3. Compare only standalone factor quantile portfolios

This is faster and useful as a diagnostic, but it does not answer what happens
when EMP008 optimizes the Size/value combination. Quantile evidence can be
included as supporting output, not as the primary result.

## Factor Semantics

- Size remains `ln_market_cap = log(month-end market cap)` and is preferred in
  the low direction.
- FCF/TEV remains `FCF / (market cap + interest-bearing liabilities - quick
  assets)`, with non-positive TEV treated as missing.
- FY0 dividend yield uses `QW_DIVIDEND_YLD_FY0` and retains the existing
  seven-day forward snapshot allowance needed for month-only observations.
- TTM dividend yield remains `QW_DPS_TTM / QW_ADJ_C` at month-end.
- All three sets constrain expected factor alpha to the registered direction,
  matching the existing Origin-style small-Size/high-value interpretation.
- Size preprocessing is identical across the three sets; only the paired value
  measure and its required input datasets differ.

## Execution and Outputs

A dedicated experiment runner executes the three registered factor sets with
the same date range, factor-plus-idiosyncratic risk model, annual tracking-error
budget, no-cost close-fill backtest, benchmark, and capital. It writes all
artifacts under:

`backtesting/strategies/emp008/tests/size_value_measure_comparison/`

Each variant receives its own weights and backtest directory. The comparison
root contains:

- a machine-readable manifest of shared settings and factor definitions;
- a CSV and Excel performance table;
- daily strategy, benchmark, and excess-return series;
- cumulative strategy and excess-return charts;
- active-share summaries;
- a concise Markdown interpretation ranked by excess-return and risk metrics.

The primary judgment metrics are total and annualized excess return, realized
excess volatility, information ratio, maximum drawdown, Sharpe ratio, and mean
active share. Gross no-cost results are the main comparison so that different
factor-induced turnover is not silently mixed with an assumed cost model.

## Error Handling and Reproducibility

- Reject unknown or duplicate variants.
- Record the exact factor IDs, datasets, date range, risk model, tracking-error
  budget, and output paths in the manifest.
- Reuse existing complete variant outputs unless `--force` is supplied.
- Fail if a required return series or benchmark series is missing rather than
  emitting a partial comparison.

## Verification

- Unit tests prove the three registry sets contain exactly Size plus the
  intended value measure and require the expected datasets.
- Runner tests use small fakes to prove isolation, output layout, metric
  aggregation, and ranking without running a full optimization.
- Existing EMP008 registry, pipeline, quantile, and runner tests must continue
  to pass.
- The real three-variant experiment is run against the local parquet catalog;
  the final report is checked for all required artifacts and finite metrics.

## Scope Boundaries

This experiment does not change existing `mfbt`, `adjust`, or Origin factor-set
behavior, does not add dependencies, and does not tune hyperparameters based on
the observed comparison.
