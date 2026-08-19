# EMP008 Trimmed-Mean Expected Alpha Design

## Goal

Improve the costed, walk-forward stability of EMP008 without changing its four
alpha factors, WICS sector neutrality, risk model, tracking-error budget, or
optimizer. The only strategy change is to make the rolling expected factor
return less sensitive to a few extreme monthly factor returns.

The fixed factor set is:

- `ln_market_cap` (preferred direction: low)
- `momentum_12m` (preferred direction: high)
- `earnings_momentum` (preferred direction: high)
- `value` (preferred direction: high)

## Non-Goals

- Do not add or replace alpha factors.
- Do not use t-statistics, standard errors, macro variables, market-regime
  variables, or stock-level forecasts.
- Do not change the `factor_idio` risk model, its 36-month covariance window,
  monthly rebalancing, the annual 70 bp tracking-error budget, or transaction
  cost assumptions.
- Do not change WICS sector neutrality or introduce WI26 as an alternative in
  this experiment.
- Do not add a second dynamic factor-weight multiplier derived from the same
  expected-return estimate. That would count the same information twice.
- Do not tune the trimming fraction over the evaluation period.

## Expected-Alpha Estimator

Add one explicit estimator named `trimmed_mean_10pct`.

At each rebalance date and for each factor-return column:

1. Select the most recent 36 monthly observations, exactly as the current
   `mean` estimator does.
2. Sort the observations.
3. Remove the three smallest and three largest observations. This is a fixed
   10% cut from each tail for a 36-observation window because
   `floor(36 * 0.10) = 3`.
4. Take the arithmetic mean of the remaining 30 observations.

For factor `i`, with ordered observations
`f_(1), ..., f_(36)`, the estimate is:

```text
trimmed_mean_i = mean(f_(4), ..., f_(33))
```

The estimator must not winsorize the retained data and must not rescale the
result. It uses no standard-error or volatility penalty.

Sector factor expected alphas remain exactly zero after estimation.

## Direction Policy And Factor Weights

The existing direction policy remains unchanged:

- A positive trimmed mean is eligible for high-preference factors.
- A negative trimmed mean is eligible for the low-preference size factor.
- An estimate pointing against the registered preferred direction is set to
  zero.

The existing factor weights are then multiplied by the trimmed expected alpha
once, using the current `apply_factor_weights` path. The trimmed mean must not
also be converted into a separate dynamic weight because that would make the
effective objective coefficient approximately quadratic in the same estimate.

This means EMP008 remains dynamic through its rolling expected-alpha vector;
the change concerns the robustness of that vector, not a new timing layer.

## Integration Surface

Extend the existing expected-alpha estimator option rather than creating a
parallel strategy path:

- `Emp008Config.expected_alpha_estimator` accepts `trimmed_mean_10pct`.
- `compute_expected_alpha` implements the estimator.
- The weights and full-run CLIs accept
  `--expected-alpha-estimator trimmed_mean_10pct`.
- Run summaries record the selected estimator as they do today.
- The README documents the exact tail-removal rule and distinguishes it from
  `mean_1se`.

No new dependency is required.

## Comparison Experiment

Use the existing WICS four-factor grid-search candidates and hold every other
condition constant. Run the same nine factor-weight combinations with
`trimmed_mean_10pct`, then pair each candidate with its saved `mean` baseline.

The comparison must include costed results and save:

- overall CAGR, cumulative excess return, information ratio, maximum drawdown,
  and average turnover;
- paired deltas for each of the nine factor-weight candidates;
- annual excess-return deltas;
- optimizer success and constraint diagnostics;
- a concise interpretation distinguishing broad improvement from gains driven
  by one candidate or one year.

The primary evidence is the median information-ratio delta across the nine
paired candidates. Breadth is reported as the number of candidates with a
positive information-ratio delta. Annual breadth and turnover changes are
secondary diagnostics, not additional parameters used to select the estimator.

The historical comparison is a rolling, no-lookahead backtest at each rebalance,
but it is not an untouched out-of-sample test because the estimator design was
chosen after inspecting earlier EMP008 results. Reports must retain that limit.

## Edge Cases

- The optimizer already requires 36 monthly factor-return rows before running,
  so the normal estimator input has exactly 36 observations per factor after
  the existing factor-return frame is completed.
- If a column nevertheless has fewer than seven valid observations, fail with a
  clear error rather than returning an empty-slice mean.
- If all four direction-adjusted alpha estimates are zero, return the benchmark
  portfolio explicitly or preserve the current zero-objective solution only if
  tests prove it is exactly the benchmark.
- Sector expected alphas must remain zero regardless of their trimmed estimates.

## Tests

Add focused unit coverage that proves:

- three observations are removed from each tail of a 36-value series;
- a single extreme positive or negative observation does not dominate the
  trimmed estimate;
- high- and low-preference direction policies still zero the wrong sign;
- sector expected alpha remains zero;
- the estimator is selectable through config and both CLIs;
- existing `mean`, `ewma36`, and `mean_1se` behavior remains unchanged;
- an all-zero alpha vector produces benchmark weights;
- the WICS grid comparison pairs exactly the same nine candidates and dates.

Minimum verification after implementation:

```powershell
uv run pytest tests/strategies/test_emp008_factor_weights.py -q
uv run pytest tests/strategies/test_emp008_factor_weight_grid_search.py -q
uv run pytest tests/strategies/test_emp008_factor_weight_grid_comparison.py -q
```

After unit verification, run the WICS nine-candidate comparison and inspect the
saved aggregate, paired, yearly, turnover, and optimizer-diagnostic artifacts.
