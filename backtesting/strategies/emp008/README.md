# MFBT EMP008 Strategy

EMP008 is a benchmark-relative KOSPI200 portfolio construction pipeline. It
converts six MFBT factor ideas into continuous Barra-style exposures, estimates
rolling factor risk, and solves for long-only target weights relative to
`QW_BM_WEIGHTS`.

This path is separate from `backtesting/strategies/mfbt.py`. The existing MFBT
strategy remains the score and audit surface. EMP008 owns the optimized target
weight generation surface.

## Pipeline

```text
catalog parquet data
  -> raw MFBT factor exposures
  -> float-market-cap preprocessing
  -> sector active exposures
  -> cross-sectional factor return regression
  -> rolling alpha and risk estimates
  -> benchmark-relative active-weight optimization
  -> target_weights.parquet / target_weights.csv bridge
  -> BacktestRunner target_weights.file backtest
  -> reporting CLI tearsheet
  -> costed backtest and gross/costed/BM comparison artifacts
```

## Inputs

The pipeline reads shquants catalog datasets from a parquet directory.

| Role | Dataset |
| --- | --- |
| Return and factor price base | `QW_ADJ_C` |
| Benchmark weights | `QW_BM_WEIGHTS` |
| Earnings momentum | `QW_OP_FWD_12M` |
| Dividend yield | `QW_DPS_TTM` |
| Retail-flow factor | `QW_RETAIL` |
| Sector labels | `QW_WI_SEC_26_BIG` |
| Market cap | `QW_MKTCAP` |
| Preprocessing weight | `QW_MKTCAP_FLT` |
| Value factor | `QW_FCF`, `QW_INT_BEARING_LIAB_NFQ0`, `QW_QUICK_ASSETS_NFQ0` |
| Tradable universe | `QW_K200_YN` |

The baseline `mfbt` set uses all factor inputs above. `adjust` replaces
`price_to_252d_high` with `momentum_12_1m` and omits `retail_flow`, so it does
not load `QW_RETAIL` for factor construction. `origin` loads only the common
portfolio inputs plus `QW_DIVIDEND_YLD_FY0`; `origin_new_dividend` uses
`QW_DPS_TTM` instead. Raw close (`QW_C`) is not part of the EMP008 calculation.

| Factor set | Ordered factors |
| --- | --- |
| `mfbt` | `price_to_252d_high`, `earnings_momentum`, `dividend_yield_ttm`, `retail_flow`, `value`, `ln_market_cap` |
| `adjust` | `momentum_12_1m`, `earnings_momentum`, `dividend_yield_ttm`, `value`, `ln_market_cap` |
| `origin` | `ln_market_cap`, `momentum_12m`, `dividend_yield_fy0` |

`Emp008Config` controls dataset choices and key parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `sector_dataset` | `QW_WI_SEC_26_BIG` | Sector taxonomy for retail-flow grouping and sector constraints |
| `bm_weights_dataset` | `QW_BM_WEIGHTS` | Benchmark weights used as the optimization anchor |
| `universe_dataset` | `QW_K200_YN` | KOSPI200 membership mask |
| `float_market_cap_dataset` | `QW_MKTCAP_FLT` | Weight source for exposure preprocessing |
| `retail_flow_lookback_days` | `252` | Rolling retail-flow window |
| `rank_transform_factors` | `("ln_market_cap",)` | Factors ranked cross-sectionally after missing-value fill |
| `large_bm_neutral_factor_names` | `("ln_market_cap",)` | Factor exposures neutralized for large benchmark constituents |
| `large_bm_neutral_weight_threshold` | `0.10` | Benchmark-weight cutoff for large-constituent neutralization |
| `risk_window` | `36` | Rolling monthly factor-risk window |
| `tracking_error` | `0.007 / sqrt(12)` | Monthly active-risk budget |
| `risk_model` | `factor_idio` | TE covariance model: `factor_idio` or `direct_covariance` |
| `expected_alpha_policy` | `mean` | Optional Origin-style directional guard applied after the 36-month mean |
| `factor_timing` | `None` | Optional factor-weight timing policy; disabled unless explicitly enabled |

Factor-set choices and dataset loading are registry-derived. `run_weights.py`,
`run_full.py`, and `run_factor_quantiles.py` all expose the same
`FactorSetId` values, and `required_datasets()` derives its parquet inputs from
the registered factor definitions instead of maintaining duplicated CLI lists.

`QW_BM_WEIGHTS` is used without modification from its first positive row
onward. When the requested 36-month warmup reaches dates before the official
series begins, EMP008 uses contemporaneous `QW_MKTCAP_FLT` normalized within
`QW_K200_YN` as a benchmark proxy. This proxy is limited to the missing prefix;
it is not allowed to replace or fill official benchmark rows.

## Registry Extension Contract

New factors are added through `factor_registry.py`, not by appending
ad hoc branches to individual runners. Each `FactorDefinition` describes one
strategy-independent calculation and must declare:

- `id`: stable public factor name used in outputs
- `builder`: raw monthly factor constructor
- `datasets`: extra parquet dependencies beyond the common EMP008 inputs
- `direction`: `HIGH` when larger exposure is preferred, `LOW` when smaller is preferred
- `rank_transform`: whether the post-fill cross section is ranked before z-scoring
- `winsor_config_attr`: optional `Emp008Config` attribute used to winsorize raw values
- `zscore_cap_config_attr`: optional `Emp008Config` attribute used to cap final z-scores
- `requires_construction_sector`: whether raw construction needs `sector_dataset`

`FactorSetDefinition` then derives:

- ordered factor membership for each CLI variant
- strategy-specific large-benchmark-weight neutralization targets
- optimizer direction constraints
- forward snapshot allowance for month-only data

That one registry drives builder selection, dataset requirements, CLI
`--factor-set` choices, preprocessing policy, and quantile manifest metadata.

## Raw Factors

All factor outputs are monthly `date x ticker` exposure panels. They are raw
continuous values, not score buckets.

| Factor | Definition |
| --- | --- |
| `price_to_252d_high` | `adjusted_close / adjusted_close.rolling(252).max()` |
| `positivity_momentum` | Rolling share of non-negative daily returns |
| `momentum_12m` | Month-end adjusted-close return over 12 months |
| `momentum_12_1m` | Prior month-end adjusted close divided by the 12-month-lag close, minus one |
| `earnings_momentum` | Monthly forward OP growth: `(current - previous) / abs(previous)` |
| `dividend_yield_ttm` | `DPS_TTM / adjusted_close` |
| `dividend_yield_fy0` | Fiscal-year-zero dividend yield snapshot |
| `retail_flow` | Negative sector-relative 252-day retail flow |
| `value` | `FCF / (market_cap + interest_bearing_liability - quick_assets)` |
| `ln_market_cap` | `log(market_cap)` |

Special handling:

- `earnings_momentum` resets extreme positive growth to `0.0` when current OP is
  below `low_op_threshold`.
- `value` treats non-positive TEV as missing.
- `ln_market_cap` is filled with the same float-market-cap weighted rule as
  other factors, then rank-transformed before centering and z-scoring.
- MFBT factor sets then set `ln_market_cap` to neutral exposure `0.0` for
  stocks whose `QW_BM_WEIGHTS` weight is at least `10%` on that date. Origin
  sets use the same independent factor without that strategy policy.
- `retail_flow` is calculated stock by stock, then de-meaned within each sector
  on the rebalance date and multiplied by `-1`. The resulting signal is sector
  neutral by construction before the common preprocessing step.

## Optimization

The optimizer solves active weights around `QW_BM_WEIGHTS`.

### Notation

For rebalance month `t`:

| Symbol | Meaning |
| --- | --- |
| `w_bm` | Benchmark weights from `QW_BM_WEIGHTS` |
| `x` | Active weights to solve |
| `w = w_bm + x` | Final target weights |
| `Z_t` | Stock-by-factor exposure matrix at the target rebalance date |
| `f_t` | Realized monthly factor returns estimated by cross-sectional regression |
| `e_t` | Regression residual, interpreted as stock-specific return |
| `a` | Expected factor alpha vector |
| `M` | Stock-level active-risk covariance matrix used in the TE constraint |

The solved target is always benchmark-relative. The optimizer does not choose a
standalone long-only portfolio from zero. It chooses `x`, then adds it to the
benchmark.

### Factor Return Regression

For each monthly interval `t-1 -> t`, EMP008 first measures stock returns and
subtracts the benchmark-weighted stock return for that month:

```text
stock_excess_return_t = stock_return_t - sum(w_bm_t * stock_return_t)
```

It then runs a cross-sectional regression using exposures known at `t-1`:

```text
stock_excess_return_t = Z_(t-1) * f_t + e_t
```

This produces one realized return per factor plus a residual per stock. These
monthly `f_t` and `e_t` observations are accumulated through the warmup period.
Optimization starts only after `risk_window = 36` monthly factor-return
observations are available. The loader retrieves those observations before the
requested output period. For example, a run requested from December 2019 uses
the preceding history and produces its first optimized portfolio on the last
2019 trading day, `2019-12-30`.

### Expected Alpha

Expected alpha is factor-based, not a direct stock-return forecast.

```text
a = mean(last 36 monthly f_t)
sector factor alpha = 0
stock_alpha_t = Z_t * a
objective = maximize stock_alpha_t' * x
```

The expected-alpha estimator is optional. The default `mean` keeps the
36-month arithmetic mean above. `--expected-alpha-estimator ewma36` instead
applies `ewm(span=36, adjust=True)` to those same trailing 36 monthly factor
returns. `--expected-alpha-estimator mean_1se` shrinks each 36-month arithmetic
mean toward zero by one standard error:

```text
sample_std = std(last 36 monthly f_t, ddof=1)
standard_error = sample_std / sqrt(number of valid observations)
adjusted_alpha = sign(mean) * max(abs(mean) - standard_error, 0)
```

This subtracts one standard error of the estimated mean, not one monthly
standard deviation. Factor weights, covariance estimation, tracking error,
sector neutrality, and costs are unchanged.

The six alpha factors contribute to `a`. Sector dummy factors are still present
in `Z_t`, but their expected alpha is forced to zero. They are included so the
same exposure matrix can both explain returns and enforce sector active
neutrality.

The `mfbt_origin_smallcap` variant applies Origin's directional policy after
the rolling mean is estimated. It does not alter raw factors, z-scores,
realized factor returns, or the risk model:

```text
price_to_252d_high, earnings_momentum, dividend_yield_ttm, retail_flow, value:
    adjusted alpha = max(mean(last 36 monthly f_t), 0)
ln_market_cap:
    adjusted alpha = min(mean(last 36 monthly f_t), 0)
```

A disabled factor therefore contributes no expected return to the objective,
while its exposure and covariance remain in the TE risk calculation.

`ln_market_cap` has one additional guardrail: if a stock has benchmark weight at
least `10%`, its `ln_market_cap` exposure is set to `0.0` for that date. That
makes the large benchmark constituent neutral to the market-cap factor's alpha
signal without removing the stock from other factors or from the risk matrix.

### Optional Factor Momentum Timing

Factor timing is opt-in. The default `factor_timing=None` and CLI
`--factor-timing none` preserve the existing fixed factor weights and output
surface. Use `--factor-timing momentum` to adjust only the alpha-factor weights
at each rebalance; factor construction, expected-alpha estimation, sector
neutrality, and the risk model remain unchanged.

For each factor, the timing policy compounds its realized monthly factor
returns over 6- and 12-month windows. A `LOW` factor such as `ln_market_cap` is
sign-reversed first so that a positive directional return always means the
registered preferred side performed well.

| 6-month directional return | 12-month directional return | State | Multiplier |
| --- | --- | --- | --- |
| Positive | Positive | `strong` | `1.25` |
| Negative | Negative | `weak` | `0.75` |
| Otherwise, including zero | Otherwise | `neutral` | `1.00` |

The base weights are multiplied by these values and normalized to sum to one.
Only factor-return observations strictly earlier than the target rebalance date
are eligible, so the factor return stamped with the current rebalance date is
not used as a timing signal. Before 12 eligible observations exist, the policy
records `insufficient_history` and leaves normalized base weights unchanged.

Enabled runs write `weights/factor_timing.csv` and
`weights/factor_timing.parquet`. Each row records the rebalance date, factor,
direction, base weight, 6- and 12-month returns, state, multiplier, final timed
weight, and last signal date. Disabled runs do not create these artifacts.

### Default Risk Model: Factor Plus Idio

The default `risk_model = factor_idio` builds the TE covariance matrix as:

```text
F = Cov(last 36 monthly factor returns)
D = diag(last 36 monthly residual variance by stock)
M = Z_t * F * Z_t' + D
tracking_error = sqrt(x' * M * x)
```

This is the same active-risk structure as the MATLAB prototype:

```text
M = D + z * cov * z'
c = x' * M * x - TE^2 <= 0
```

The split matters because the optimizer sees common factor risk and stock-
specific risk separately before they are recombined into the stock-level matrix
`M`. Mathematically the final TE constraint is still one quadratic form, but the
estimation is different:

- Factor covariance `F` is estimated from a small number of factor-return series,
  so it is more stable with a 36-month window.
- Residual risk `D` is diagonal, so stock-specific noise does not create
  unstable pairwise correlations from only 36 observations.
- The resulting `M` has a controlled structure and prevents the optimizer from
  overusing poorly estimated stock covariance relationships.

### Direct Covariance Experiment

For comparison, `risk_model = direct_covariance` keeps the exact same alpha,
benchmark, long-only, sector-neutral, and TE settings, but replaces the risk
matrix with a direct 36-month covariance of stock excess returns:

```text
M = Cov(last 36 monthly stock_excess_return)
tracking_error = sqrt(x' * M * x)
```

Because there are roughly 200 stocks and only 36 monthly observations, the direct
sample covariance is low-rank and much less stable. Missing stock return
observations are filled with `0.0` before covariance calculation so the matrix
remains positive semi-definite; a tiny diagonal ridge is added inside the
optimizer for numerical stability.

### Constraints

```text
maximize expected_alpha_exposure.T @ active_weight

sum(active_weight) = 0
sector_active_exposure.T @ active_weight = 0
active_weight.T @ covariance @ active_weight <= tracking_error^2
final_weight = benchmark_weight + active_weight >= 0
```

The final result is long-only, fully invested, benchmark-relative, and sector
active neutral under the configured sector exposure model.

For the standard 70bp annual TE run, the monthly constraint is:

```text
tracking_error = 0.007 / sqrt(12)
```

This matches the MATLAB convention `TE = 0.7 / sqrt(12)` when the MATLAB inputs
are expressed in percent units.

### Risk Model Comparison Result

The risk model comparison was run with the compatibility wrapper scripts:

```powershell
uv run python scripts\run_mfbt_emp008_full.py `
  --name mfbt_emp008_70bp_36m_retail_rel `
  --start 2020-01-31 `
  --tracking-error-annual 0.007

uv run python scripts\run_mfbt_emp008_full.py `
  --name mfbt_emp008_70bp_36m_retail_rel_direct_cov `
  --start 2020-01-31 `
  --tracking-error-annual 0.007 `
  --risk-model direct_covariance
```

Both runs use the same weights/backtest/report/comparison pipeline and the same
cost assumptions. Results are saved under
`experiment_results/risk_model_comparison/` so this one experiment can be
reviewed in git even though normal `results/` output is ignored.

| Risk model | Gross total excess bp | Gross annual excess bp | Gross realized excess vol bp | Gross IR | Net total excess bp | Net annual excess bp | Mean active share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `factor_idio` | 316.69 | 95.11 | 127.06 | 0.75 | 224.68 | 67.69 | 7.52% |
| `direct_covariance` | 83.84 | 25.38 | 418.24 | 0.06 | -353.29 | -108.61 | 41.41% |

Interpretation:

- The default factor-plus-idio model produced better excess return and much
  lower realized excess volatility in this sample.
- Direct covariance used the same ex-ante monthly TE limit, but its realized
  excess volatility was much higher because the 36-month stock covariance matrix
  is not a stable estimate for a roughly 200-stock universe.
- Direct covariance also allowed much larger active weights: mean active share
  was `41.41%` versus `7.52%`. This is a sign that the optimizer found directions
  that looked low-risk in the short sample covariance but were not low-risk out
  of sample.
- The conclusion is not that direct covariance is impossible, but that the plain
  36-month sample covariance is not the right production risk estimator here.
  A direct stock covariance approach would need shrinkage, a longer window,
  factor structure, or other regularization before it should be preferred.

## Factor Attribution

Factor attribution is a post-process on saved EMP008 weights. It does not rerun
the backtest. For each rebalance month `t`, it decomposes the next rebalance
period's active return using the same factor model language as the optimizer:

```text
factor contribution = active factor exposure at t * realized factor return over t -> t+1
specific contribution = active_weight at t * cross-sectional regression residual over t -> t+1
model active return = alpha factor contribution + sector contribution + specific contribution
```

Primary interpretation should focus on the six alpha factors plus `specific`.
Sector factors are retained in the workbook for reconciliation, but their
expected alpha is constrained to zero in optimization and their active
contribution should be interpreted as a constraint/model residual rather than a
standalone alpha signal.

## Factor Quantile Diagnostics

EMP008 also exposes a standalone single-factor diagnostic surface through
`run_factor_quantiles.py`, and `run_full.py` runs it by default after the
optimizer and before the optional costed comparison backtest.

The evaluator uses the same prepared bundle as optimization and attribution, so
all three surfaces share identical monthly factor exposures, benchmark
completion, sector preprocessing, and factor-set semantics.

### Timing And Eligibility

- Signal dates are EMP008 rebalance month-ends drawn from the prepared monthly panel.
- Eligibility is limited to stocks inside the KOSPI200 universe on the signal date.
- Membership and any market-cap weighting are determined on the signal date only.
- Returns are measured from month-end `t` to the next month-end `t+1`; no return-date information is used for portfolio formation.
- For Origin's forward-snapshot case, factor preparation still honors the
  registered next-month allowance before the common month-end panel is built.

### Portfolio Construction

Each factor is evaluated in `q` quantile buckets, default `q=5`, with two
weighting modes over the same exact bucket membership:

- `equal_weight`: every stock in the bucket receives `1 / n`
- `market_cap_weight`: every stock is weighted by signal-date total market cap
  from `QW_MKTCAP`, normalized within the bucket

Direction-aware spreads are written alongside the raw quantiles:

- `high_minus_low`: `Qq - Q1` using the raw bucket order
- `preferred_minus_avoided`: best-direction bucket minus worst-direction bucket
  using the registered factor direction (`HIGH` prefers `Qq`, `LOW` prefers `Q1`)

Rank IC is reported in both raw and directional form. `directional_rank_ic`
equals raw Spearman rank IC for `HIGH` factors and its negation for `LOW`
factors.

### Diagnostic Limits

The quantile artifacts are research diagnostics only:

- no transaction costs
- no sector-neutral portfolio construction inside the quantile buckets
- no automatic factor-weight optimization

They are intended to put single-factor evidence beside the optimizer, not to
replace the production benchmark-relative portfolio construction path.

## Run Files

Use the top-level wrappers under `scripts/` for normal execution:

| Wrapper | Implementation | Purpose |
| --- | --- | --- |
| `scripts/run_mfbt_emp008_weights.py` | `run_weights.py` | Generate target weights only |
| `scripts/run_mfbt_emp008_backtest.py` | `run_backtest.py` | Backtest existing weights and optionally create a report |
| `scripts/run_mfbt_emp008_full.py` | `run_full.py` | Generate weights, backtest, report, and comparison artifacts in one command |
| `python -m backtesting.strategies.emp008.run_factor_quantiles` | `run_factor_quantiles.py` | Run standalone factor quantile diagnostics |

There is no separate `cli_common.py`. Shared runner helpers live in the concrete
runner modules that use them:

- weights/config/output helpers: `run_weights.py`
- backtest spec and summary helpers: `run_backtest.py`
- full orchestration: `run_full.py`

The implementation package itself uses neutral module names:
`strategy.py`, `data.py`, `factors.py`, `factor_builders.py`,
`factor_pipeline.py`, `factor_quantiles.py`, `factor_registry.py`, `factor_timing.py`,
`optimize.py`, `preprocess.py`, `risk.py`, and `experiments/`.
The retained `mfbt_emp008` naming in wrapper scripts, factor-set values, and
default run/output names identifies the real MFBT variant rather than the
shared package code.

## Size + Value Measure Comparison

The isolated value-measure experiment holds Size (`ln_market_cap`) constant and
runs three two-factor EMP008 portfolios: Size + FCF/TEV, Size + FY0 dividend
yield (`QW_DIVIDEND_YLD_FY0`), and Size + TTM dividend yield
(`QW_DPS_TTM / QW_ADJ_C`). Momentum and the other MFBT factors are excluded.

```powershell
uv run python -m backtesting.strategies.emp008.experiments.size_value_measure_comparison `
  --start 2020-01-31 `
  --tracking-error-annual 0.007 `
  --risk-model factor_idio
```

Results are written to
`backtesting/strategies/emp008/tests/size_value_measure_comparison/`. The root
contains a manifest, ranked CSV/XLSX performance tables, aligned daily returns,
the combined `performance_dashboard.png`, cumulative return and excess-return
charts, and `interpretation.md`. All three
portfolios use the same no-cost close-fill backtest settings and common date
range.

## Recommended Runs

Generate weights once:

```powershell
uv run python scripts\run_mfbt_emp008_weights.py `
  --start 2020-01-31 `
  --name mfbt_emp008
```

Run a no-cost close-fill backtest and report from those weights:

```powershell
uv run python scripts\run_mfbt_emp008_backtest.py `
  --weights-name mfbt_emp008 `
  --name mfbt_emp008 `
  --fill-mode close
```

Run another backtest from the same weights with cost assumptions:

```powershell
uv run python scripts\run_mfbt_emp008_backtest.py `
  --weights-name mfbt_emp008 `
  --name mfbt_emp008 `
  --fill-mode close `
  --fee 0.0002 `
  --sell-tax 0.0015 `
  --slippage 0.0005
```

Run weights, backtest, report, and comparison artifacts in one command:

```powershell
uv run python scripts\run_mfbt_emp008_full.py `
  --start 2020-01-31 `
  --name mfbt_emp008
```

Run the same pipeline with optional factor momentum timing:

```powershell
uv run python scripts\run_mfbt_emp008_full.py `
  --start 2020-01-31 `
  --name mfbt_emp008_factor_momentum `
  --factor-timing momentum
```

Run the standalone factor quantile diagnostics:

```powershell
python -m backtesting.strategies.emp008.run_factor_quantiles --factor-set mfbt --start 2020-01-31 --end 2026-06-30
```

Run all registered single-factor diagnostics with 5-quantile subplot outputs:

```powershell
python -m backtesting.strategies.emp008.run_factor_quantiles `
  --factor-set all_factors `
  --quantiles 5 `
  --start 2020-01-31 `
  --end 2026-06-30 `
  --output-dir results\emp008_factor_quantiles\all_factors
```

Run the full pipeline with explicit quantile bucket count:

```powershell
python -m backtesting.strategies.emp008.run_full --factor-set mfbt --factor-quantiles 5 --end 2026-06-30
```

The backtest-only runner accepts `--capital`, `--fill-mode`, `--fee`,
`--sell-tax`, `--slippage`, `--no-fractional`, `--start`, `--end`, and
`--no-report`. Reusing one weights run is faster when only execution assumptions
change.

The full runner also creates a second costed backtest for comparison by default.
The default comparison costs are `--comparison-fee 0.0002`,
`--comparison-sell-tax 0.0015`, and `--comparison-slippage 0.0005`. Pass
`--no-comparison` to skip this stage. Factor quantiles default to `5` buckets;
pass `--factor-quantiles <int>` to change the count or
`--no-factor-quantiles` to skip this diagnostic stage.

If a saved backtest run already exists, build a report directly:

```powershell
uv run python -m backtesting.reporting.cli `
  --runs <saved_run_id> `
  --name mfbt_emp008 `
  --kind tearsheet `
  --title "MFBT EMP008"
```

## Artifacts

Default EMP008 run outputs are grouped under `results/emp008_runs/<name>/`.

| Path | Purpose |
| --- | --- |
| `weights/target_weights.csv` | CSV bridge consumed by `BacktestRunner` |
| `weights/target_weights.parquet` | Primary optimized target weights |
| `weights/active_weights.parquet` | Active weights versus benchmark |
| `weights/active_share.csv` | Monthly active share from active weights |
| `weights/diagnostics.parquet` | Solver success and constraint diagnostics |
| `weights/factor_timing.csv` and `.parquet` | Optional rebalance-by-factor momentum state and timed weights |
| `weights/weights_export.xlsx` | Review-friendly Excel export |
| `backtests/<run_id>/` | Saved `BacktestRunner` output for the run |
| `backtests/<run_id>/series/active_share.csv` | Monthly active share copied into the saved backtest |
| `reports/<name>/report.html` | Static tearsheet report |
| `comparison/performance.xlsx` | Gross, costed, benchmark, excess, drawdown, and active-weight data |
| `comparison/cumulative_excess_drawdown.png` | Cumulative return with cumulative excess fill and drawdown |
| `comparison/monthly_excess_heatmap.png` | Gross and costed monthly excess-return heatmap |
| `comparison/active_weight_sum.*` | Monthly `sum(abs(active weight))` data and chart |
| `comparison_summary.json` | Comparison-stage summary |
| `factor_quantiles/monthly_returns.csv` and `.parquet` | Long-form monthly return observations by signal date, return date, factor, weighting, and portfolio |
| `factor_quantiles/portfolio_weights.parquet` | Long-form bucket holdings with signal date, return date, factor, weighting, quantile, ticker, and weight |
| `factor_quantiles/rank_ic.csv` and `.parquet` | Monthly raw and directional rank IC by factor and signal date |
| `factor_quantiles/cumulative_returns.csv` | Cumulative return path by signal date, return date, factor, weighting, and portfolio |
| `factor_quantiles/daily_cumulative_returns.csv` and `.parquet` | Daily NAV path from fixed-share holdings between monthly rebalances |
| `factor_quantiles/cumulative_quintiles_equal_weight.png` | Equal-weight daily NAV subplots for each factor with preferred-minus-avoided spread |
| `factor_quantiles/cumulative_quintiles_market_cap_weight.png` | Market-cap-weight daily NAV subplots for each factor with preferred-minus-avoided spread |
| `factor_quantiles/summary.csv` and `.json` | Portfolio-level annualized return, volatility, Sharpe, drawdown, turnover, IC, and monotonicity metrics |
| `factor_quantiles/manifest.json` | Registry-derived factor order, direction metadata, weighting modes, timing contract, and row counts |
| `factor_attribution/factor_attribution.xlsx` | Monthly factor contribution, exposure, return, and reconciliation data |
| `factor_attribution/*.png` | Cumulative, monthly heatmap, and yearly factor-contribution charts |
| `factor_attribution_summary.json` | Factor-attribution-stage summary |
| `logs/*.log` | Stage timing and summary logs |
| `weights_summary.json` | Weights-only runner summary |
| `backtest_summary.json` | Backtest-only runner summary |
| `run_summary.json` | Full runner summary |

Saved backtest runs and reports are written inside `results/emp008_runs/<name>/`
by default. Pass `--backtests-root` or `--reports-root` only when a global output
root is explicitly desired.

The quantile artifacts use long-form schemas so they can be audited without
reconstructing hidden pivots:

- `monthly_returns`: `signal_date`, `return_date`, `factor`, `weighting`,
  `portfolio`, `return`, `constituent_count`
- `portfolio_weights`: `signal_date`, `return_date`, `factor`, `weighting`,
  `quantile`, `ticker`, `weight`
- `rank_ic`: `signal_date`, `return_date`, `factor`, `rank_ic`,
  `directional_rank_ic`, `n_obs`
- `daily_cumulative_returns`: `signal_date`, `date`, `factor`, `weighting`,
  `portfolio`, `cumulative_return`. Membership and initial weights are set monthly;
  daily points value those fixed-share holdings without daily re-ranking.
- `cumulative_returns`: `signal_date`, `return_date`, `factor`, `weighting`,
  `portfolio`, `cumulative_return`
- `summary`: `factor`, `weighting`, `portfolio`, plus annualization, drawdown,
  turnover, IC, and monotonicity fields

## Current Issues And Gaps

- Full multi-year runs are slow. The SLSQP optimizer runs once per rebalance
  month with roughly the full benchmark universe, so long windows can exceed a
  short command timeout.
- The generated primary output is parquet, while `target_weights.file` currently
  reads CSV only. The runner writes `target_weights.csv` as a bridge.
- Report generation works from a saved `BacktestRunner` run, not directly from
  `Emp008Result`.
- Local "latest" is bounded by the slowest required parquet input. Some datasets
  may be newer than others, but EMP008 should use the common available date.
- The risk model is intentionally simple: plain cross-sectional least squares,
  rolling sample covariance, diagonal residual variance, and no shrinkage.
- The optimizer uses median residual-variance fallback for new benchmark
  entrants. This is conservative enough for continuity, but it is still a model
  assumption that should be reviewed before production use.
- Factor-return warmup before the first official `QW_BM_WEIGHTS` row uses a
  normalized float-market-cap proxy. Official benchmark weights remain the
  optimization anchor as soon as they are available.
- Sector neutrality depends on the configured sector dataset and float-market-cap
  weights. Changes to sector taxonomy can change active constraints.
- The supported package path uses the neutral modules listed above, while the
  `scripts/run_mfbt_emp008_*.py` wrappers remain for compatibility.

## Verification

```powershell
uv run pytest tests/scripts/test_run_emp008_full.py -q
uv run pytest tests/ingest/test_pipeline.py tests/catalog/test_groups.py tests/data/test_loader.py -q
```
