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
| Benchmark-weight source support | `QW_C` |
| Benchmark weights | `QW_BM_WEIGHTS` |
| Earnings momentum | `QW_OP_FWD_12M` |
| Dividend yield | `QW_DPS_TTM` |
| Retail-flow factor | `QW_RETAIL` |
| Sector labels | `QW_WI_SEC_26_BIG` |
| Market cap | `QW_MKTCAP` |
| Preprocessing weight | `QW_MKTCAP_FLT` |
| Value factor | `QW_FCF`, `QW_INT_BEARING_LIAB_NFQ0`, `QW_QUICK_ASSETS_NFQ0` |
| Tradable universe | `QW_K200_YN` |

`MfbtEmp008Config` controls dataset choices and key parameters:

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

## Raw Factors

All factor outputs are monthly `date x ticker` exposure panels. They are raw
continuous values, not score buckets.

| Factor | Definition |
| --- | --- |
| `price_momentum` | `adjusted_close / adjusted_close.rolling(252).max()` |
| `earnings_momentum` | Monthly forward OP growth: `(current - previous) / abs(previous)` |
| `dividend_yield` | `DPS_TTM / adjusted_close` |
| `retail_flow` | Negative sector-relative 252-day retail flow |
| `value` | `FCF / (market_cap + interest_bearing_liability - quick_assets)` |
| `ln_market_cap` | `log(market_cap)` |

Special handling:

- `earnings_momentum` resets extreme positive growth to `0.0` when current OP is
  below `low_op_threshold`.
- `value` treats non-positive TEV as missing.
- `ln_market_cap` is filled with the same float-market-cap weighted rule as
  other factors, then rank-transformed before centering and z-scoring.
- `ln_market_cap` is then set to neutral exposure `0.0` for stocks whose
  `QW_BM_WEIGHTS` weight is at least `10%` on that date.
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
observations are available. With the current data, the requested run starts from
`2020-01-31`, but the first optimized portfolio is `2022-12-29`.

### Expected Alpha

Expected alpha is factor-based, not a direct stock-return forecast.

```text
a = mean(last 36 monthly f_t)
sector factor alpha = 0
stock_alpha_t = Z_t * a
objective = maximize stock_alpha_t' * x
```

The six alpha factors contribute to `a`. Sector dummy factors are still present
in `Z_t`, but their expected alpha is forced to zero. They are included so the
same exposure matrix can both explain returns and enforce sector active
neutrality.

`ln_market_cap` has one additional guardrail: if a stock has benchmark weight at
least `10%`, its `ln_market_cap` exposure is set to `0.0` for that date. That
makes the large benchmark constituent neutral to the market-cap factor's alpha
signal without removing the stock from other factors or from the risk matrix.

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

The risk model comparison was run with:

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

## Run Files

Use the top-level wrappers under `scripts/` for normal execution:

| Wrapper | Implementation | Purpose |
| --- | --- | --- |
| `scripts/run_mfbt_emp008_weights.py` | `run_weights.py` | Generate target weights only |
| `scripts/run_mfbt_emp008_backtest.py` | `run_backtest.py` | Backtest existing weights and optionally create a report |
| `scripts/run_mfbt_emp008_full.py` | `run_full.py` | Generate weights, backtest, report, and comparison artifacts in one command |

There is no separate `cli_common.py`. Shared runner helpers live in the concrete
runner modules that use them:

- weights/config/output helpers: `run_weights.py`
- backtest spec and summary helpers: `run_backtest.py`
- full orchestration: `run_full.py`

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

The backtest-only runner accepts `--capital`, `--fill-mode`, `--fee`,
`--sell-tax`, `--slippage`, `--no-fractional`, `--start`, `--end`, and
`--no-report`. Reusing one weights run is faster when only execution assumptions
change.

The full runner also creates a second costed backtest for comparison by default.
The default comparison costs are `--comparison-fee 0.0002`,
`--comparison-sell-tax 0.0015`, and `--comparison-slippage 0.0005`. Pass
`--no-comparison` to skip this stage.

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
| `weights/weights_export.xlsx` | Review-friendly Excel export |
| `backtests/<run_id>/` | Saved `BacktestRunner` output for the run |
| `backtests/<run_id>/series/active_share.csv` | Monthly active share copied into the saved backtest |
| `reports/<name>/report.html` | Static tearsheet report |
| `comparison/performance.xlsx` | Gross, costed, benchmark, excess, drawdown, and active-weight data |
| `comparison/cumulative_excess_drawdown.png` | Cumulative return with cumulative excess fill and drawdown |
| `comparison/monthly_excess_heatmap.png` | Gross and costed monthly excess-return heatmap |
| `comparison/active_weight_sum.*` | Monthly `sum(abs(active weight))` data and chart |
| `comparison_summary.json` | Comparison-stage summary |
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

## Current Issues And Gaps

- Full multi-year runs are slow. The SLSQP optimizer runs once per rebalance
  month with roughly the full benchmark universe, so long windows can exceed a
  short command timeout.
- The generated primary output is parquet, while `target_weights.file` currently
  reads CSV only. The runner writes `target_weights.csv` as a bridge.
- Report generation works from a saved `BacktestRunner` run, not directly from
  `MfbtEmp008Result`.
- Local "latest" is bounded by the slowest required parquet input. Some datasets
  may be newer than others, but EMP008 should use the common available date.
- The risk model is intentionally simple: plain cross-sectional least squares,
  rolling sample covariance, diagonal residual variance, and no shrinkage.
- The optimizer uses median residual-variance fallback for new benchmark
  entrants. This is conservative enough for continuity, but it is still a model
  assumption that should be reviewed before production use.
- Sector neutrality depends on the configured sector dataset and float-market-cap
  weights. Changes to sector taxonomy can change active constraints.
- Legacy EMP008 surfaces are not the public path for this strategy. The supported
  path is the `mfbt_emp008*.py` plus `run_*.py` set described here.

## Verification

```powershell
uv run pytest tests/scripts/test_run_mfbt_emp008_full.py -q
uv run pytest tests/ingest/test_pipeline.py tests/catalog/test_groups.py tests/data/test_loader.py -q
```
