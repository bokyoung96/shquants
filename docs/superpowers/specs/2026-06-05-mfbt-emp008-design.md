# MFBT EMP008 Barra Portfolio Design

## Goal

Build an `emp008`-internal MFBT Barra pipeline that converts the MFBT factor ideas into continuous Barra-style factor exposures and produces rolling KOSPI200 benchmark-relative target weights.

The existing `backtesting/strategies/mfbt.py` remains unchanged. It continues to represent the current score/audit-oriented MFBT strategy. The new work lives inside `backtesting/strategies/emp008/` and treats MFBT factors as raw exposure inputs for regression, risk modeling, and constrained optimization.

## Non-Goals

- Do not reuse the existing MFBT `0/1` or `0..4` score outputs as Barra exposures.
- Do not keep legacy `emp008/DATA.xlsx` factor-sheet dependencies for the new MFBT EMP008 path.
- Do not preserve unused legacy `emp008` code or data paths once they are no longer needed for the new path.
- Do not modify `mfbt.py` as part of this integration.

## Data Sources

The new path uses shquants catalog/parquet datasets rather than legacy `emp008/DATA` factor sheets.

Required datasets:

- `QW_C`: raw close used to build KOSPI200 benchmark weights.
- `QW_BM_WEIGHTS`: derived benchmark weights, `date x ticker`.
- `QW_ADJ_C`: adjusted close for factor and return calculations where total-return-like continuity is needed.
- `QW_OP_FWD_12M`: 12-month forward operating profit estimate.
- `QW_DPS_TTM`: trailing DPS.
- `QW_RETAIL`: retail flow.
- `QW_WI_SEC_26_BIG`: default sector taxonomy.
- `QW_MKTCAP`: market cap.
- `QW_MKTCAP_FLT`: float market cap, used for preprocessing weights.
- `QW_FCF`: free cash flow.
- `QW_INT_BEARING_LIAB_NFQ0`: interest-bearing liability.
- `QW_QUICK_ASSETS_NFQ0`: quick assets.
- `QW_K200_YN`: KOSPI200 universe.

`QW_BM_WEIGHTS` is a formal derived ingest dataset. It is built from `raw/krx_ks200_weight.xlsx` `Sheet2` and `QW_C`:

```text
float_index_value = Index_Share * Free_Float_Factor * qw_c
bm_weight = float_index_value / row_sum(float_index_value)
```

The `Capping_Factor` column is ignored. `Sheet1` is ignored.

## Sector Dataset

The default sector dataset is `QW_WI_SEC_26_BIG`.

Reasons:

- It matches the sector source already used by `mfbt.py` for retail flow.
- It is daily, which aligns better with 252-day retail-flow rolling windows.
- It can be aligned to the close grid and forward-filled before month-end extraction.

The pipeline keeps `sector_dataset: DatasetId = DatasetId.QW_WI_SEC_26_BIG` configurable for future experiments, but the first implementation defaults to `QW_WI_SEC_26_BIG`.

Sector data has two separate roles:

1. Retail-flow alpha exposure: sector-level flow is computed and mapped to member stocks.
2. Barra sector exposure: sector dummy/active exposure columns are added for risk and sector-neutral constraints.

These two uses must remain conceptually separate.

## Proposed File Boundaries

Use small files inside `backtesting/strategies/emp008/`:

```text
mfbt_emp008.py              # public runner / orchestration
mfbt_emp008_data.py         # shquants catalog loading
mfbt_emp008_factors.py      # raw MFBT factor calculations
mfbt_emp008_preprocess.py   # fill, float-mktcap center, z-score, sector exposure
mfbt_emp008_risk.py         # regression, factor covariance, residual covariance, expected alpha
mfbt_emp008_optimize.py     # monthly optimizer and rolling target weights
```

Avoid one large all-purpose file.

## Factor Definitions

All factors are continuous raw exposures. No score buckets are used.

| Factor | Raw exposure |
| --- | --- |
| `price_momentum` | `close / close.rolling(252).max()` |
| `earnings_momentum` | `(monthly_op_fwd_12m - previous_month_op_fwd_12m) / abs(previous_month_op_fwd_12m)` |
| `dividend_yield` | `monthly_dps_ttm / monthly_close` |
| `retail_flow` | `-sector_avg_252d_retail_flow`, mapped from sector to stock |
| `value` | `lagged_fcf / (monthly_market_cap + lagged_debt - lagged_quick_asset)` |
| `ln_market_cap` | `log(monthly_market_cap)` |

Special rules:

- `earnings_momentum` keeps the existing low-OP extreme-growth reset: if current OP is below `100_000_000_000` and growth is above `0.50`, set growth to `0.0`.
- `dividend_yield` removes the existing three-year dividend-growth bonus.
- `retail_flow` uses negative sector average flow so larger retail net selling maps to higher exposure, matching current MFBT intent.
- `value` treats `TEV <= 0` as missing (`NaN`) for Barra exposure. Do not use `-inf`; that was only suitable for score ranking.
- `ln_market_cap` follows the legacy EMP008 treatment: fill missing log market cap using the float-market-cap weighted mean, rank cross-sectionally, then center and z-score.
- After preprocessing, `ln_market_cap` is neutralized to exposure `0.0` for stocks whose `QW_BM_WEIGHTS` weight is at least `10%` on that date, so very large benchmark names are not pushed by the size score.

All factor outputs use `date x ticker` monthly panels.

## Preprocessing

For each date and factor:

```text
raw exposure
-> apply KOSPI200 universe mask
-> fill missing values with QW_MKTCAP_FLT weighted mean
-> optionally rank-transform selected factors such as `ln_market_cap`
-> subtract QW_MKTCAP_FLT weighted mean
-> z-score standardize
-> optionally set selected factor exposures to `0.0` for benchmark weights above configured large-name thresholds
```

`QW_MKTCAP_FLT` is the preprocessing weight source. It is separate from `QW_BM_WEIGHTS`, which is the final benchmark weight source for optimization.

The centered exposure should have float-mktcap-weighted mean near zero on each date.

## Sector Exposures

Sector exposure columns are built from the configured sector dataset:

```text
sector_active_exposure = sector_dummy - sector_float_mktcap_weight
```

The optimizer must receive explicit `alpha_factor_names` and `sector_factor_names`. Do not use positional assumptions such as `sector_start_idx = 4`.

`alpha_factor_names`:

```text
price_momentum
earnings_momentum
dividend_yield
retail_flow
value
ln_market_cap
```

`sector_factor_names` are derived from sector dummy columns.

## Rolling Regression And Risk Model

For each target date:

1. Build preprocessed factor exposures at the factor date.
2. Build next-period stock returns and benchmark-relative excess returns.
3. Run cross-sectional regression of excess returns on factor exposures.
4. Store factor returns and residuals.
5. Compute expected alpha from the recent rolling window of factor returns.
6. Compute factor covariance from recent factor returns.
7. Compute residual covariance as diagonal stock residual variance.
8. Build full covariance:

```text
M = D + Z @ factor_cov @ Z.T
```

Expected alpha for alpha factors is the recent rolling mean factor return. Expected alpha for sector factors is fixed at `0.0`; sector columns are risk/constraint exposures, not alpha sources.

## Optimization

Optimize active weights relative to `QW_BM_WEIGHTS`.

Objective:

```text
maximize alpha_exp.T @ Z.T @ active_weight
```

Constraints:

```text
sum(active_weight) = 0
sector_exposure.T @ active_weight = 0
active_weight.T @ M @ active_weight <= TE^2
final_weight = bm_weight + active_weight >= 0
```

Bounds:

```text
active_weight_i >= -bm_weight_i
```

Stocks missing benchmark weights for the target date are excluded from that optimization universe.

## Outputs

Primary output:

```text
target_weights.parquet
index: rebalance date
columns: ticker
values: final portfolio weight
```

This is the standard backtest bridge format.

Review/export output:

```text
weights_export.xlsx
sheet1: ticker x date final weights
sheet2: summary by date
sheet3: active weights
sheet4: diagnostics
```

Run artifacts are grouped under `results/emp008_runs/<name>/` by default:

```text
weights/target_weights.parquet
weights/target_weights.csv
weights/active_weights.parquet
weights/active_share.csv
weights/active_share.parquet
weights/diagnostics.parquet
backtests/<run_id>/
backtests/<run_id>/series/active_share.csv
reports/<name>/report.html
run_summary.json
```

`active_share` is computed monthly as `0.5 * sum(abs(active_weight))`.

Diagnostics should include:

- `success`
- `objective_value`
- `tracking_error`
- `n_active_positions`
- `max_weight`
- `min_weight`
- `sum_final_weight`
- `sum_active_weight`
- `sector_active_exposure_abs_max`
- `factor_names`
- `alpha_factor_names`
- `sector_factor_names`

Optional debug inputs may be saved when `save_inputs=True`:

```text
inputs/<date>/exposures.parquet
inputs/<date>/factor_cov.parquet
inputs/<date>/residual_var.parquet
inputs/<date>/alpha_exp.parquet
inputs/<date>/bm_weight.parquet
```

## Cleanup Policy

The new MFBT EMP008 path should keep only necessary code and data surfaces.

Before deleting any legacy `emp008` code or data:

1. Identify whether the legacy path is used by the new pipeline.
2. Add or preserve tests covering the behavior the new pipeline needs.
3. Remove only code/data that is clearly not needed for the new MFBT EMP008 path.
4. Do not modify or delete unrelated user changes.
5. Do not delete external/raw source data unless it has been replaced by a catalog ingest source and the deletion is explicitly scoped.

Prefer deletion over adapter layers once a legacy component has no remaining purpose in the new path.

## Tests

Add focused tests under `tests/strategies/emp008/`.

Factor tests:

- `price_momentum` emits `close / 252d_high`.
- `earnings_momentum` emits OP growth and applies low-OP extreme reset.
- `dividend_yield` emits `DPS_TTM / close` and does not include dividend-growth bonus.
- `retail_flow` emits negative sector average 252-day retail flow mapped to stocks.
- `value` emits `FCF / TEV` and uses `NaN` for `TEV <= 0`.
- `ln_market_cap` emits `log(monthly_market_cap)`.

Preprocess tests:

- Universe mask excludes non-members.
- Missing raw exposure is filled with float-mktcap-weighted mean.
- Centered exposure has float-mktcap-weighted mean near zero.
- Z-scored exposure is finite for valid rows.

Sector tests:

- Sector dummy columns are generated from `QW_WI_SEC_26_BIG`.
- Sector active exposures are dummy minus float-mktcap sector weight.
- Alpha factor names and sector factor names are explicit and disjoint.

Risk tests:

- Cross-sectional regression returns factor returns and residuals.
- Sector expected alpha is zero.
- Factor covariance shape matches factor names.
- Residual covariance is diagonal.

Optimizer tests:

- Final weights sum to one.
- Active weights sum to zero.
- Final weights are non-negative.
- Sector active exposure residual is within tolerance.
- Output is `date x ticker`.

Integration smoke test:

- Run a short two- or three-month rolling pipeline on real parquet data.
- Verify outputs and diagnostics exist and have expected shapes.

Minimum verification commands after implementation:

```powershell
uv run pytest tests/scripts/test_run_mfbt_emp008_full.py -q
uv run pytest tests/ingest/test_pipeline.py tests/catalog/test_groups.py tests/data/test_loader.py -q
```
