# assetallocation Feature Map

## Research Basis

The first assetallocation feature set uses simple, auditable transformations before any shallow model or TFT model consumes the data.

| Feature family | Rationale | Source signal |
| --- | --- | --- |
| Time-series momentum | Prior returns can contain continuation information across asset classes; volatility scaling is important when evaluating momentum signals. | Price-like tickers |
| Realized volatility | Volatility is a core risk and allocation input; rolling return volatility is a standard risk proxy. | Price-like tickers |
| OHLC range and open-close change | Range-based volatility research uses high, low, open, and close information instead of close-only returns. | All tickers |
| Yield-curve level, changes, and inversion flags | Term structure spreads are common macro state and recession-risk indicators. | `USYC2Y10 Index`, `USYC3M2Y Index`, `USGG10YR Index` |
| Bond-rate context from yield change | `USGG10YR Index` remains useful macro context even though the investable bond leg is now IEF. | `USGG10YR Index` |
| Cross-asset relative momentum | The allocation decision is SPY versus IEF, so relative ETF signals are more direct than standalone forecasts. | `SPY US Equity`, `IEF US Equity`, rates, commodities |

Key references used to choose the map:

- Moskowitz, Ooi, and Pedersen document time-series momentum across futures markets: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463
- Goyal and Welch review equity-premium predictors including stock variance, long-term yield, and term spread: https://www.nber.org/papers/w10483
- Estrella and Mishkin document the predictive use of yield-curve term structure for recession risk: https://business.columbia.edu/faculty/research/predictive-power-term-structure-interest-rates-implications-european-central-bank
- Cochrane and Piazzesi show yield-curve information is relevant to bond risk premia: https://web.stanford.edu/~piazzesi/cp.pdf
- Yang and Zhang provide an OHLC-based volatility estimator, motivating use of open-high-low-close range information: https://ideas.repec.org/a/ucp/jnlbus/v73y2000i3p477-91.html

## Output Files

`assetallocation/feature/builder.py` writes:

- `features.parquet`: model input features indexed by date.
- `targets.parquet`: forward return targets indexed by date.

Current generated dataset:

- `features.parquet`: 6,883 rows x 122 feature columns.
- `targets.parquet`: 6,883 rows x 12 target columns.

## Feature Families

### Price-Like Momentum And Risk

Applied to:

- `USDJPY Curncy`
- `GC1 Comdty`
- `HG1 Comdty`
- `SPX Index`
- `INDU Index`
- `RTY Index`
- `SPY US Equity`
- `IEF US Equity`

Generated columns:

- `{asset}_ret_1d`
- `{asset}_mom_5d`
- `{asset}_mom_20d`
- `{asset}_mom_60d`
- `{asset}_vol_20d`
- `{asset}_vol_60d`
- `{asset}_drawdown_60d`

`CL1 Comdty` is treated as a level/change series rather than a percentage-return series because the WTI front future has a negative historical print in the source data.

### Level, Change, And Z-Score

Applied to:

- `USYC2Y10 Index`
- `USYC3M2Y Index`
- `USGG10YR Index`
- `CL1 Comdty`

Generated columns:

- `{asset}_level`
- `{asset}_chg_5d_*`
- `{asset}_chg_20d_*`
- `{asset}_chg_60d_*`
- `{asset}_chg_vol_20d`
- `{asset}_chg_vol_60d`
- `{asset}_z_252d`

For `USGG10YR Index`, changes are stored in basis points using `{asset}_chg_{horizon}d_bp`.

### OHLC Features

Applied to all 12 tickers:

- `{asset}_hl_range`
- `{asset}_oc_change`

### Cross-Asset Features

Generated columns:

- `us10y_proxy_ret_1d`
- `spx_excess_us10y_proxy_ret_1d`
- `spy_excess_ief_ret_1d`
- `spx_vs_us10y_proxy_mom_5d`
- `spx_vs_us10y_proxy_mom_20d`
- `spx_vs_us10y_proxy_mom_60d`
- `spy_vs_ief_mom_5d`
- `spy_vs_ief_mom_20d`
- `spy_vs_ief_mom_60d`
- `curve_2y10_inverted`
- `curve_3m2y_inverted`
- `gold_vs_spx_mom_20d`
- `copper_vs_gold_mom_20d`
- `equity_breadth_mom_20d`

## Targets

The current target set supports the two-asset allocation objective between `SPY US Equity` and `IEF US Equity`:

- `target_spy_fwd_5d`
- `target_ief_fwd_5d`
- `target_spy_excess_ief_fwd_5d`
- `target_spy_over_ief_direction_5d`
- Same columns for 20-day and 60-day horizons.

`USGG10YR Index` is still converted to a simple duration-based return proxy for feature context:

```text
us10y_proxy_ret_1d = -7.0 * daily_change(USGG10YR yield in decimal)
```

This is no longer the investable bond-leg target. It is a macro/rate feature only.

## Feature/Target Alignment

Each row is indexed by the decision date `t`.

`features.loc[t]` contains only information observable at or before `t`.

Examples:

- `spx_mom_20d[t]`: SPX return over the 20 trading days ending at `t`.
- `spx_vol_20d[t]`: SPX realized volatility over the 20 trading days ending at `t`.
- `us10y_chg_20d_bp[t]`: 10-year Treasury yield change over the 20 trading days ending at `t`.

`targets.loc[t]` contains the future outcome that a model is trained to predict from `features.loc[t]`.
Because `features.loc[t]` can use the close at `t`, target returns assume the position is entered on the next trading day.

Examples:

- `target_spy_fwd_20d[t]`: SPY return from `t + 1` to `t + 21` trading days.
- `target_ief_fwd_20d[t]`: IEF return from `t + 1` to `t + 21` trading days.
- `target_spy_excess_ief_fwd_20d[t]`: `target_spy_fwd_20d[t] - target_ief_fwd_20d[t]`.

Conceptually:

```text
features[t]  ->  model  ->  predicted target[t]

target_spy_fwd_20d[t] = SPY_close[t + 21 trading days] / SPY_close[t + 1 trading day] - 1
```

This means a row such as `2020-01-02` uses market information available through `2020-01-02` as features, while its 20-day target stores the realized outcome from the next trading day through 20 trading days of holding.

NaN placement follows the direction of each calculation:

- Rolling features can be NaN near the beginning of the dataset because there is not enough prior history.
- Forward targets can be NaN near the end of the dataset because there is not enough future data.

During model training, the usual pairing is:

```text
X = features.loc[train_dates]
y = targets.loc[train_dates, "target_spy_excess_ief_fwd_20d"]
```

During live prediction, only `features.loc[t]` is available. The true `targets.loc[t]` becomes known only after the forecast horizon has passed.
