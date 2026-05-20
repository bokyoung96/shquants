# assetallocation Asset Map

## Source Data

The parquet files in this directory are split from the Bloomberg-style OHLC workbook by field:

- `open.parquet`: `PX_OPEN`
- `high.parquet`: `PX_HIGH`
- `low.parquet`: `PX_LOW`
- `close.parquet`: `PX_LAST`

Rows are indexed by date. Columns are Bloomberg tickers.

## Ticker Map

| Ticker | Meaning | Asset Class / Interpretation |
| --- | --- | --- |
| `USYC2Y10 Index` | US 2-year minus 10-year yield spread | Yield curve steepener/flattener indicator |
| `USYC3M2Y Index` | US 3-month minus 2-year yield spread | Common recession-leading yield curve indicator |
| `USDJPY Curncy` | USD/JPY exchange rate | FX, yen carry, risk-on/risk-off proxy |
| `USGG10YR Index` | US 10-year Treasury yield | Global long-rate benchmark |
| `GC1 Comdty` | Gold front futures | Gold, safe-haven asset, real-rate sensitive |
| `CL1 Comdty` | WTI crude oil front futures | Oil, inflation, cyclicality |
| `HG1 Comdty` | Copper front futures | Copper, Dr. Copper, cyclical growth proxy |
| `SPX Index` | S&P 500 | US large-cap equities |
| `INDU Index` | Dow Jones Industrial Average | US traditional industrial / cyclical equities |
| `RTY Index` | Russell 2000 | US small-cap equities |
| `SPY US Equity` | SPDR S&P 500 ETF Trust | Tradable US large-cap equity ETF |
| `IEF US Equity` | iShares 7-10 Year Treasury Bond ETF | Tradable intermediate Treasury ETF |

## Allocation Objective

The investable allocation target is restricted to two assets:

- `SPY US Equity`
- `IEF US Equity`

`SPX Index` and `USGG10YR Index` remain context features. They may be used as predictors for regime, macro state, inflation pressure, rate risk, growth sensitivity, or risk appetite, but they should not receive portfolio weights in the current assetallocation workflow.

The current target and backtest compare tradable ETF returns:

```text
target = forward_return(SPY US Equity) - forward_return(IEF US Equity)
```

ETF return quality depends on the Bloomberg field used in the source workbook. If the workbook uses unadjusted `PX_LAST`, distributions are not fully reflected; adjusted or total-return ETF series would be preferable for production-grade performance measurement.
