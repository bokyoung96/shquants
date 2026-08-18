# MFBT EMP008 Origin Directional Warmup Design

## Goal

Preserve the Origin EMP008 portfolio-construction scheme while extending it to
the six MFBT factors, and produce an investable portfolio at the end of 2019 by
using the preceding 36 months of factor-return history.

## Expected-alpha policy

Raw factor values, preprocessed z-scores, realized factor returns, and risk
estimates remain unchanged. The policy is applied only to the rolling 36-month
mean factor return passed to the optimizer.

The five MFBT factors whose economically intended direction is positive use a
zero floor:

- `price_momentum`
- `earnings_momentum`
- `dividend_yield`
- `retail_flow`
- `value`

For each of these factors, `adjusted_alpha = max(mean_36m, 0)`. The size factor
retains Origin's small-cap direction through
`adjusted_alpha["ln_market_cap"] = min(mean_36m, 0)`. Sector expected returns
remain zero.

This makes the rolling estimate an on/off strength filter. A negative estimate
does not reverse an economically positive signal into a contrarian strategy.
The factor exposure remains in the factor covariance and therefore continues
to affect portfolio risk.

## Historical benchmark weights

The factor inputs extend to 1999, but `QW_BM_WEIGHTS` begins on 2020-01-02. The
current runner therefore cannot estimate pre-2020 factor returns and waits for
36 new observations before producing a portfolio.

For dates before the first official benchmark-weight row, EMP008 will construct
a warmup benchmark proxy by normalizing `QW_MKTCAP_FLT` across the contemporaneous
`QW_K200_YN` universe. Official `QW_BM_WEIGHTS` rows remain unchanged from their
first available date onward. On 2020-01-02, the proxy and official weights have
0.99975 cross-sectional correlation; the proxy is used only where the official
series does not exist.

The proxy is used both to demean stock returns during the 36-month warmup and
as the benchmark anchor for the initial 2019 year-end portfolio. This avoids
backfilling future official weights across the entire warmup period.

## Date convention

2019-12-31 and 2020-01-01 were not KRX trading days. The corresponding first
signal is dated 2019-12-30, the last trading day of 2019. With the existing
close-to-close EMP008 convention, the portfolio is in place for the first 2020
trading return on 2020-01-02.

The full run will request output from December 2019 and will load enough padded
history to accumulate at least 36 monthly factor-return observations before the
2019-12-30 optimization.

## Unchanged components

- Six raw MFBT factor definitions
- Float-market-cap fill, centering, ranking, and z-scoring
- Cross-sectional factor-return regression
- Factor-plus-idiosyncratic covariance construction
- WICS sector neutrality
- Long-only benchmark-relative SLSQP optimization
- Annual 0.70% ex-ante tracking-error budget
- Gross and costed backtests, comparison, and factor attribution

## Acceptance criteria

1. Negative rolling means for the five positive-direction factors become zero.
2. Positive `ln_market_cap` rolling means become zero; negative means remain.
3. Historical factor returns and factor exposures are not modified by the sign
   policy.
4. Pre-official benchmark rows are positive, normalized float-cap proxies.
5. Official benchmark rows are byte-for-byte/numerically unchanged.
6. A full WICS Factor+Idio run produces its first target weight on 2019-12-30.
7. The first 2020 trading return is included in the backtest.
8. Every rebalance optimization succeeds; final weights sum to one, remain
   nonnegative, and satisfy sector neutrality and the TE constraint.
9. Gross, costed, comparison, and factor-attribution artifacts are generated.
