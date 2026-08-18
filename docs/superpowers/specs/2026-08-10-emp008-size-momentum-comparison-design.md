# EMP008 Size and Momentum Measure Comparison

## Goal

Compare three momentum measures while keeping the existing Size-only EMP008 portfolio as the base portfolio and KOSPI200 as the benchmark.

## Portfolio set

Run four cost-aligned EMP008 portfolios:

1. Size-only Base: `ln_market_cap`
2. Size + 12-month momentum: `ln_market_cap + momentum_12m`
3. Size + 12-1-month momentum: `ln_market_cap + momentum_12_1m`
4. Size + high-relative momentum: `ln_market_cap + price_to_252d_high`

Size keeps its LOW direction. All three momentum factors keep their existing HIGH direction. Each factor set constrains expected alpha to factor direction.

## Shared assumptions

- Period: 2020-01-31 through 2026-06-30.
- Benchmark: KOSPI200 BM (`IKS200`).
- Fee: 0.0002.
- Sell tax: 0.0015.
- Slippage: 0.0005.
- Close fills and fractional positions.
- Risk model: `factor_idio`.
- Annual tracking-error budget: 0.007.

## Implementation approach

Extend the existing size-measure comparison runner with a momentum comparison profile instead of copying the complete runner. The value profile remains the default and its existing output directory and artifacts remain unchanged. Momentum outputs use a separate `size_momentum_measure_comparison` directory.

## Outputs

Generate the same artifacts as the Size–Value comparison:

- cumulative returns image with KOSPI200 and four portfolios;
- cumulative KOSPI200-relative excess-return image;
- performance dashboard;
- Korean yearly excess-return subplots, reset to zero basis points each year;
- annual return and annual excess-return CSV tables;
- annual performance Excel workbook;
- daily returns, summary CSV/Excel, interpretation, and manifest.

The yearly subplot additionally includes existing EMP008, first-adjustment EMP008, and second-adjustment EMP008 from `results/emp008_adjust_comparison/20260630/daily_net_returns.csv`.

## Verification

- Registry tests prove each new factor set contains Size plus exactly one momentum measure.
- Profile tests prove Value defaults remain unchanged and Momentum selects the four intended portfolios.
- All excess returns retain KOSPI200 as denominator.
- Manifest proves period, costs, risk conditions, and factor membership.
- Generated images are visually inspected for Korean labels, clipping, and yearly zero resets.
- Performance tables are checked against plotted endpoints.
