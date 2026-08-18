# EMP008 Size and Flow Comparison

## Goal

Compare the existing Size-only EMP008 portfolio with a portfolio that adds the
registered EMP008 retail-flow factor. Reuse the established Value and Momentum
comparison/reporting workflow without defining a new flow signal.

## Portfolio set

Run two cost-aligned portfolios:

1. Size-only Base: `ln_market_cap`
2. Size + Retail Flow: `ln_market_cap + retail_flow`

Size retains its LOW direction. Retail Flow retains the existing HIGH direction;
its builder already negates sector-average 252-day retail flow, so higher exposure
represents the existing contrarian retail-flow intent. The Flow factor uses
`qw_retail` and the existing construction-sector mapping.

## Shared assumptions

- Baseline weight date: 2019-12-30.
- First realized return: 2020-01-02.
- End date: 2026-06-30.
- Benchmark: KOSPI200 BM (`IKS200`).
- Fee: 0.0002.
- Sell tax: 0.0015.
- Slippage: 0.0005.
- Close fills and fractional positions.
- Risk model: `factor_idio`.
- Annual tracking-error budget: 0.007.
- Existing, first-adjustment, and second-adjustment EMP008 returns use the same
  final comparison series already used by the Value and Momentum profiles.

## Implementation approach

Add a `flow` profile to the existing comparison runner. Register one new factor
set, `size_retail_flow`, containing exactly `ln_market_cap` and `retail_flow`.
Reuse the verified Size-only cache and write Flow results to a separate
`size_flow_measure_comparison` directory. Do not duplicate the runner or alter
the Value and Momentum portfolio definitions.

## Outputs

Generate the same output package as the Value and Momentum profiles:

- cumulative returns image;
- cumulative KOSPI200-relative excess-return image;
- performance dashboard;
- Korean yearly excess-return subplots;
- full-period Korean performance table;
- annual return and annual excess-return CSV tables;
- annual performance Excel workbook;
- daily returns, detailed summary CSV/Excel, interpretation, and manifest.

The cumulative and yearly comparison artifacts include Size-only, Size + Retail
Flow, existing EMP008, first-adjustment EMP008, second-adjustment EMP008, and
KOSPI200 BM. Each yearly excess series includes the first trading-day return and
starts from a zero-basis point baseline immediately before that trading day.

## Verification

- Registry tests prove `size_retail_flow` contains exactly Size and Retail Flow.
- Profile tests prove `flow` selects only Size-only and Size + Retail Flow and
  does not change Value or Momentum defaults.
- Manifest checks prove dates, factor membership, datasets, and exact costs.
- Daily output checks prove the 2019-12-30 zero baseline, 2020-01-02 first
  realized return, 2026-06-30 endpoint, and complete aligned observations.
- Performance and yearly tables include all required historical EMP008 rows.
- Generated images are inspected for complete legends, Korean yearly labels,
  clipping, and correct annual zero resets.
