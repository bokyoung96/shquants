# EMP008 Size Base and Yearly Excess Charts

## Goal

Add a Size-only EMP008 portfolio to the existing value-measure comparison while retaining KOSPI200 as the benchmark for every excess-return calculation.

## Portfolio set

The main comparison contains four portfolios under the same risk model, tracking-error budget, execution timing, costs, and test period:

1. Size-only Base: `ln_market_cap`
2. Size + FCF/TEV
3. Size + Dividend FY0
4. Size + Dividend TTM

The annual excess-return chart additionally overlays the existing EMP008, first-adjustment EMP008, and second-adjustment EMP008 return series. KOSPI200 remains the benchmark and is not replaced by the Size-only portfolio.

## Charts

### Cumulative returns

Add the Size-only Base as a fourth portfolio line. Retain KOSPI200 BM as the benchmark line. All series start from a common NAV of 100.

### Cumulative excess returns

Add the Size-only Base as a fourth portfolio line. Calculate every line as:

`portfolio cumulative wealth / KOSPI200 cumulative wealth - 1`

Render the result in basis points and start all series at zero.

### Yearly excess-return subplots

Create one subplot for each calendar year from 2020 through 2026. Each subplot resets portfolio and benchmark wealth to one on that year's first common return date, so every line starts at zero basis points.

Plot these seven portfolios against KOSPI200:

- Size-only Base
- Size + FCF/TEV
- Size + Dividend FY0
- Size + Dividend TTM
- Existing EMP008
- First-adjustment EMP008
- Second-adjustment EMP008

Use a shared y-axis range across years for direct magnitude comparison, a zero reference line in every panel, and one figure-level legend.

Localize the yearly chart title, axis labels, and legend labels in Korean. Portfolio names may retain standard factor abbreviations such as FCF/TEV, FY0, and TTM, but the surrounding descriptions must be Korean.

## Yearly performance tables

Produce two year-by-portfolio tables for 2020 through 2026:

1. Annual return (%), including all seven portfolios and KOSPI200 BM.
2. Annual cumulative excess return versus KOSPI200 (bp), including all seven portfolios and a zero KOSPI200 BM column.

Use the same common-date yearly reset convention as the yearly subplot so the excess-return table endpoints match the plotted endpoints exactly. Write separate CSV files and one Excel workbook containing both tables.

## Data and assumptions

- Period: 2020-01-31 through 2026-06-30.
- EMP008 portfolio costs: fee 0.0002, sell tax 0.0015, slippage 0.0005.
- Execution: close fills, fractional positions allowed.
- Existing/first/second adjustment returns come from `results/emp008_adjust_comparison/20260630/daily_net_returns.csv`.
- All chart inputs are restricted to common dates with no forward filling.
- The Size-only Base is a portfolio line, not the benchmark and not the denominator of excess returns.

## Outputs

Regenerate the existing cumulative-return, cumulative-excess-return, dashboard, tables, and manifest outputs. Add a separate yearly excess-return subplot image, two yearly CSV tables, and a yearly Excel workbook. Include the Size-only Base in performance tables and interpretation output.

## Verification

- Registry tests prove the Size-only factor set contains only `ln_market_cap`.
- Comparison tests prove KOSPI200 remains the excess-return denominator.
- Plot-frame tests prove every yearly line begins at zero.
- Table tests prove the yearly excess endpoints equal the subplot endpoints and KOSPI200 excess is zero.
- The full comparison ends on 2026-06-30 and records the existing EMP008 cost bundle.
- Generated images are inspected for clipping, unreadable legends, and inconsistent axes.
