# EMP008 Size + Earnings Momentum Comparison Design

## Goal

Extend the existing Size + Momentum comparison with one independent portfolio that combines the unchanged Size signal with the registered operating-profit consensus momentum factor. Regenerate the same charts and Korean performance tables through 2026-06-30.

## Comparison Set

Keep the existing portfolios and append one new variant in this order:

1. `size_only`
2. `size_momentum_12m`
3. `size_momentum_12_1m`
4. `size_momentum_high`
5. `size_earnings_momentum`

The new factor set contains exactly:

- `ln_market_cap`
- `earnings_momentum`

It uses the existing directional alpha constraint and does not add rank transforms, benchmark-weight neutralization, or snapshot-forward handling.

## Signal Definition

Reuse the registered `earnings_momentum` factor without changing its calculation:

- Dataset: `qw_op_fwd_12m`
- Frequency: month-end observations
- Formula: `(current 12MF OP consensus - previous month 12MF OP consensus) / abs(previous month 12MF OP consensus)`
- Missing current or previous consensus remains unavailable.
- When current 12MF operating profit is below KRW 100 billion and growth exceeds 50%, reset growth to `0.0` using the existing EMP008 configuration.

## Execution Contract

- Period: 2020-01-31 through 2026-06-30.
- Benchmark: KOSPI200 (`IKS200`).
- Fill mode: close.
- Costs: fee `0.0002`, sell tax `0.0015`, slippage `0.0005`.
- Reuse valid cached results for Size-only and the three existing price-momentum variants.
- Compute only the new Size + earnings-momentum portfolio unless cache validation requires otherwise.
- Do not silently shorten the requested end date if consensus data is unavailable; surface the mismatch.

## Outputs

Regenerate the existing momentum comparison package in place:

- `cumulative_returns.png`
- `cumulative_excess_returns.png`
- `performance_dashboard.png`
- `yearly_excess_returns.png`
- `performance_summary.csv` and `.xlsx`
- `performance_table_ko.csv`
- `daily_returns.csv`
- `yearly_returns_pct.csv`
- `yearly_excess_returns_bp.csv`
- `yearly_performance.xlsx`
- `interpretation.md`
- `manifest.json`

The charts and tables include the five current comparison portfolios. The annual chart and annual tables continue to include existing, first-adjustment, and second-adjustment EMP008 alongside KOSPI200. Use a distinct magenta color for the new portfolio.

## Verification

- Registry tests lock the new factor-set membership and dataset dependency.
- Comparison tests lock profile ordering, labels, output generation, and parser behavior.
- Cache metadata must match period, risk model, factor set, fill mode, and all three cost fields.
- The regenerated manifest must list `ln_market_cap` and `earnings_momentum` with `qw_op_fwd_12m`.
- All four images must be readable and visually inspected.
- Daily returns must cover the requested dates with no missing values.
- Korean performance and annual tables must include the new portfolio and KOSPI200.

## Scope Boundaries

This work does not change the earnings-momentum formula, optimizer settings, benchmark, historical EMP008 series, or the three existing price-momentum definitions. It makes no statistical-significance claim.
