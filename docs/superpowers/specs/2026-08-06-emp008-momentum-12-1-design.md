# EMP008 12-1 Momentum and Factor Quintile Subplots

## Goal

Add a strategy-independent 12-1 month momentum factor, expose it through a new
Origin variant, and produce subplot images for five-quantile performance across
the full independent factor catalog.

## Factor and factor sets

`momentum_12_1m` is calculated at signal month `t` as:

```text
month_end_close(t-1) / month_end_close(t-12) - 1
```

`origin_12_1m` selects `ln_market_cap`, `momentum_12_1m`, and
`dividend_yield_fy0`. It retains Origin's directional-alpha constraint and
seven-day FY0 snapshot allowance. Existing factor sets remain unchanged.

`all_factors` is diagnostics-only. It selects every independent `FactorId` in
registry order, applies no strategy-specific large-benchmark neutralization,
and loads the union of required datasets. It uses the seven-day snapshot
allowance because it includes `dividend_yield_fy0`.

## Quantile plots

The existing quantile evaluator remains authoritative for Q1-Q5 membership,
returns, cumulative returns, spreads, rank IC, and summary metrics. Output
writing adds two images generated from `cumulative_returns`:

- `cumulative_quintiles_equal_weight.png`
- `cumulative_quintiles_market_cap_weight.png`

Each image contains one subplot per factor. Every subplot shows Q1-Q5 and the
direction-aware spread using consistent colors and a shared legend.

## Verification

- Unit-test the exact 12-1 formula and missing warmup values.
- Verify new factor-set membership and unchanged existing memberships.
- Verify subplot files are non-empty and contain one axis per factor.
- Run `all_factors` with five buckets from 2020-01-31 through 2026-06-30 and
  inspect both generated images.

