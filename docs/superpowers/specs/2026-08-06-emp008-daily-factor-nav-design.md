# EMP008 Daily Factor NAV Design

## Goal

Keep the existing month-end factor sort and monthly rebalance schedule, but show the realized path between rebalances with daily adjusted-close valuations. The existing monthly return, IC, turnover, and summary artifacts remain unchanged.

## Portfolio semantics

- A factor is observed and Q1-Q5 membership is fixed on each existing `signal_date`.
- Equal-weight and market-cap-weight allocations are established once at that signal date.
- Each allocation is held as fixed shares until the existing `return_date`; it is not re-sorted or rebalanced daily.
- Daily portfolio value is `sum(initial_weight * close[date] / close[signal_date])`.
- Missing intermediate prices use the most recent price observed after the valid signal price. Endpoint eligibility remains governed by the existing monthly diagnostic rules.
- The next monthly allocation starts after the prior `return_date`, so boundary dates are not duplicated.

## Daily cumulative paths

Add `daily_cumulative_returns` to `Emp008FactorQuantileResult` with columns:

`signal_date`, `date`, `factor`, `weighting`, `portfolio`, `cumulative_return`.

Q1-Q5 paths chain each holding-period NAV onto the prior month-end wealth. `high_minus_low` and `preferred_minus_avoided` are derived from the two extreme quantile holding-period NAVs and chained at monthly boundaries. Their month-end values must exactly match the existing monthly cumulative-return artifact.

The first eligible signal date is emitted as a zero-return baseline. All subsequent observations use actual dates present in the adjusted-close data.

## Outputs

- Preserve `cumulative_returns.csv` as the existing monthly audit artifact.
- Add `daily_cumulative_returns.csv` and `daily_cumulative_returns.parquet`.
- Generate both existing subplot PNG names from daily cumulative paths.
- Extend the manifest and CLI payload with daily artifact paths, row counts, `rebalance_frequency: monthly`, and `nav_frequency: daily`.

## Validation

- A deterministic multi-date price fixture proves intermediate daily NAV values.
- Daily month-end endpoints equal the existing monthly cumulative paths for every factor, weighting, and portfolio.
- Output validation rejects missing, duplicated, non-finite, or inconsistent daily rows.
- Artifact publication remains atomic, including both daily files and the PNGs.
- The full all-factor run covers 2020-01-31 through 2026-06-30 and both images are inspected visually.

## Non-goals

- No daily factor re-ranking or daily rebalance.
- No changes to existing factor definitions, factor-set membership, monthly performance summaries, IC, turnover, or optimizer behavior.
