# Flow-Filtered Breakout Strategy Design

Date: 2026-06-29

## Intent

Test a simpler version of the 5-minute 52-week breakout strategy by removing OP revision filters and using investor flow only as a positive confirmation gate.

The first version should answer one question: does requiring foreign or institutional sponsorship improve the quality of positivity-supported breakout entries?

## Strategy Shape

The strategy remains a long-only KOSPI200 historical-member breakout sleeve.

Entry candidates are created from 5-minute stock bars, but the context filters are daily and use only information available before the signal date.

Base entry conditions:

1. The ticker is a KOSPI200 member on the signal date.
2. Current 5-minute close is above the prior 52-week daily close high.
3. Previous 5-minute close was not above that same prior 52-week high.
4. Breakout distance is at least the configured buffer in basis points.
5. Signal time is after the opening range cutoff and before the configured intraday exit cutoff.
6. Positivity spread passes the configured margin.
7. Flow filter passes.

Execution:

- Signal time is the 5-minute bar that confirms the first breakout by close.
- Entry price is the next 5-minute open.
- The first implementation stays long-only.

## Positivity Spread

Daily positivity is computed from each ticker's daily close returns and shifted by one day before use.

The benchmark is a same-date cross-sectional reference:

- sector cap-weighted positivity
- index cap-weighted positivity
- sector equal-weight positivity
- index equal-weight positivity

The filter is:

```text
daily_positivity - positivity_benchmark >= positivity_margin
```

Initial grid:

```text
positivity_lookback_days = 60, 90, 126
positivity_benchmark = sector_cap_weighted, index_cap_weighted
positivity_margin = 0.00, 0.02, 0.05
```

## Flow Filter

OP revision is intentionally removed from this experiment. The only research overlay is investor flow.

Flow is measured as trailing net flow divided by market cap, shifted so the signal date cannot see same-day flow.

Initial flow filters:

```text
none
foreign_positive
institution_positive
foreign_or_institution_positive
foreign_and_institution_positive
```

Filter definitions:

```text
foreign_positive = foreign_flow_to_cap > 0
institution_positive = institution_flow_to_cap > 0
foreign_or_institution_positive = foreign_flow_to_cap > 0 OR institution_flow_to_cap > 0
foreign_and_institution_positive = foreign_flow_to_cap > 0 AND institution_flow_to_cap > 0
```

Initial grid:

```text
flow_lookback_days = 20, 40, 60
```

Recommended first comparison:

```text
A. no_flow
B. foreign_or_institution_positive
C. foreign_and_institution_positive
```

This keeps the first run interpretable before adding score blending or rank weighting.

## Exit Logic

The first version keeps the continuation exit model:

1. Do not exit before `min_holding_days`.
2. Exit at ATR stop if daily low touches:

```text
entry_price - ATR * atr_stop_multiplier
```

3. Exit on breakout failure if daily close falls back to or below the prior 52-week high.
4. If neither condition occurs, close at the end of available data for research accounting.

Trading cost remains round-trip 3 bps in net return.

## Outputs

Each run should write:

- grid summary
- top strategies
- best strategy JSON
- selected strategy config
- trade list
- daily returns
- entry-condition visualization for representative examples

The first review should compare no-flow and flow-filtered variants on:

- trade count
- average net bps
- hit rate
- max drawdown
- robust early/late performance
- active month coverage

## Non-Goals

This version does not:

- use OP revision or OP sector rank
- blend flow into score
- create a short leg
- add portfolio capital constraints beyond existing research accounting
- replace the current continuation exit model

## Implementation Notes

The current data folder is expected at:

```text
parquet/KR_STOCK_5m
```

The implementation should adapt the existing `tech_gamma_*` pipeline rather than creating a second strategy framework. Rename `factor_filter` only if the implementation can do so without broad churn; otherwise keep the existing field name and restrict allowed values to the flow-only filters for this experiment.

