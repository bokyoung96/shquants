# Positivity Event-Driven Long Alpha Design

## Goal

Research and then implement an event-driven aggressive long-alpha overlay built
around positivity.

The strategy should not behave like a calendar rebalancing top-N portfolio. It
should behave like an index-management overlay:

- enter only when a stock emits a qualifying signal,
- keep the active overweight while the signal remains valid,
- exit when a volatility stop or sell signal fires,
- manage a small event queue of active names before deciding final active-weight
  bands.

The first objective is to prove whether the alpha exists. Position-size limits,
active-weight bands, and final portfolio construction are later implementation
choices.

## Non-Goals

- Do not optimize active-weight bands in the first research pass.
- Do not assume weekly or monthly forced rebalancing.
- Do not build a standalone equal-weight stock-picking portfolio as the final
  product.
- Do not use low positivity alone as a short signal.
- Do not introduce a complex hybrid rule before the simple breakout-confirmation
  alpha is measured.

## Strategy Thesis

Positivity appears to describe the stability of a stock's return path. For an
aggressive long-alpha overlay, positivity should be paired with a catalyst that
can turn stable strength into near-term upside.

The first catalyst to test is price position:

```text
high positivity
+ near 252-day high or new 252-day breakout
+ optional revision or flow confirmation
= candidate for temporary active overweight
```

The expected source of alpha is not that positivity alone predicts all returns.
The expected source is that high-positivity stocks near highs have more reliable
continuation after strength signals than ordinary breakout names.

## Event-Driven Shape

The strategy has four states per stock:

| State | Meaning |
| --- | --- |
| `inactive` | No current active overweight. |
| `candidate` | Entry signal exists, but the event queue has not allocated a slot. |
| `active` | Stock is held as an active overweight candidate. |
| `exit_pending` | A stop, sell signal, or stronger replacement has fired. |

There is no calendar rebalance that refreshes all names. Calendar dates are only
used as observation points in historical data. Trades are caused by state
transitions.

## Entry Rule

Initial research should allow near-high entries, not only exact new-high
breakouts, to preserve sample size.

Required entry conditions:

```text
positivity_rank is in the top eligible group
close is within the configured threshold of the 252-day high
liquidity and universe filters pass
```

Default first-pass near-high definition:

```text
close / rolling_252d_high >= 0.95
```

Breakout variants should also be tested:

```text
close >= rolling_252d_high
```

Optional confirmation overlays:

- positive OP or EPS revision,
- foreign or institutional sponsorship,
- no severe short-term reversal immediately before entry.

These overlays should be measured after the base positivity-plus-near-high signal
is measured, so the research can identify where the alpha comes from.

## Event Queue

The strategy manages a small queue of active candidates rather than a broad
stock list.

Initial queue sizes:

```text
N = 1
N = 3
N = 5
```

`N=3` and `N=5` are the main operating candidates. `N=1` is included to measure
whether the signal concentrates well enough for very selective active bets.

Queue behavior:

1. A stock with a valid entry signal enters the candidate queue.
2. If active slots are available, the best candidate becomes active.
3. If all slots are full, a candidate can replace an active name only when its
   signal score exceeds the weakest active name by a configured margin.
4. Replacement is treated as an exit from the old active name and entry into the
   new name.

The replacement margin is required to prevent excessive churn.

## Signal Score

The first research score should be simple and inspectable:

```text
score =
  positivity_component
+ near_high_component
+ optional_revision_component
+ optional_flow_component
```

Recommended ordering for research:

1. `positivity + near_high`
2. `positivity + breakout`
3. `positivity + near_high + revision`
4. `positivity + near_high + flow`
5. `positivity + near_high + revision + flow`

The score is used only to rank candidates and resolve queue replacement. It
should not set position size in the first pass.

## Volatility Stop

Stops should be volatility-based rather than fixed percentage stops.

Initial stop definition:

```text
stop_price = entry_price - k * ATR
```

Initial `k` grid:

```text
k = 2.0
k = 2.5
k = 3.0
```

ATR should use an established daily window such as 14 or 20 trading days. The
research pass should test both if the data path already supports them cleanly;
otherwise use 20 trading days as the default because it aligns with a monthly
market horizon.

The stop is evaluated while the position is active. When hit, the stock exits
the active queue and returns to benchmark weight in the eventual overlay
interpretation.

## Sell Signals

Sell signals are separate from the volatility stop.

Initial sell-signal candidates:

- positivity rank falls out of the eligible group,
- close loses the near-high structure by a wider threshold,
- breakout failure after entry,
- OP or EPS revision turns negative if revision confirmation was part of entry,
- a stronger candidate replaces the active name.

Recommended first-pass near-high failure definition:

```text
close / rolling_252d_high < 0.90
```

This creates hysteresis:

```text
entry allowed at >= 0.95 of 252-day high
exit triggered below 0.90 of 252-day high
```

The gap reduces whipsaw from small moves around the entry threshold.

## Research Outputs

The first research pass should report alpha existence before implementation
details:

- event count,
- active days,
- average holding period,
- turnover / replacement count,
- win rate by event,
- average forward return by event,
- CAGR and drawdown of an equal-notional research sleeve,
- benchmark-relative alpha of the active sleeve,
- results by queue size `N=1,3,5`,
- results by stop multiplier `k=2.0,2.5,3.0`,
- results by entry variant: near-high vs exact breakout,
- early-period and late-period split stability.

The equal-notional sleeve is only a research device. It is not the final
portfolio construction.

## Acceptance Criteria

The research supports moving to implementation only if:

- `N=3` or `N=5` produces positive benchmark-relative alpha after costs,
- the result is not dependent on a single stock or a single short window,
- volatility stops improve drawdown or left-tail behavior without eliminating
  most upside,
- event count is large enough to evaluate the rule,
- turnover is compatible with index overlay management,
- the base signal's contribution is distinguishable from optional overlays.

If alpha appears only when `N` is large, the strategy is not suitable for the
intended management style. If alpha appears only for exact breakouts with very
few events, the strategy needs a broader entry definition or should be rejected.

## Implementation Boundary

The first implementation plan should build a research runner, not a production
portfolio optimizer.

Likely components:

```text
event detector
candidate scorer
event queue simulator
ATR stop evaluator
sell-signal evaluator
research summary exporter
```

The existing positivity research code should be reused where possible. The
runner should produce transparent event logs so the strategy can be audited name
by name.

## Verification Plan

Minimum verification after implementation:

- unit tests for entry event detection,
- unit tests for ATR stop behavior,
- unit tests for queue capacity and replacement margin,
- unit tests for sell-signal exits,
- smoke test on a small real-data window,
- full research run that writes event logs and summary tables.

The implementation should not be considered complete until the research output
can answer whether the alpha survives for `N=3` and `N=5`.
