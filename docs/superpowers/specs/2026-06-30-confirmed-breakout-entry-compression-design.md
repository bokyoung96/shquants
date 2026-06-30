# Confirmed Breakout Entry Compression Design

Date: 2026-06-30

## Intent

Reduce the number of executed entries in the 5-minute KOSPI200 breakout strategy without introducing arbitrary quality-score blending.

The current baseline already combines:

- 5-minute close breakout above the prior 52-week daily close high
- positive positivity spread versus the configured benchmark
- positive foreign or institutional trailing flow

The next experiment should not add weighted multi-factor scores. The goal is to test whether stricter signal structure can remove weak repeat or fake breakout entries while preserving the right-tail winners that drive the strategy.

## Core Hypotheses

### H1: Confirmed Breakout Reduces Fake Breakouts

The baseline enters after the first 5-minute bar closes above the prior 52-week daily close high.

The confirmed variant waits for one additional 5-minute close above the same breakout level before entering.

```text
baseline:
first 5-minute close > prior 52-week daily close high
-> enter at next 5-minute open

confirmed_breakout:
first 5-minute close > prior 52-week daily close high
next 5-minute close remains > same prior 52-week daily close high
-> enter at following 5-minute open
```

This tests a single structural idea: real breakouts should persist for at least one more bar, while many false breakouts should fail immediately.

### H2: Episode Compression Removes Duplicate Ideas

Multiple signals from the same ticker during the same breakout regime should not be treated as independent ideas.

An episode starts when a ticker produces a confirmed breakout. After an episode starts, later signals for the same ticker are ignored until the ticker resets.

Recommended reset rule:

```text
allow a new episode only after:
daily close <= prior 52-week daily close high
then a new confirmed breakout occurs
```

This is not an alpha filter. It is a position-management interpretation: one breakout regime should produce at most one entry attempt.

## Non-Goals

This experiment should not:

- create a weighted `quality_score`
- optimize arbitrary factor weights
- introduce daily max-entry caps
- introduce sector max-entry caps
- strengthen the flow filter to foreign-and-institution by default
- change the KOSPI200 historical universe
- change the existing positivity spread or flow definitions
- change exit logic unless required to measure the entry experiment correctly

## Experiment Arms

### A. Baseline

Use the current flow-filtered breakout baseline:

- 5-minute first-close breakout
- positivity spread margin from the existing config
- foreign or institutional positive flow
- continuation holding exit logic

### B. Confirmed Breakout

Use the same filters and exits as baseline, but require one additional 5-minute close above the prior 52-week high before entry.

### C. Confirmed Breakout With Episode Compression

Use confirmed breakout and suppress repeated same-ticker entries during the same breakout episode.

The reset condition is daily close falling back to or below the prior 52-week daily close high, followed by a new confirmed breakout.

## Evaluation

The comparison should focus on whether entry compression improves tradability without destroying the return distribution.

Required metrics:

- total entries
- entry reduction versus baseline
- unique active tickers
- average net bps
- median net bps
- hit rate
- profit factor
- max drawdown
- final equity, treated only as relative research evidence
- average holding days
- average and max concurrent positions
- yearly entry counts
- score-free right-tail preservation:
  - baseline top 5% winner threshold
  - percentage of compressed-strategy winners above that threshold
  - total net return contribution from trades above that threshold

The key acceptance signal is not simply higher final equity. A useful compressed strategy should show materially fewer entries, better or similar profit factor, and no severe loss of right-tail contribution.

## Interpretation Rules

If confirmed breakout reduces entries and improves profit factor, it is evidence that immediate fake breakouts are a meaningful source of noise.

If confirmed breakout reduces entries but also removes too many large winners, the delay is too expensive and should not be used.

If episode compression reduces entries with limited right-tail damage, it is a good operations rule even if average bps changes only modestly.

If both confirmed breakout and episode compression improve average bps but reduce final equity, the strategy may still be preferable for discretionary or index-fund-style management because the primary objective is manageable entry count, not unconstrained research compounding.

## Implementation Shape

The implementation should extend the existing flow-filtered breakout path rather than creating a separate strategy framework.

Likely touchpoints:

- `scripts/run_flow_filtered_breakout_single.py`
- tests under `tests/scripts/test_run_flow_filtered_breakout_single.py`
- comparison utilities if new metrics are needed

The entry mode should be explicit in config or CLI output so result folders can be audited:

```text
entry_confirmation = first_close | next_close_confirmed
episode_compression = true | false
```

Existing baseline behavior must remain reproducible.

