# KOSPI200 Signal Event Rotation Design

## Goal

Build a registered KOSPI200 strategy family that seeks alpha from economically interpretable signal events: operating-profit consensus improvement, sector relative-strength/OP-cycle confirmation, foreign/institutional flow support, and price confirmation such as new highs or reclaim events.

## Constraints

- No parameter fitting. Candidate values are fixed, coarse, and economically motivated before backtesting.
- Use KOSPI200 membership only.
- Use next-open execution with realistic fee, tax, and slippage in research sweeps.
- Keep every candidate reproducible from code and saved artifacts.
- Register only a small selected strategy surface; keep broad variants in a research runner.

## Strategy Shape

The production strategy is `signal_event_rotation`. It builds stock-level OP revision scores, maps sector price RRG and OP RRG states back to constituents, applies a discrete event gate, optionally requires investor-flow confirmation, then ramps participation after an event instead of entering at full size immediately.

The research runner tests 500 predeclared candidates:

- 5 score modes: quarterly average OP revision, next-fiscal-year OP revision used as the OP12 proxy when 12M data is unavailable, blended, acceleration, EPS+OP agreement.
- 5 event modes: revision cross-up, revision acceleration, sector turn, 252-day high with revision, moving-average reclaim with revision.
- 5 flow gates: none, smart-money positive, foreign positive, institution positive, retail contra.
- 4 construction modes: one, two, three, or breadth-weighted leaders per active sector.
- 5 risk modes: long-only and four predeclared long/short gross-short levels.

## Selection Rule

Selection is not based on fitting a threshold. Rank candidates by a robustness score that rewards Sharpe, CAGR, benchmark monthly win rate, and drawdown control, while flagging turnover/name-count concerns. The selected registered strategy should favor high risk-adjusted return and acceptable drawdown rather than maximum final equity.

## Validation

Unit tests cover variant count/uniqueness, event participation ramp behavior, strategy registration, dataset requirements, and basic portfolio construction constraints. Research validation writes per-run artifacts under `results/backtests/` and aggregate summaries under `results/signal_event_research/`.
