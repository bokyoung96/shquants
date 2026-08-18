# KOSPI200 Double-Bottom Staged-Buy Design

## Objective

Measure whether staged buying after a confirmed double-bottom pattern improves the forward performance and drawdown profile of a KOSPI200 index exposure. The analysis must be reproducible from the repository's local KOSPI200 minute parquet source and must expose the exact signal, fill, holding, cost, and comparison definitions.

## Data contract

- Source: `parquet/KOSPI200_1m.parquet`.
- Convert UTC timestamps to `Asia/Seoul`, group by local calendar date, and derive daily OHLC from the first open, maximum high, minimum low, and last close.
- The source is an index/futures-style continuous series; the report must identify it as KOSPI200 index exposure and must not imply constituent-level or dividend-adjusted stock returns.

## Signal definition

1. A pivot low is a daily low that is no higher than the lows of the two preceding and two following trading sessions. The pivot is usable only after those two following sessions have printed.
2. For each confirmed first pivot, search later confirmed pivots 10–60 trading sessions away.
3. The second pivot must be within ±3% of the first pivot low.
4. The neckline is the maximum daily high strictly between the two pivots. The base case requires an interim rebound of at least 10% from the first pivot low to the neckline; 15%, 20%, 25%, and 30% are reported as sensitivity thresholds because the original 30% threshold produces no candidates in the available KOSPI200 sample.
5. The pattern becomes actionable on the first later close above the neckline. The entry fill is the next trading session's open, preventing same-bar look-ahead.
6. After a signal is selected, ignore later signals until the trade's 60-session holding window ends. This creates an event-level and a non-overlapping portfolio view.

## Execution schemes

- `lump_sum`: 100% of the target notional at the next-session open.
- `staged_50_25_25`: 50% at the next-session open, 25% at the open five sessions later, and 25% at the open ten sessions later. Missing future fills are held as cash and are not silently backfilled.
- Each fill incurs 5 bps one-way cost. The strategy is long-only, uses no leverage, and has no stop-loss or take-profit; positions are valued at each daily close and fully liquidated at the 60th close after the first fill. The cost assumption is configurable.

## Outputs

- Daily derived OHLC and signal ledger with pivot dates, neckline, entry/fill dates, fill prices, and validation fields.
- Event-level forward returns at 20, 60, and 120 sessions for both schemes and an unconditioned KOSPI200 buy-and-hold reference from each entry date.
- Non-overlapping portfolio equity, returns, turnover, and summary metrics: total return, CAGR, annualized volatility, Sharpe (0% risk-free), maximum drawdown, win rate, trade count, and average invested fraction.
- A Korean Markdown report and CSV files under `results/kospi200_double_bottom/`, plus an equity/pattern chart when matplotlib is available.

## Validation

- Unit tests cover local-date aggregation, pivot confirmation, neckline/rebound and tolerance filters, next-open execution, staged fill timing, missing fill handling, and non-overlap selection.
- A smoke run against the local parquet source must produce non-empty daily data and a machine-readable summary. The report must state the exact source period and all assumptions.

## Known limitations

The continuous KOSPI200 source may contain contract-roll or data-quality artifacts. This study is an index-pattern timing study, not a tradable constituent portfolio, and results are sensitive to the pattern tolerances, fixed holding horizon, and cost assumption. A later extension can add walk-forward parameter selection or a directly investable ETF/futures contract.
