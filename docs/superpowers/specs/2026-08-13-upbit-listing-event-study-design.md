# Upbit Listing Short Event Study Design

## Goal

Build a reproducible event study that measures the net short return available on
Binance USD-M perpetual futures around Upbit listing announcements. The study is
descriptive research, not yet a trading strategy. Its primary question is:

> If one unit of notional were shorted immediately after an Upbit listing event,
> what net return and intraperiod price path would have followed?

The first release covers Binance only. Stop-loss, take-profit, leverage,
portfolio sizing, overlapping-position capital constraints, and statistical
significance tests are explicitly deferred.

## Research Contract

### Unit of observation

Each Upbit asset-notice pair is one base case, and each
asset-notice-event-type row is one analysis observation. When one notice lists
several assets, every asset remains a separate observation and a separate
hypothetical short. No notice-level clustering, aggregation, p-values,
t-statistics, or confidence intervals are part of this study.

The report retains the notice UUID so multi-asset notices remain identifiable,
but every included asset receives equal weight in the primary descriptive
summary.

### Event definitions

Every eligible asset may generate two independent event rows:

1. `ANNOUNCEMENT`: the notice's immutable `first_listed_at` timestamp.
2. `SCHEDULED_SUPPORT`: the trading-support time stated in the original notice
   body as it existed at first publication.

Both event timestamps are normalized from KST (`Asia/Seoul`, UTC+09:00) to UTC
before joining Binance data. `listed_at` is retained only as revision metadata;
it must never replace `first_listed_at` in an event row.

The scheduled-support event always uses the original published schedule. A
later postponement or acceleration does not rewrite that event. Revision
timestamps and revised schedules may be retained for audit, but they are not
used as entry signals in this release.

### Entry and horizons

The entry time is the open of the one-minute candle whose timestamp is the event
time rounded upward to an exact minute:

- `11:11:38` enters at the `11:12:00` one-minute open.
- `12:30:00` enters at the `12:30:00` one-minute open.

The Binance contract must have exact one-minute candles at `entry_time - 1
minute` and at `entry_time`. The first candle proves that the perpetual contract
was already trading during the minute containing the event; the second supplies
the entry price. Missing or delayed entry candles fail closed, and the pipeline
does not search forward for a more convenient price.

Exit prices are the one-minute opens exactly 5, 15, 30, 60, 240, and 1,440
minutes after the actual entry timestamp. A missing exact exit candle makes only
that horizon unavailable.

For a linear USD-M short with entry price `P0` and exit price `P1`, gross return
on entry notional is:

```text
gross_short_return = (P0 - P1) / P0
```

No leverage multiplier is applied.

### Costs and primary outcome

The primary ranking metric is equal-notional mean net short return per
asset-event and horizon. Every event row contains three return variants:

1. `gross_short_return`: no costs.
2. `fee_only_short_return`: 5 basis points per side, 10 basis points round trip.
3. `net_short_return`: the primary scenario, 5 basis points of fee plus 5 basis
   points of slippage per side, 20 basis points round trip.

Fee and slippage assumptions are command-line parameters, and the report prints
their values prominently. The defaults are research assumptions rather than a
claim about every account tier or every historical fill.

For every horizon, the study also reports the signed intraperiod excursions:

```text
MFE = (entry_price - minimum_low) / entry_price
MAE = (entry_price - maximum_high) / entry_price
```

MFE is favorable to the short and normally positive. MAE is adverse to the
short and normally negative. Excursions are gross path diagnostics; fees and
slippage are not deducted from them.

### Market-relative diagnostic

Raw net short profit is the primary output. A secondary BTC-relative diagnostic
uses BTCUSDT over the same entry and exit timestamps:

```text
btc_relative_short_return = btc_long_return - asset_long_return
```

This shows whether the listed asset underperformed BTC without replacing the
actual tradable short P&L. It is descriptive only and is not a hedge simulation.

## Upbit Source Boundary

Playwright's asynchronous request context is the collection boundary. A run
first visits the public notice page and then uses the JSON endpoints observed
from the rendered application:

```text
https://www.upbit.com/service_center/notice
https://pub-info.upbit.com/api/v1/categories?os=web
https://pub-info.upbit.com/api/v1/announcements?os=web&page={page}&per_page=20&category=trade
https://pub-info.upbit.com/api/v1/announcements/{uuid}
```

The observed `trade` category currently exposes 749 notices across 38 pages,
reaching back to October 2017. Counts are discovery evidence, not hard-coded
acceptance criteria.

The collector follows pagination until the API-reported final page. It uses one
Playwright request context, bounded retry, a global minimum delay, and an
explicit timeout. Successful list and detail responses are cached before
parsing. Normal reruns use valid cached responses; an explicit refresh fetches
again and creates a new immutable detail snapshot rather than overwriting an
older body.

Raw cache manifests record source URL, request parameters, HTTP retrieval time,
response SHA-256, schema version, and notice UUID. This makes later revisions
observable on prospective runs.

## Listing Notice Selection

The pipeline starts with the `trade` category but does not treat every trade
notice as a listing. Candidate titles include current and historical listing
families such as:

- `신규 거래지원 안내`
- `마켓 디지털 자산 추가`
- historical `상장` wording

Final inclusion requires an original-body statement that the asset is being
newly added or supported. Warnings, support termination, market-policy changes,
and generic trading notices are excluded even if their titles contain related
words.

The original body parser extracts one row per listed asset:

```text
notice_id
notice_uuid
title
asset_name
upbit_ticker
upbit_markets
network
first_listed_at_kst
listed_at_kst
original_scheduled_at_kst
original_scheduled_text
revision_detected
source_confidence
detail_snapshot_sha256
```

## Point-in-Time and Revision Safety

Historical API responses expose the current notice body, so the parser must not
pretend every historical body is a first-publication snapshot. Source confidence
is explicit:

- `UNMODIFIED_CURRENT`: `listed_at == first_listed_at` and the current body is
  used as the original body.
- `RECONSTRUCTED_ORIGINAL`: the body contains explicit update blocks above a
  preserved original section; only the preserved original section is parsed.
- `FIRST_SEEN_SNAPSHOT`: a prospective immutable snapshot captured while
  `listed_at == first_listed_at`; later revisions cannot overwrite it.
- `UNRESOLVED_ORIGINAL`: the original section or original schedule cannot be
  recovered without guessing.

`UNRESOLVED_ORIGINAL` rows may still produce an `ANNOUNCEMENT` event when the
asset identity is unambiguous because `first_listed_at` comes from list metadata.
They cannot produce a `SCHEDULED_SUPPORT` event. The parser never treats a
revised schedule as the original schedule and never infers a missing time from
nearby prose.

For Korean month/day/time strings without a year, the year starts from
`first_listed_at`. A parsed schedule earlier than the announcement is allowed
only for a validated year rollover into the following calendar year; all other
backward schedules are rejected.

## Binance Eligibility and Market Data

The first release targets Binance USD-M perpetual futures only. Historical
one-minute OHLCV comes first from Binance's official public-data archive under
`data/futures/um/`; downloaded ZIP files are accepted only after their published
checksum passes. The public `/fapi/v1/klines` endpoint is a bounded fallback for
recent dates not yet present in the archive. Current public contract metadata is
cached for matching support, but it is not treated as a complete history of
delisted contracts. No credentials or private trading endpoints are used.

Asset matching is fail-closed:

1. Prefer an exact `{ticker}USDT` symbol supported by current USD-M contract
   metadata or by an official USD-M archive containing the required historical
   candles.
2. Multiplier contracts such as `1000{ticker}USDT` require an explicit verified
   alias recorded in a small source-controlled mapping file.
3. Rebrands, ticker collisions, and ambiguous candidates remain unresolved.
4. The `entry_time - 1 minute` and exact entry-minute candle requirements are
   the final empirical proof that the matched contract was tradable at the
   event.

An unresolved, non-perpetual, not-yet-trading, or missing-candle asset is
excluded with a machine-readable reason. It is never silently replaced with a
spot, margin, delivery-futures, or different-token market.

Cached Binance rows preserve open time, open, high, low, close, volume, quote
volume, trade count, and source symbol. UTC millisecond timestamps are converted
to timezone-aware pandas timestamps without dropping timezone information.

External contracts used by the implementation are the [Binance USD-M API
documentation](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/Introduction)
and Binance's [official public-data archive
documentation](https://github.com/binance/binance-public-data). Endpoint and
archive schema checks remain runtime validations because these external
contracts can change.

## Components and Repository Shape

Implementation belongs to a focused research package:

```text
backtesting/strategies/upbit_listing_event/
    __init__.py
    models.py
    upbit_client.py
    upbit_parser.py
    binance_client.py
    matching.py
    study.py
    report.py
scripts/run_upbit_listing_event_study.py
tests/strategies/upbit_listing_event/
```

Responsibilities are separated as follows:

- `models.py` defines immutable notice, asset, event, market-match, exclusion,
  and cost assumptions.
- `upbit_client.py` owns the Playwright request context, pagination, retry,
  throttling, immutable JSON cache, and manifests. It follows the proven
  transport patterns in `kind/client.py` without coupling the two domains.
- `upbit_parser.py` owns title-family filtering, body revision splitting,
  original table extraction, Korean datetime parsing, and confidence labels.
- `binance_client.py` owns official USD-M archive retrieval and checksum
  verification, the bounded recent-data REST fallback, public contract
  metadata, caching, and response validation.
- `matching.py` resolves Upbit tickers to verified Binance perpetual symbols and
  materializes every rejection reason.
- `study.py` creates the two event types, enforces exact candle availability,
  and calculates forward returns, MFE, MAE, costs, and BTC-relative diagnostics.
- `report.py` writes deterministic data tables, an Excel workbook, Markdown
  findings, and charts.
- The script is a thin CLI adapter and owns no research calculations.

No new dependency is required. The repository already provides Playwright,
pandas, numpy, matplotlib, openpyxl, and the relevant testing stack.

## Output Contract

Default outputs are written under:

```text
results/upbit_listing_event_study/
    notice_assets.csv
    events.csv
    exclusions.csv
    forward_returns.csv
    horizon_summary.csv
    yearly_summary.csv
    liquidity_summary.csv
    upbit_listing_event_study.xlsx
    report.md
    event_return_paths.png
    horizon_net_returns.png
```

`events.csv` has one row per asset and event type. `forward_returns.csv` has one
row per asset-event with all six horizons as explicit columns. Raw amounts are
stored as decimal returns; report surfaces render percentages.

The primary horizon table contains:

```text
event_type
horizon_minutes
events
mean_net_short_return
median_net_short_return
win_rate
p10
p25
p75
p90
worst_return
best_return
mean_mfe
mean_mae
total_equal_notional_return
btc_relative_short_return
```

These are descriptive quantities only. The report contains no significance
stars, hypothesis tests, clustered errors, confidence intervals, or language
claiming that historical profit guarantees future profit.

The exclusion table includes at least:

```text
NOT_A_LISTING
ASSET_PARSE_FAILED
ORIGINAL_SCHEDULE_UNRESOLVED
BINANCE_SYMBOL_UNRESOLVED
BINANCE_PERPETUAL_NOT_ACTIVE
PRE_EVENT_CANDLE_MISSING
ENTRY_CANDLE_MISSING
EXIT_CANDLE_MISSING
BINANCE_SCHEMA_ERROR
UPBIT_SCHEMA_ERROR
```

Missing exits are recorded per horizon so shorter valid horizons remain usable.

## Command Surface

The checkout-local command is:

```powershell
uv run python scripts/run_upbit_listing_event_study.py `
  --output-dir results/upbit_listing_event_study
```

Operational flags are limited to cache/output paths, refresh controls, request
concurrency, request delay, timeout, fee basis points per side, and slippage
basis points per side. Defaults favor auditable recovery and respectful request
volume over maximum crawl speed.

## Validation Rules

The pipeline fails closed at each external boundary:

1. Every notice list response matches the expected success/data/page schema.
2. Detail UUIDs match their list records.
3. Every included asset has a non-empty ticker and evidence from the original
   listing section.
4. `ANNOUNCEMENT` uses `first_listed_at`, never `listed_at`.
5. `SCHEDULED_SUPPORT` uses only an original, confidence-qualified schedule.
6. Every event timestamp is timezone-aware and has a deterministic UTC value.
7. Every Binance match is a USD-M perpetual and has exact pre-event and entry
   candles.
8. Entry and exit calculations use one-minute opens; MFE and MAE use highs and
   lows inside the same event window.
9. Cost arithmetic is identical across rows and printed in output metadata.
10. Included plus excluded candidates reconcile to the discovered candidate
    population without silent row loss.

Schema errors are materialized in exclusions and logs. A run may still write
inspectable partial outputs, but exits non-zero when source schema validation or
population reconciliation fails.

## Test Strategy

Development follows red-green-refactor. Deterministic tests cover:

- Upbit list pagination and cache manifests through an injected fake transport.
- Current, historical, multi-asset, updated, and malformed notice fixtures.
- Separation of appended update blocks from the preserved original body.
- `first_listed_at` versus `listed_at` look-ahead protection.
- Korean datetime parsing, exact-minute behavior, and year rollover.
- One asset row per coin in a multi-coin notice.
- Exact, multiplier-alias, ambiguous, rebrand, and missing Binance matches.
- Official USD-M archive checksum validation and recent-data REST fallback.
- Proof-of-tradability using `entry_time - 1 minute` and the entry-minute
  candles.
- Exact entry/exit open selection at all six horizons.
- Linear short return, fee, slippage, BTC-relative return, MFE, and MAE math.
- Partial-horizon exclusion without discarding otherwise valid rows.
- Deterministic CSV/Excel/report output from cached fixtures.

A bounded live Playwright smoke test then verifies the current Upbit category,
list, and detail endpoints and one Binance market-data request. Live tests are
integration evidence, not substitutes for deterministic fixtures.

## Deferred Work

The following work requires a separate design after the event-study results are
visible:

- Stop-loss and take-profit rules.
- ATR or other volatility-normalized exits.
- Leverage, liquidation, funding, margin, or account-level sizing.
- Simultaneous-event portfolio capital allocation.
- Other futures venues such as Bybit or OKX.
- Automated scheduling or live trade execution.
- Statistical inference or causal claims.

## Completion Criteria

The first release is complete only when:

- Historical Upbit trade notices are cached and parsed through Playwright.
- Eligible listing assets generate separate announcement and original-schedule
  events when their required timestamps are recoverable.
- Only verified, already-trading Binance USD-M perpetuals enter the study.
- All six forward horizons, gross/fee/net returns, MFE, MAE, and BTC-relative
  diagnostics are calculated without look-ahead.
- Every omitted asset or horizon has an explicit exclusion reason.
- The deterministic test suite and bounded live smoke checks pass.
- The result tables, workbook, Markdown report, and charts are written under the
  documented output directory.
- The report ranks results by net short profit and does not introduce deferred
  strategy optimization.
