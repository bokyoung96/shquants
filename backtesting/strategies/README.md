# Strategy Registry

This folder keeps only strategies that are currently worth running from the CLI,
execution specs, or dashboard.

## Naming Scheme

Strategy names should be short, descriptive, and close to the actual investment
logic:

- `trend_rank`: price trend ranking.
- `earnings_revision`: forward EPS/OP consensus revision.
- `revision_signal`: signal-triggered earnings revision with market trend risk-off.
- `mfbt`: multi-factor backtest scaffold with price, earnings, dividend, and retail-flow signals.
- `benchmark_overlay`: benchmark-weighted portfolio with a soft active overlay.
- `benchmark_tilt`: benchmark-weighted portfolio tilted by revision, flow, and trend.
- `op_rrg_strat`: final OP RRG sector strategy using one-day-lagged month-end one-month OP revisions.
- `rrg_sector_rotation`: RRG sector rotation with OP revision confirmation and a weak short sleeve.
- `rrg_sector_rotation_prune90`: RRG sector rotation with sector-preserving small-position pruning.
- `rrg_sector_rotation_op_rrg_k2`: price RRG confirmed by OP RRG, compressed to two long leaders and one short leader per active sector.
- `rrg_sector_rotation_op_rrg_k1`: price RRG confirmed by OP RRG, compressed to one long leader and one short leader per active sector.
- `rrg_sector_rotation_op_rrg_ex10_k2`: OP RRG K2 with all BM-weight-above-10% names excluded from OP RRG calculation.
- `rrg_sector_rotation_op_rrg_ex10_k1`: OP RRG K1 with all BM-weight-above-10% names excluded from OP RRG calculation.
- `signal_event_rotation`: KOSPI200 OP-consensus signal events with sector price/OP RRG confirmation, flow gates, and staged participation.
- `signal_event_rotation_selected`: selected 500-candidate signal-event variant, fixed to OP12-proxy acceleration, retail-contra confirmation, K2 compression, and 0.3 gross short.

Each strategy entry below follows the same schema:

- `id`: value passed to `--strategy` or `ExecutionSpec.strategy`.
- `file`: implementation module under `backtesting/strategies/`.
- `class`: public strategy class registered in `registry.py`.
- `profile`: reporting/evaluation style.
- `data`: semantic frames expected from `DataLoader`.
- `signal`: alpha source.
- `construction`: portfolio construction shape.
- `use`: when to prefer the strategy.

Older names are intentionally not kept as aliases. Specs and dashboard presets
should use the current `id` values.

## Active Strategies

### Trend Rank

- `id`: `trend_rank`
- `file`: `trend_rank.py`
- `class`: `TrendRank`
- `profile`: absolute or benchmark-relative, depending on the report config.
- `data`: `close`
- `signal`: trailing price return rank over `lookback`.
- `construction`: long-only top-N equal weight through `LongOnlyTopN`.
- `use`: fast baseline strategy and smoke test for data, execution, reporting, and dashboard flows.

### Earnings Revision

- `id`: `earnings_revision`
- `file`: `earnings_revision.py`
- `class`: `EarningsRevision`
- `profile`: absolute or benchmark-relative, depending on the report config.
- `data`: `close`, `eps_fwd_q1`, `op_fwd_q1`
- `signal`: EPS and OP forward revision ranks, restricted to names where both revisions are positive.
- `construction`: long-only top-N equal weight through `LongOnlyTopN`.
- `use`: simple long-only consensus revision strategy. Prefer daily or weekly rebalancing; the monthly smoke window can miss active dates and understate this strategy.

### Revision Signal

- `id`: `revision_signal`
- `file`: `revision_signal.py`
- `class`: `RevisionSignal`
- `profile`: absolute or benchmark-relative, depending on the report config.
- `data`: `close`, `benchmark`, `eps_fwd_q1`, `op_fwd_q1`
- `signal`: hold every KOSPI200 name where both EPS and OP forward revisions are positive over `lookback`; move to cash when KOSPI200 is below its fixed 120-day trend average.
- `construction`: long-only equal weight across all currently passing signal names; no top-N rank cap.
- `use`: signal-triggered KOSPI200 strategy for lower drawdown than concentrated revision ranking. Prefer the `kospi200_revision_signal` preset so `signal_dates` rebalances only when target weights change.

### MFBT

- `id`: `mfbt`
- `file`: `mfbt.py`
- `class`: `Mfbt`
- `profile`: absolute or benchmark-relative, depending on the report config.
- `data`: `close`, `op_fwd_12m`, `dps_ttm`, `dividend_cash_ttm`, `retail_flow`, `sector_big` from daily `qw_wi_sec_26_big`, `market_cap`, `free_cash_flow`, `interest_bearing_liability`, `quick_asset`
- `universe`: factor scores and cross-sectional buckets use the active strategy universe, typically `legacy_k200`. Non-universe names remain `NaN` in factor metadata.
- `frequency`: all factor outputs are month-end observations. Non-month-end rows remain `NaN` in factor metadata and alpha. The final `mfbt` factor metadata is additionally masked to dates where every required factor has at least one computable signal; ticker-level availability remains factor-specific, so missing raw inputs stay `NaN` instead of becoming a `0.0` score.
- `signal`: `price_momentum`, a month-end binary factor set to `1.0` when `close / close.rolling(252).max() > 0.8`; otherwise `0.0`. It requires a full 252-trading-day close history. Earlier month-end rows stay `NaN`; a valid ratio at or below the threshold is the real `0.0` score.
- `signal`: `earnings_momentum`, a 0-4 monthly cross-sectional score based on month-end 12MF operating-profit estimate growth, with missing current/previous consensus kept as `NaN` and extreme growth for `op_fwd_12m < 100bn` filtered to `0.0` before scoring. Non-month-end rows are `NaN`.
- `signal`: `dividend_yield`, a 0-4 monthly cross-sectional score based on month-end `dps_ttm / close`, plus `1.0` when same-month `dividend_cash_ttm` values from 24 months ago, 12 months ago, and the signal month increased consecutively. Missing dividend yield inputs stay `NaN`; non-month-end rows are `NaN`.
- `signal`: `retail_flow`, a 0-4 monthly WI26 big-sector score from each sector's average 252-day cumulative retail net flow inside the active universe. Larger average retail net selling gets the higher score, then the sector score is assigned to member names. Non-month-end rows are `NaN`. The sector statistic is an average, not a sum, so sectors with more K200 constituents are not mechanically pushed toward larger absolute flow values.
- `retail_flow` size note: the source flow is still an unscaled currency amount, not a market-cap-normalized or trading-value-normalized ratio. Averaging removes the constituent-count bias, but large-cap and highly liquid names can still make their sector's average absolute flow larger than sectors dominated by smaller names. Treat the factor as a sector-level retail pressure signal with residual size/liquidity characteristics.
- `signal`: `value`, a 0-4 monthly cross-sectional score based on `free_cash_flow / TEV`, where `TEV = market_cap + interest_bearing_liability - quick_asset`. Market cap uses the signal month-end value. Financial inputs use fixed source-month windows: April-May signals use March-end data, June-August use May-end data, September-November use August-end data, and December-March use November-end data.
- `value` data policy: missing `free_cash_flow`, `interest_bearing_liability`, or `quick_asset` leaves the final score as `NaN`. `TEV <= 0` is not excluded; it is forced into the lowest value metric before scoring so negative-TEV edge cases cannot receive a high FCF/TEV score. A 2026-02-27 K200 check showed G40 financials had 0/21 non-null `free_cash_flow`, 0/21 non-null `quick_asset`, and 0/21 computable TEV, so financial names are expected to fall out as `NaN` under this definition.
- `construction`: long-only equal weight across selected `price_momentum == 1.0` names, capped by `top_n`.
- `use`: multi-factor backtest scaffold. Current portfolio selection still uses `price_momentum` as the alpha while preserving factor scores in signal metadata for later combination.

### Benchmark Overlay

- `id`: `benchmark_overlay`
- `file`: `benchmark_overlay.py`
- `class`: `BenchmarkOverlay`
- `profile`: index
- `data`: `close`, `benchmark`, `eps_fwd_q1`, `op_fwd_q1`, `foreign_flow`, `inst_flow`, `retail_flow`, `sector_big`, `market_cap`, `k200_yn`
- `signal`: consensus revision, order imbalance, and benchmark beta context.
- `construction`: K200 market-cap base with active overlay caps by stock and sector.
- `use`: benchmark-aware participation where staying close to the index matters more than raw concentration.

### Benchmark Tilt

- `id`: `benchmark_tilt`
- `file`: `benchmark_tilt.py`
- `class`: `BenchmarkTilt`
- `profile`: index
- `data`: `close`, `benchmark`, `eps_fwd_q1`, `op_fwd_q1`, `foreign_flow`, `inst_flow`, `retail_flow`, `sector_big`, `market_cap`, `k200_yn`
- `signal`: consensus revision, institution/foreign-vs-retail flow, and beta-adjusted trend.
- `construction`: K200 market-cap base with active-share target and stock/sector active caps.
- `use`: index-like mandate that should lean toward improving consensus and supportive flow without leaving the benchmark too far behind.

### `rrg_sector_rotation`

- `profile`: concentrated long plus weak-hedge long-short stock strategy.
- `data`: `close`, `benchmark`, `sector_big` from daily `qw_wi_sec_26_big`, K200 membership, market cap/float market cap, and forward OP estimates.
- `signal`: RRG state is a regime gate; stock and cap-weighted sector OP forward revision are the confirmation signals. Investor flow is not used.
- `long sleeve`: `Leading`, `Improving`, or `Weakening` sectors whose sector and stock OP revisions are both positive.
- `short sleeve`: `Lagging` sectors whose sector and stock OP revisions are both negative.
- `weighting`: selected long and short sleeves are weighted by cross-sectional OP revision rank, not equal weight or top-N count.
- `concentration`: optional quantile/min-revision/max-name controls can reduce holdings for personal-account execution without forcing a fixed top-N portfolio. Suggested experiment: `long_quantile=0.70`, `short_quantile=0.70`, `min_long_revision=0.03`, `min_short_revision=0.03`, `max_long_names=20`, `max_short_names=5`.
- `exposure`: default `gross_long=1.0` and `gross_short=0.5`; neither long nor short sleeve is force-filled.
- `use`: selected RRG candidate after validation.

### `rrg_sector_rotation_prune90`

- `profile`: concentrated long plus weak-hedge long-short stock strategy.
- `data`: same as `rrg_sector_rotation`.
- `signal`: same RRG sector gate and OP revision confirmation as `rrg_sector_rotation`.
- `construction`: first builds the OP-rank long/short portfolio, compresses each active sector to at most two long names and one short name, then prunes small positions by side while preserving sector exposure. Long and short books each keep the names needed to explain 90% of absolute side exposure, always keep at least one name per active sector/side, and softly cap total names at 20 by removing the smallest non-protected positions.
- `weighting`: removed weights are redistributed within the same sector and same side, so sector rotation exposure is preserved while small operationally noisy positions are removed.
- `use`: personal-account variant of `rrg_sector_rotation` when the baseline has too many small positions but sector pinpoint concentration should be avoided.

### `rrg_sector_rotation_op_rrg_k2`

- `profile`: aggressive long plus weak-hedge long-short stock strategy.
- `data`: same price/sector/K200/market-cap inputs as `rrg_sector_rotation`, plus `op_fwd_12m` for OP RRG state.
- `price RRG`: WI26 sector relative price versus KOSPI200, classified with 126D relative strength and 42D/21D relative momentum.
- `OP RRG`: WI26 sector 12M forward OP share versus total KOSPI200 12M forward OP, classified with the same RRG state logic. Non-positive sector or market OP leaves that sector `Unclassified` for that date.
- `long sleeve`: price RRG state in `Leading`, `Improving`, or `Weakening`; OP RRG state in `Leading` or `Improving`; and stock qavg OP revision positive.
- `short sleeve`: price RRG state `Lagging`; OP RRG state `Lagging` or `Weakening`; and stock qavg OP revision negative.
- `construction`: builds OP-rank long/short weights and then preserves active sector exposure while keeping up to two long OP leaders and one short OP leader per active sector.
- `use`: alpha-forward RRG variant when OP cycle confirmation is preferred over the simpler sector OP sign gate.

### `op_rrg_strat`

- `profile`: final selected OP RRG sector strategy.
- `class`: `RrgSectorRotationOpRrgMonthly1M`.
- `data`: same as `rrg_sector_rotation_op_rrg_k2`.
- `signal`: same price RRG and OP RRG sector gates as `rrg_sector_rotation_op_rrg_k2`, but stock OP revision is computed from month-end OP consensus divided by previous month-end OP consensus minus one, averaged across Q1, Q2, and 12M forward OP. Both OP RRG state and stock OP revision are shifted by one trading day before they can affect target weights, so same-day consensus updates cannot be traded.
- `construction`: same K2 compression: after OP-rank long/short weights are built across the selected stock pool, each active sector keeps up to two long leaders and one short leader while preserving sector exposure.
- `use`: preferred RRG production/research candidate after the monthly OP revision and look-ahead checks. It better matches the monthly update cadence of OP consensus than the older 20-trading-day revision proxy.

### `rrg_sector_rotation_op_rrg_k1`

- `profile`: more concentrated version of `rrg_sector_rotation_op_rrg_k2`.
- `data`, `price RRG`, `OP RRG`, and sleeve gates`: same as `rrg_sector_rotation_op_rrg_k2`.
- `construction`: preserves active sector exposure while keeping one long OP leader and one short OP leader per active sector.
- `use`: concentrated research variant for smaller personal-account name count. It should be evaluated with drawdown sensitivity because single-name concentration is meaningfully higher than k2.

### `rrg_sector_rotation_op_rrg_ex10_k2` / `rrg_sector_rotation_op_rrg_ex10_k1`

- `profile`: diagnostic variants of `rrg_sector_rotation_op_rrg_k2` and `rrg_sector_rotation_op_rrg_k1`.
- `data`: same as the base OP RRG variants, plus daily KOSPI200 benchmark weights.
- `OP RRG exclusion`: before computing sector OP share and total market OP, every stock whose daily BM weight is greater than 10% is masked out. This excludes all such high-index-weight names date by date, not only Samsung Electronics.
- `construction`: same per-sector leader compression as the matching base variant.
- `use`: test whether mega-cap OP consensus is distorting the OP RRG denominator. The 2026-06-18 48-variant grid suggests this exclusion is mostly diagnostic rather than a preferred production path, because the main qavg ex10pct variants gave up Sharpe and return versus the base OP RRG.

### `signal_event_rotation`

- `profile`: KOSPI200 signal-event alpha sleeve for index-aware alpha research.
- `data`: adjusted close, KOSPI200 benchmark and membership, WICS big sector, market cap/float market cap, OP/EPS forward estimates, and foreign/institution/retail flow.
- `signal`: stock OP/EPS revision score gated by sector price RRG and sector OP RRG state. A position starts only after a discrete event such as revision cross-up, revision acceleration, sector turn, 252-day high, or moving-average reclaim.
- `flow`: optional confirmation from smart money, foreign, institution, or retail-contra flow normalized by market cap.
- `construction`: rank-weighted long sleeve with one/two/three leaders per active sector or breadth mode. Long/short risk modes are fixed coarse gross-short levels rather than fitted thresholds.
- `participation`: new long events ramp over fixed steps, so the strategy behaves like a band-trading entry rather than a same-day all-in signal.
- `use`: selected production surface for the 500-candidate signal-event sweep in `scripts/run_signal_event_rotation_grid.py`.

### `signal_event_rotation_selected`

- `profile`: fixed selected variant from the 2026-06-18 500-candidate sweep.
- `parameters`: `score_mode=op12`, `event_mode=accel`, `flow_gate=retail_contra`, `construction_mode=k2`, `risk_mode=ls03`.
- `result`: 2020-01-01..2026-05-11 weekly next-open sweep showed 79.70% CAGR, -28.93% MDD, 2.00 Sharpe, 64.94% benchmark monthly win rate, 24.38% average turnover, and 6.3 average names.
- `use`: default registered strategy when a single production candidate is needed instead of the broad research surface.

## Dashboard Defaults

The dashboard launches all active strategies with one shared global config unless
a preset overrides schedule or fill mode:

- `start`: `2020-01-01`
- `end`: `2026-05-29`
- `capital`: `100,000,000`
- `fee`: `0.0002` (2bp)
- `sell_tax`: `0.0015` (15bp)
- `slippage`: `0.0005` (5bp)
- default `schedule`: `monthly`
- default `fill_mode`: `next_open`
- `earnings_revision`: `daily`, `close`
- `revision_signal`: use `uv run python -m backtesting.run --preset kospi200_revision_signal`
- `benchmark_overlay`: `monthly`, `close`
- `benchmark_tilt`: `monthly`, `close`
- `op_rrg_strat`: `weekly`, `next_open`
- `rrg_sector_rotation`: `weekly`, `next_open`
- `rrg_sector_rotation_prune90`: `weekly`, `next_open`
- `rrg_sector_rotation_op_rrg_k2`: `weekly`, `next_open`
- `rrg_sector_rotation_op_rrg_k1`: `weekly`, `next_open`
- `rrg_sector_rotation_op_rrg_ex10_k2`: `weekly`, `next_open`
- `rrg_sector_rotation_op_rrg_ex10_k1`: `weekly`, `next_open`
- `signal_event_rotation`: `weekly`, `next_open`
- `signal_event_rotation_selected`: `weekly`, `next_open`

## Screening Notes

Recent smoke screen:

- window: `2026-01-02..2026-04-15`
- schedule: monthly
- fill mode: close
- writer: disabled

Kept:

| current id | total return | nonzero-weight days | avg turnover |
| --- | ---: | ---: | ---: |
| `trend_rank` | 15.5155% | 35 | 7.1096% |
| `benchmark_tilt` | 9.1619% | 55 | 2.2848% |
| `benchmark_overlay` | 9.0183% | 55 | 2.3328% |

Retested and restored:

| current id | window | schedule | total return | CAGR | MDD | Sharpe |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `earnings_revision` | `2015-01-02..2026-03-25` | daily, close fill | 3484.1396% | 38.7330% | -32.9083% | 1.8985 |

New signal-based implementation:

| current id | window | schedule | total return | CAGR | MDD | Sharpe |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `revision_signal` | `2015-01-02..2026-03-25` | signal dates, close fill | 681.9187% | 20.6973% | -18.2929% | 1.4999 |

Removed after screening:

| old id | screening result |
| --- | --- |
| `beta_boost_ls` | 0.0000% total return |
| `breadth_long` | 0.0000% total return and 0.0000% avg turnover |
| `q1q5_ls` | 0.0000% total return |
| `regime_ls` | 0.0000% total return and only 2 nonzero-weight days |
| `sector_tilt` | 0.0000% total return and 0.0000% avg turnover |
| `soft_long` | 0.0000% total return |
| `squeeze_ls` | 0 nonzero-weight days |

Previously removed before this cleanup:

| old id | prior removal note |
| --- | --- |
| `consensus_beta_persistence_concentrated_longonly` | 0 nonzero-weight days, 0.0 total return, and 0.0 turnover over the prior 2023-01-02..2026-04-15 evaluation. |
| `revision_asymmetric_relay_hedge_ls` | 0.0 total return over the prior 2026-01-02..2026-04-15 smoke evaluation. |
| prior earnings-revision prototype | restored as `earnings_revision` after the 2015-start daily retest showed strong long-horizon performance; the prior short monthly smoke window was not representative. |
| `revision_oi_beta_momo_gate_ls` | 0 nonzero-weight days, 0.0 total return, and 0.0 gross exposure over the prior 2026-01-02..2026-04-15 smoke evaluation. |
| `revision_oi_high_beta_momentum_ls` | -0.067% total return with only 1 nonzero-weight day over the prior 2026-01-02..2026-04-15 smoke evaluation. |
| `revision_oi_soft_beta_tilt_momentum_ls` | -0.067% total return with only 1 nonzero-weight day over the prior 2026-01-02..2026-04-15 smoke evaluation. |
| `revision_oi_state_conditioned_beta_gate_ls` | only 3.07% total return over the prior 2026-01-02..2026-04-15 smoke evaluation. |
| `revision_oi_state_conditioned_short_squeeze_beta_cap_ls` | removed as a low-return duplicate of the short-squeeze exclusion idea. |
| `revision_oi_state_conditioned_short_squeeze_beta_exclusion_ls` | removed as a low-return duplicate of the short-squeeze exclusion idea. |
