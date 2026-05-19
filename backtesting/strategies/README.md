# Strategy Registry

This folder keeps only strategies that are currently worth running from the CLI,
execution specs, or dashboard.

## Naming Scheme

Strategy names should be short, descriptive, and close to the actual investment
logic:

- `trend_rank`: price trend ranking.
- `earnings_revision`: forward EPS/OP consensus revision.
- `revision_signal`: signal-triggered earnings revision with market trend risk-off.
- `benchmark_overlay`: benchmark-weighted portfolio with a soft active overlay.
- `benchmark_tilt`: benchmark-weighted portfolio tilted by revision, flow, and trend.

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

### RRG Sector Rotation

- `id`: `rrg_sector_rotation`
- `file`: `rrg_sector_rotation.py`
- `class`: `RrgSectorRotation`
- `profile`: long-short
- `data`: `close`, `benchmark`, `k200_yn`, `sector_big`, `market_cap`, `volume`, `eps_fwd_q1`, `eps_fwd_q2`, `eps_fwd`, `op_fwd_q1`, `op_fwd_q2`, `op_fwd`, `foreign_flow`, `inst_flow`, `retail_flow`
- `signal`: RRG sector leadership/lagging context plus sector-internal forward revision and investor flow imbalance ranks.
- `construction`: dollar-neutral sector rotation long-short with sector-directed long and short books; not sector-neutral.
- `use`: benchmark-aware sector rotation when you want relative-strength regime context with balanced gross long and short exposure.

## Dashboard Defaults

The dashboard launches all active strategies with one shared global config unless
a preset overrides schedule or fill mode:

- `start`: `2020-01-01`
- `end`: `2026-05-11`
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
