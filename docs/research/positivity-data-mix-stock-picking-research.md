# Positivity Data-Mix Stock Picking Research

Date: 2026-06-24

## Question

Positivity alone already looks useful as a KOSPI200 ranking signal. The next question is which extra data should be mixed with positivity to build practical long or short stock-picking strategies.

## Evidence Used

- Re-ran the current structural grid:
  `python scripts/run_pos_strategy_grid.py --output results/pos_research/strategy_mix_research_20260624`
- Grid window: 2020-01-01 through 2026-06-04.
- Grid count: 50 strategy structures.
- Existing momentum comparison evidence:
  `results/pos_research/momentum_reports/summary.csv`
- Existing positivity research evidence:
  `results/pos_research/summary.csv`,
  `results/pos_research/sponsorship/summary.csv`,
  `results/pos_research/sector_event_core/summary.csv`

## Main Finding

The best immediate mix is not another fundamental factor. It is:

1. 60-day positivity rank
2. 252-day price breakout or near-high confirmation
3. sector breadth cap
4. optional flow/revision overlay depending on whether the goal is raw return or drawdown control

This points to positivity as a persistence filter, not as a full entry rule by itself.

## Long Candidate Ranking

Top current-code structural grid candidates:

| Candidate | Data mix | CAGR | MDD | Sharpe | Late CAGR | Late Sharpe | Robust score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `new_high_px_n5_sec1` | positivity + 252d price breakout + sector cap 1 | 42.45% | -41.05% | 1.125 | 91.53% | 1.796 | 1.034 |
| `new_high_flow_n5_sec1` | positivity + breakout + foreign/institution sponsorship | 30.31% | -42.79% | 0.945 | 88.90% | 1.887 | 0.708 |
| `stable_oprev_n10_sec1` | positivity stability + OP revision | 17.35% | -26.10% | 0.803 | 37.25% | 1.308 | 0.665 |
| `new_high_retail_n15_sec2` | positivity + breakout + retail contrarian | 29.34% | -44.71% | 1.004 | 66.93% | 1.866 | 0.656 |
| `pullback_epsrev_n10_sec2` | positivity + pullback reclaim + EPS revision | 16.23% | -26.07% | 0.778 | 39.07% | 1.392 | 0.622 |

Interpretation:

- `new_high_px_n5_sec1` is the best raw return candidate, but drawdown is large.
- `new_high_flow_n5_sec1` is the best non-price breakout candidate, but its early half is weak. Treat it as a confirmation overlay, not a standalone truth.
- `stable_oprev_n10_sec1` and `pullback_epsrev_n10_sec2` have much lower drawdown and are better first choices for a managed long sleeve.

## Which Data Helps

### 1. Price confirmation: strongest

The strongest entry structure is positivity plus price confirmation:

- 60-day positivity top group
- price breaks above prior 252-day high
- hold max 5 names
- max 1 name per sector
- stop on 20-day low

This works because positivity says the path has been persistently supported, while the breakout says the market is paying attention now.

Use this as the high-octane long strategy.

### 2. OP/EPS revisions: best risk-control overlay

Revision overlays did not beat price breakout on raw return, but they materially improved drawdown:

- OP revision stable sleeve: MDD near -26.10%
- EPS pullback reclaim: MDD near -26.07%

These are better for a portfolio sleeve that should stay invested without being entirely event-driven.

Use OP revision for stable holdings. Use EPS revision for pullback/reclaim entries.

### 3. Investor flow: useful but unstable

Flow overlays are promising late in the sample:

- sponsorship best candidate late Sharpe: 1.887
- retail contrarian best candidate late Sharpe: 1.866

But some flow versions had weak early-half performance. Flow should be a tie-breaker or confirmation filter, not the primary entry gate.

Recommended flow use:

- long: foreign + institution 20-day sponsorship in the upper half of sector ranks
- long: retail selling/absorption as a contrarian confirmation
- avoid: dual sponsorship as a blind long signal; existing sponsorship group evidence shows dual sponsorship was weak

### 4. Sector state: necessary constraint, not enough as alpha

The `sector_event_core_v2` output was weak:

- CAGR 6.95%
- MDD -43.25%
- Sharpe 0.421

Sector state is still useful for concentration control, sector-relative ranking, and avoiding isolated names in weak sectors. It is not strong enough as the main signal in the current version.

## Short Strategy Read

Do not short low-positivity names by positivity alone.

Evidence: low-ranked Q1 portfolios still had positive absolute CAGR in the 6M/12M windows, for example:

| Signal | Horizon | Q1 CAGR | Q1 MDD | Q1 Sharpe |
| --- | --- | ---: | ---: | ---: |
| positivity | 12M | 4.42% | -46.17% | 0.297 |
| positivity | 12-1M | 3.53% | -44.23% | 0.264 |
| positivity | 6M | 8.18% | -45.81% | 0.433 |

That means the short leg works better as a relative hedge than as a standalone directional short. The evidence for long-short is in the spread:

| Spread | Horizon | CAGR | MDD | Sharpe |
| --- | --- | ---: | ---: | ---: |
| positivity Q5-Q1 | 12M | 16.58% | -22.43% | 0.984 |
| positivity Q5-Q1 | 12-1M | 17.48% | -20.36% | 1.049 |
| positivity Q5-Q1 | 6-1M | 13.55% | -21.38% | 0.882 |

Recommended short construction:

- sector-neutral pair trade, not broad market short
- short candidates must satisfy multiple negatives:
  - bottom positivity bucket
  - below sector-relative near-high or failed reclaim
  - negative 20-day OP/EPS revision
  - no foreign/institution sponsorship
  - preferably retail buying pressure without institutional sponsorship
- cap gross short by borrow/shortability, because Korea short constraints matter

## Recommended Strategy Set

### A. Primary long: Positivity breakout sleeve

Use when the goal is aggressive alpha.

- Long top 60-day positivity names.
- Require 252-day high breakout.
- Rank entries by positivity rank + breakout strength.
- Max 5 names, max 1 per sector.
- Exit on 20-day low.

This is closest to `new_high_px_n5_sec1`.

### B. Managed long: Positivity plus revisions

Use when drawdown matters more.

- Stable sleeve: 60/120/120-day positivity ranks.
- Require upper-half OP revision inside sector.
- Max 10 names, max 1 per sector.
- Monthly rebalance.

This is closest to `stable_oprev_n10_sec1`.

### C. Long-short research path

Use as the next implementation/research target.

- Long leg: top positivity + near-high/breakout + positive revision.
- Short leg: bottom positivity + weak near-high + negative revision + no sponsorship.
- Build sector-neutral pairs first.
- Report long leg, short leg, spread, borrow-cost sensitivity, and shortable mask sensitivity separately.

## Next Implementation Shape

The next useful code change is a dedicated long-short grid, not another long-only grid:

1. Add overlay ranks for near-high, OP revision, EPS revision, sponsorship, retail pressure.
2. Create a sector-neutral long-short selector that chooses top long candidates and bottom short candidates within sector.
3. Evaluate:
   - long-only return
   - short-only return
   - Q5-Q1 spread
   - sector exposure
   - active day ratio
   - borrow-cost sensitivity
4. Save selected candidates and latest holdings to a report directory.

This should decide whether the short leg is genuinely tradable or only useful as a diagnostic hedge.
