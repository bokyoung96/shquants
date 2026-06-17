# RRG Sector Rotation 30 Strategy Research

Generated: 2026-06-17T18:09:12

## Scope

- Universe: KOSPI200
- Sector taxonomy: `QW_WI_SEC_26_BIG`
- Schedule/fill: weekly, next open
- Costs: fee 2bp, sell tax 15bp, slippage 5bp
- Strategy family: RRG sector regime plus OP consensus confirmation, compressed by sector exposure preservation.

## Variant Grid

Five OP definitions were crossed with six sector-preserving compression methods for 30 total strategies.

OP modes:
- `qavg`: Q1/Q2/FY1 average OP revision
- `op12`: 12M forward OP revision only
- `agree2`: At least 2 of Q1/Q2/FY1/12M agree
- `agree3`: At least 3 of Q1/Q2/FY1/12M agree
- `breadth50`: Q-average with sector positive/negative breadth >= 50%

Compression modes:
- `k2_weight`: sector exposure, top 2/1 by baseline weight
- `k3_weight`: sector exposure, top 3/2 by baseline weight
- `k2_op`: sector exposure, top 2/1 by OP revision
- `k3_op`: sector exposure, top 3/2 by OP revision
- `k2_momo21`: sector exposure, top 2/1 by 21D stock-vs-sector momentum
- `k3_mcap`: sector exposure, top 3/2 by float market cap

## Top 10 By Sharpe

| rank | strategy | CAGR | MDD | Sharpe | avg names | turnover |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `rrg_agree3_k2_weight` | 84.90% | -15.96% | 2.50 | 22.7 | 22.31% |
| 2 | `rrg_agree3_k2_op` | 84.90% | -15.96% | 2.50 | 22.7 | 22.31% |
| 3 | `rrg_agree2_k2_weight` | 75.07% | -15.63% | 2.39 | 24.7 | 22.72% |
| 4 | `rrg_agree2_k2_op` | 75.07% | -15.63% | 2.39 | 24.7 | 22.72% |
| 5 | `rrg_agree3_k3_weight` | 72.48% | -20.54% | 2.39 | 31.5 | 21.84% |
| 6 | `rrg_agree3_k3_op` | 72.48% | -20.54% | 2.39 | 31.5 | 21.84% |
| 7 | `rrg_qavg_k2_op` | 70.75% | -14.19% | 2.32 | 23.5 | 22.71% |
| 8 | `rrg_qavg_k2_weight` | 70.65% | -14.19% | 2.31 | 23.5 | 22.71% |
| 9 | `rrg_breadth50_k2_op` | 73.88% | -19.92% | 2.29 | 19.3 | 22.59% |
| 10 | `rrg_breadth50_k2_weight` | 73.78% | -19.92% | 2.29 | 19.3 | 22.59% |

## Readout

- Baseline reference: `rrg-sector-rotation_20260617_155800` had CAGR 48.25%, MDD -14.33%, Sharpe 2.03, and about 63.6 names.
- The cleanest name-count reduction is `rrg_qavg_k2_op` / `rrg_qavg_k2_weight`: about 23.5 names, MDD -14.19%, and Sharpe above 2.31.
- `agree3` is the strongest return/Sharpe family, but its MDD is slightly worse than the baseline MDD target.
- `op12` alone is not a good replacement in this test. It keeps more names and produces much worse drawdown.
- The 21D stock-vs-sector momentum compression is consistently poor: it raises turnover and worsens drawdown across OP modes.
- The sensitivity around Q1/Q2 comes from sign-gated OP filters. Small forecast revisions can flip a stock or sector from eligible to ineligible, so agreement/breadth checks are better used as stabilizers than as tightly optimized thresholds.

## Baseline-MDD-Compatible Candidates

| strategy | CAGR | MDD | Sharpe | avg names | turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rrg_qavg_k2_op` | 70.75% | -14.19% | 2.32 | 23.5 | 22.71% |
| `rrg_qavg_k2_weight` | 70.65% | -14.19% | 2.31 | 23.5 | 22.71% |

## Artifacts

- CSV summary: `results\rrg_research\rrg_30_strategy_summary_20260617_180731.csv`
- JSON summary: `results\rrg_research\rrg_30_strategy_summary_20260617_180731.json`
- Per-run artifacts are under `results/backtests/` using each strategy slug.

## Notes

These are research variants, not registered production strategies. The grid intentionally uses broad, explainable axes rather than optimizing thresholds against the backtest window.
