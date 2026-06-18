# Signal Event Rotation 500-Candidate Results

## Setup

- Universe: KOSPI200.
- Execution: weekly rebalance, next-open fill, 2bp fee, 15bp sell tax, 5bp slippage.
- Candidate count: 500 fixed combinations. No optimized thresholds.
- Economic rationale: OP consensus events, sector price/OP cycle confirmation, and investor-flow confirmation.

## Selected Candidates

| rank | strategy | CAGR | MDD | Sharpe | BM monthly win | turnover | avg names |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `signal_event_rotation_sev_op12_accel_retail_contra_k2_ls03` | 79.70% | -28.93% | 2.00 | 64.94% | 24.38% | 6.3 |
| 2 | `signal_event_rotation_sev_op12_accel_smart_k3_lo` | 79.34% | -38.23% | 2.01 | 61.04% | 18.35% | 7.0 |
| 3 | `signal_event_rotation_sev_qavg_accel_none_k1_ls02` | 93.68% | -37.27% | 1.85 | 53.25% | 19.32% | 4.3 |
| 4 | `signal_event_rotation_sev_op12_accel_retail_contra_k3_ls05` | 72.77% | -27.61% | 1.89 | 61.04% | 28.07% | 8.1 |
| 5 | `signal_event_rotation_sev_eps_op_accel_none_k3_ls03` | 67.54% | -26.63% | 1.82 | 63.64% | 19.50% | 7.7 |
| 6 | `signal_event_rotation_sev_op12_accel_retail_contra_k1_ls02` | 82.83% | -30.66% | 1.81 | 51.95% | 23.20% | 4.0 |
| 7 | `signal_event_rotation_sev_qavg_accel_none_k3_ls05` | 67.18% | -23.87% | 1.81 | 59.74% | 24.11% | 9.5 |
| 8 | `signal_event_rotation_sev_blend_accel_retail_contra_k1_ls03` | 80.32% | -35.38% | 1.80 | 55.84% | 24.26% | 4.0 |
| 9 | `signal_event_rotation_sev_op12_accel_foreign_k2_lo` | 79.32% | -43.49% | 1.81 | 61.04% | 18.01% | 4.9 |
| 10 | `signal_event_rotation_sev_blend_accel_smart_k2_lo` | 65.99% | -33.05% | 1.77 | 61.04% | 18.60% | 5.2 |

## Artifacts

- CSV: `results/signal_event_research/signal_event_grid_summary_20260618_222313.csv`
- JSON: `results/signal_event_research/signal_event_grid_summary_20260618_222313.json`
- Per-run artifacts: `results/backtests/signal_event_rotation_*`
