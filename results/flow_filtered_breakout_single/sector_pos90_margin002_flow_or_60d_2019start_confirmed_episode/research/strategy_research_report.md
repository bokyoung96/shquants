# Confirmed Episode Strategy Research Report

## Scope

This report compares a 5-minute 52-week-new-high-only baseline against the current strategy.
All variants use the same confirmed 5-minute breakout entry, daily continuation exit, ATR stop rule, transaction cost model, episode compression, and fixed 20-slot notional portfolio accounting.
The variants differ only by positivity and foreign/institution flow filters.

## Performance Summary

| strategy | prefilter_candidates | input_trades | trades | skipped_trades | final_return | mdd | hit_rate | profit_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5m_new_high_only | 16847 | 2755 | 2738 | 17 | 87.2062 | -2.3734 | 23.0825 | 1.9328 |
| positivity_only | 7807 | 2755 | 2738 | 17 | 87.2062 | -2.3734 | 23.0825 | 1.9328 |
| flow_only | 15876 | 2646 | 2630 | 16 | 81.9385 | -2.2115 | 23.1939 | 1.9288 |
| current | 7507 | 2646 | 2630 | 16 | 81.9385 | -2.2115 | 23.1939 | 1.9288 |

## Factor Impact

| strategy | prefilter_candidates | trades | return_delta_vs_baseline | candidate_reduction_vs_baseline | trade_reduction_vs_baseline | return_per_trade_bps |
| --- | --- | --- | --- | --- | --- | --- |
| 5m_new_high_only | 16847 | 2738 | 0.0000 | 0.0000 | 0.0000 | 3.1850 |
| positivity_only | 7807 | 2738 | 0.0000 | 53.6594 | 0.0000 | 3.1850 |
| flow_only | 15876 | 2630 | -5.2677 | 5.7636 | 3.9445 | 3.1155 |
| current | 7507 | 2630 | -5.2677 | 55.4401 | 3.9445 | 3.1155 |

Interpretation:

- Positivity materially reduces daily prefilter candidates, but in this data it does not reduce final confirmed entries after the 5-minute confirmation and episode compression layers.
- Foreign/institution flow removes 108 selected trades versus the 5m-only baseline and reduces MDD slightly, but it also lowers final fixed-notional return by about 5.27 percentage points.
- The current combined strategy is therefore not better than the 5m-only baseline on this historical fixed-notional metric; its main benefit is a small position-count reduction and slightly smaller drawdown.

## Walk-Forward Style Yearly Stability

These are chronological yearly fixed-notional equity increments, not a parameter re-optimization walk-forward. They test whether the fixed rule survives across calendar regimes without changing parameters.

| strategy | year | year_return_pct | year_end_equity |
| --- | --- | --- | --- |
| 5m_new_high_only | 2019 | 0.7911 | 1.0079 |
| 5m_new_high_only | 2020 | 14.8037 | 1.1559 |
| 5m_new_high_only | 2021 | 16.0499 | 1.3164 |
| 5m_new_high_only | 2022 | -0.4996 | 1.3115 |
| 5m_new_high_only | 2023 | 6.7591 | 1.3790 |
| 5m_new_high_only | 2024 | 13.0301 | 1.5093 |
| 5m_new_high_only | 2025 | 16.0319 | 1.6697 |
| 5m_new_high_only | 2026 | 20.2401 | 1.8721 |
| positivity_only | 2019 | 0.7911 | 1.0079 |
| positivity_only | 2020 | 14.8037 | 1.1559 |
| positivity_only | 2021 | 16.0499 | 1.3164 |
| positivity_only | 2022 | -0.4996 | 1.3115 |
| positivity_only | 2023 | 6.7591 | 1.3790 |
| positivity_only | 2024 | 13.0301 | 1.5093 |
| positivity_only | 2025 | 16.0319 | 1.6697 |
| positivity_only | 2026 | 20.2401 | 1.8721 |
| flow_only | 2019 | 0.7911 | 1.0079 |
| flow_only | 2020 | 12.5139 | 1.1330 |
| flow_only | 2021 | 13.8362 | 1.2714 |
| flow_only | 2022 | -0.0930 | 1.2705 |
| flow_only | 2023 | 6.2080 | 1.3326 |
| flow_only | 2024 | 12.7931 | 1.4605 |
| flow_only | 2025 | 14.9577 | 1.6101 |
| flow_only | 2026 | 20.9314 | 1.8194 |
| current | 2019 | 0.7911 | 1.0079 |
| current | 2020 | 12.5139 | 1.1330 |
| current | 2021 | 13.8362 | 1.2714 |
| current | 2022 | -0.0930 | 1.2705 |
| current | 2023 | 6.2080 | 1.3326 |
| current | 2024 | 12.7931 | 1.4605 |
| current | 2025 | 14.9577 | 1.6101 |
| current | 2026 | 20.9314 | 1.8194 |

## Bias And Integrity Audit

- Current final fixed-notional return: 81.94%
- 5m-only baseline final fixed-notional return: 87.21%
- Current trades: 2,630
- 5m-only baseline trades: 2,738
- Return accounting mismatches: 0
- Entry price mismatches: 0
- Signal confirmation violations: 0
- Exit condition violations: 0
- KOSPI200 membership violations: 0

## Live Readiness Notes

- Positivity uses shifted daily returns; the signal date never uses that day's completed return for positivity.
- Market cap and 60-day foreign/institution flow-to-cap are shifted by one trading day before signal use.
- The daily prefilter uses same-day 5-minute max close only to decide which candidate days require intraday loading; final entry is revalidated at 5-minute signal/confirmation bars.
- ATR stop is modelled as stop-price fill once daily low breaches the stop. This is practical but optimistic if the market gaps through the stop or trades below the stop without available liquidity.
- Walk-forward parameter retraining is not applicable because this is a fixed-rule strategy. The stronger live-readiness test is an untouched holdout or paper-trading period after strategy selection.
- Backtest-overfitting remains a material research-process risk because the strategy was selected after multiple comparisons. Bailey, Borwein, Lopez de Prado, and Zhu propose PBO specifically for this type of investment-simulation selection risk.
- Based on this audit, the code path is internally consistent, but I would not approve immediate full-size live deployment. The safer gate is paper trading or tiny notional live shadowing with real stop execution and slippage logging.

## External Bias References

- Bailey, Borwein, Lopez de Prado, and Zhu, The Probability of Backtest Overfitting: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey et al., Statistical Overfitting and Backtest Performance: https://sdm.lbl.gov/oapapers/ssrn-id2507040-bailey.pdf
- Lopez de Prado, Backtesting: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2606462
- Risk.net summary of The Probability of Backtest Overfitting: https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting

## Artifacts

- `baseline_vs_current_comparison.png`
- `factor_impact.png`
- `bias_audit/strategy_integrity_report.md`

## Variant Audit Snapshot

```json
{
  "5m_new_high_only": {
    "max_positions": 20,
    "slot_weight": 0.05,
    "input_trades": 2755,
    "selected_trades": 2738,
    "skipped_trades": 17,
    "max_active_positions": 20,
    "fixed_notional_final_return": 0.8720618748845892,
    "fixed_notional_mdd": -0.02373373492046571,
    "rebalanced_final_return": 1.4340048385724677,
    "rebalanced_mdd": -0.031955328251668025,
    "selected_avg_trade_return": 0.006370064827498831,
    "selected_hit_rate": 0.2308254200146092,
    "selected_profit_factor": 1.9327593464404285,
    "skipped_avg_trade_return": 0.025473658661420814,
    "skipped_hit_rate": 0.47058823529411764
  },
  "positivity_only": {
    "max_positions": 20,
    "slot_weight": 0.05,
    "input_trades": 2755,
    "selected_trades": 2738,
    "skipped_trades": 17,
    "max_active_positions": 20,
    "fixed_notional_final_return": 0.8720618748845892,
    "fixed_notional_mdd": -0.02373373492046571,
    "rebalanced_final_return": 1.4340048385724677,
    "rebalanced_mdd": -0.031955328251668025,
    "selected_avg_trade_return": 0.006370064827498831,
    "selected_hit_rate": 0.2308254200146092,
    "selected_profit_factor": 1.9327593464404285,
    "skipped_avg_trade_return": 0.025473658661420814,
    "skipped_hit_rate": 0.47058823529411764
  },
  "flow_only": {
    "max_positions": 20,
    "slot_weight": 0.05,
    "input_trades": 2646,
    "selected_trades": 2630,
    "skipped_trades": 16,
    "max_active_positions": 20,
    "fixed_notional_final_return": 0.8193847008700885,
    "fixed_notional_mdd": -0.022115431196809476,
    "rebalanced_final_return": 1.3033077044029269,
    "rebalanced_mdd": -0.030677307724524372,
    "selected_avg_trade_return": 0.006231062364031095,
    "selected_hit_rate": 0.23193916349809887,
    "selected_profit_factor": 1.9288323352602812,
    "skipped_avg_trade_return": 0.02074883864012373,
    "skipped_hit_rate": 0.375
  },
  "current": {
    "max_positions": 20,
    "slot_weight": 0.05,
    "input_trades": 2646,
    "selected_trades": 2630,
    "skipped_trades": 16,
    "max_active_positions": 20,
    "fixed_notional_final_return": 0.8193847008700885,
    "fixed_notional_mdd": -0.022115431196809476,
    "rebalanced_final_return": 1.3033077044029269,
    "rebalanced_mdd": -0.030677307724524372,
    "selected_avg_trade_return": 0.006231062364031095,
    "selected_hit_rate": 0.23193916349809887,
    "selected_profit_factor": 1.9288323352602812,
    "skipped_avg_trade_return": 0.02074883864012373,
    "skipped_hit_rate": 0.375
  }
}
```
