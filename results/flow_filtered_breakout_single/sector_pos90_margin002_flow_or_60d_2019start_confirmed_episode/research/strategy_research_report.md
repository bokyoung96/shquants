# Confirmed Episode Strategy Research Report

## Scope

This report compares a 5-minute 52-week-new-high-only baseline, factor-filter variants, the selected canonical full strategy, and the compressed multi-timeframe variant.
All variants use the same confirmed 5-minute breakout entry, daily continuation exit, ATR stop rule, transaction cost model, episode compression, and fixed 20-slot notional portfolio accounting.
The original filter variants differ by positivity and foreign/institution flow filters. Follow-up experiments showed that positivity does not change the final selected trade set, so the selected canonical full schema now uses flow confirmation without positivity.

## Selected Strategy Schema

### Canonical Full Strategy

- Name: `Flow-Confirmed 52-Week High Breakout Strategy`
- Universe: KOSPI200 historical members.
- Entry setup: 52-week high breakout after 09:20 KST, with next 5-minute close confirmation and next 5-minute open entry.
- Filters: foreign or institution 60-day flow-to-cap confirmation.
- Excluded from canonical schema: positivity hard filter, positivity rank priority, and positivity exit/regime overlays.
- Exit: entry ATR touch stop at stop price, or daily close losing the prior 52-week close high.
- Portfolio accounting: fixed 20-slot notional, 5% per selected position, 35bp round-trip costs.

### Compressed Strategy Variant

- Name: `Sector-Relative Volatility-Compressed Breakout Strategy`
- Starts from the canonical full strategy mechanics and keeps the same entry/exit execution rules.
- Adds `weekly_sector_rs_ok`: prior completed week 12-week stock return must exceed same-sector 12-week average return.
- Adds `daily_vol_compression_ok`: prior-day 20-day realized volatility must be less than or equal to prior-day 60-day realized volatility.
- Purpose: reduce trade count and operating complexity, not maximize total fixed-notional return.

Compressed variant evidence:

- Selected trades: 1,047
- Compression vs canonical universe: 60.43%
- Fixed-notional final return: 28.40%
- Fixed-notional MDD: -2.29%
- Average selected trade return: 54.3 bps
- Profit factor: 1.763
- Max active positions: 11

## Performance Summary

| strategy | prefilter_candidates | input_trades | trades | skipped_trades | final_return | mdd | hit_rate | profit_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5m_new_high_only | 16847 | 2755 | 2738 | 17 | 72.1472 | -3.1780 | 22.8269 | 1.6864 |
| positivity_only | 7807 | 2755 | 2738 | 17 | 72.1472 | -3.1780 | 22.8269 | 1.6864 |
| flow_only | 15876 | 2646 | 2630 | 16 | 67.4735 | -3.0313 | 22.9278 | 1.6791 |
| current | 7507 | 2646 | 2630 | 16 | 67.4735 | -3.0313 | 22.9278 | 1.6791 |

## Factor Impact

| strategy | prefilter_candidates | trades | return_delta_vs_baseline | candidate_reduction_vs_baseline | trade_reduction_vs_baseline | return_per_trade_bps |
| --- | --- | --- | --- | --- | --- | --- |
| 5m_new_high_only | 16847 | 2738 | 0.0000 | 0.0000 | 0.0000 | 2.6350 |
| positivity_only | 7807 | 2738 | 0.0000 | 53.6594 | 0.0000 | 2.6350 |
| flow_only | 15876 | 2630 | -4.6737 | 5.7636 | 3.9445 | 2.5655 |
| current | 7507 | 2630 | -4.6737 | 55.4401 | 3.9445 | 2.5655 |

Interpretation:

- Positivity materially reduces daily prefilter candidates, but in this data it does not reduce final confirmed entries after the 5-minute confirmation and episode compression layers.
- Removing positivity from the flow-confirmed strategy leaves the same selected trade set and the same fixed-notional result in the current evidence set.
- Foreign/institution flow removes 108 selected trades versus the 5m-only baseline and reduces MDD slightly, but it also lowers final fixed-notional return by about 4.67 percentage points.
- The canonical full strategy is therefore the flow-confirmed 52-week high breakout, not the legacy positivity+flow combination.
- The compressed strategy is an optional operating-simplicity version: it reduces selected trades materially and improves trade quality/profit factor, while sacrificing total fixed-notional return.

## Next Improvement Candidates

1. Convert daily volatility compression from a hard filter into a selection priority when slots are scarce. The hard filter improves trade quality but cuts too much notional deployment.
2. Test ATR stop multiplier and re-entry rules, while keeping ATR touch execution. ATR removal and close-confirmed ATR materially worsened drawdown in follow-up experiments.
3. Add execution realism checks for ATR gaps and stop slippage, because the current stop-price fill is optimistic when the market gaps through the stop.
4. Evaluate a separate positivity sleeve instead of forcing positivity into this breakout schema. Positivity did not alter selected trades as a hard filter or rank priority here, but may still be useful in a standalone cross-sectional momentum/rebalance design.
5. Run a true forward/paper-trading holdout after schema selection. Further in-sample feature additions should be treated as exploratory until validated out of sample.

## Walk-Forward Style Yearly Stability

These are chronological yearly fixed-notional equity increments, not a parameter re-optimization walk-forward. They test whether the fixed rule survives across calendar regimes without changing parameters.

| strategy | year | year_return_pct | year_end_equity |
| --- | --- | --- | --- |
| 5m_new_high_only | 2019 | -0.0834 | 0.9992 |
| 5m_new_high_only | 2020 | 13.0822 | 1.1300 |
| 5m_new_high_only | 2021 | 12.8379 | 1.2584 |
| 5m_new_high_only | 2022 | -1.3796 | 1.2446 |
| 5m_new_high_only | 2023 | 5.1311 | 1.2959 |
| 5m_new_high_only | 2024 | 10.5056 | 1.4009 |
| 5m_new_high_only | 2025 | 13.3259 | 1.5342 |
| 5m_new_high_only | 2026 | 18.7276 | 1.7215 |
| positivity_only | 2019 | -0.0834 | 0.9992 |
| positivity_only | 2020 | 13.0822 | 1.1300 |
| positivity_only | 2021 | 12.8379 | 1.2584 |
| positivity_only | 2022 | -1.3796 | 1.2446 |
| positivity_only | 2023 | 5.1311 | 1.2959 |
| positivity_only | 2024 | 10.5056 | 1.4009 |
| positivity_only | 2025 | 13.3259 | 1.5342 |
| positivity_only | 2026 | 18.7276 | 1.7215 |
| flow_only | 2019 | -0.0834 | 0.9992 |
| flow_only | 2020 | 11.0069 | 1.1092 |
| flow_only | 2021 | 10.8222 | 1.2175 |
| flow_only | 2022 | -0.9400 | 1.2081 |
| flow_only | 2023 | 4.6295 | 1.2544 |
| flow_only | 2024 | 10.2906 | 1.3573 |
| flow_only | 2025 | 12.2682 | 1.4799 |
| flow_only | 2026 | 19.4794 | 1.6747 |
| current | 2019 | -0.0834 | 0.9992 |
| current | 2020 | 11.0069 | 1.1092 |
| current | 2021 | 10.8222 | 1.2175 |
| current | 2022 | -0.9400 | 1.2081 |
| current | 2023 | 4.6295 | 1.2544 |
| current | 2024 | 10.2906 | 1.3573 |
| current | 2025 | 12.2682 | 1.4799 |
| current | 2026 | 19.4794 | 1.6747 |

## Bias And Integrity Audit

- Canonical full final fixed-notional return: 67.47%
- Legacy positivity+flow final fixed-notional return: 67.47%
- 5m-only baseline final fixed-notional return: 72.15%
- Canonical full trades: 2,630
- Legacy positivity+flow trades: 2,630
- 5m-only baseline trades: 2,738
- Return accounting mismatches: 0
- Entry price mismatches: 0
- Signal confirmation violations: 0
- Exit condition violations: 0
- KOSPI200 membership violations: 0

## Live Readiness Notes

- Positivity is retained as rejected research evidence, not as a selected canonical filter.
- Market cap and 60-day foreign/institution flow-to-cap are shifted by one trading day before signal use.
- Weekly sector-relative strength in the compressed variant uses only the prior completed weekly bar.
- Daily volatility compression in the compressed variant uses only returns available through the prior completed trading day.
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
- `multi_timeframe_filter_comparison/combo/paper.png`
- `positivity_utilization_comparison/positivity_utilization_report.md`
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
    "fixed_notional_final_return": 0.7214718748845907,
    "fixed_notional_mdd": -0.03178019427007417,
    "rebalanced_final_return": 1.0979444016806097,
    "rebalanced_mdd": -0.03714076089550988,
    "selected_avg_trade_return": 0.00527006482749883,
    "selected_hit_rate": 0.22826880934989044,
    "selected_profit_factor": 1.6864382562148001,
    "skipped_avg_trade_return": 0.024373658661420817,
    "skipped_hit_rate": 0.47058823529411764
  },
  "positivity_only": {
    "max_positions": 20,
    "slot_weight": 0.05,
    "input_trades": 2755,
    "selected_trades": 2738,
    "skipped_trades": 17,
    "max_active_positions": 20,
    "fixed_notional_final_return": 0.7214718748845907,
    "fixed_notional_mdd": -0.03178019427007417,
    "rebalanced_final_return": 1.0979444016806097,
    "rebalanced_mdd": -0.03714076089550988,
    "selected_avg_trade_return": 0.00527006482749883,
    "selected_hit_rate": 0.22826880934989044,
    "selected_profit_factor": 1.6864382562148001,
    "skipped_avg_trade_return": 0.024373658661420817,
    "skipped_hit_rate": 0.47058823529411764
  },
  "flow_only": {
    "max_positions": 20,
    "slot_weight": 0.05,
    "input_trades": 2646,
    "selected_trades": 2630,
    "skipped_trades": 16,
    "max_active_positions": 20,
    "fixed_notional_final_return": 0.6747347008700886,
    "fixed_notional_mdd": -0.03031329202569366,
    "rebalanced_final_return": 0.9969360448313358,
    "rebalanced_mdd": -0.034327811999782964,
    "selected_avg_trade_return": 0.005131062364031095,
    "selected_hit_rate": 0.22927756653992395,
    "selected_profit_factor": 1.6791175494890274,
    "skipped_avg_trade_return": 0.019648838640123734,
    "skipped_hit_rate": 0.375
  },
  "current": {
    "max_positions": 20,
    "slot_weight": 0.05,
    "input_trades": 2646,
    "selected_trades": 2630,
    "skipped_trades": 16,
    "max_active_positions": 20,
    "fixed_notional_final_return": 0.6747347008700886,
    "fixed_notional_mdd": -0.03031329202569366,
    "rebalanced_final_return": 0.9969360448313358,
    "rebalanced_mdd": -0.034327811999782964,
    "selected_avg_trade_return": 0.005131062364031095,
    "selected_hit_rate": 0.22927756653992395,
    "selected_profit_factor": 1.6791175494890274,
    "skipped_avg_trade_return": 0.019648838640123734,
    "skipped_hit_rate": 0.375
  }
}
```
