# SK Hynix 15:30-15:45 Futures Strategy Research

## Data

- Complete 15:30 entry / 15:45 exit samples: 10 trading days.
- Signal timestamp discipline: signals use information available by 15:30 only.
- Cost assumption: 2.0 bps round-trip per futures trade.
- 2026-06-11 is excluded because the futures sheet stops before 15:45.
- The supplied investor-flow table is daily close-level futures net buy by investor type.
- `t1_` strategies are tradable with this file alone because they use T-1 futures flow plus same-day futures rally by 15:30.
- `same_day_oracle_` strategies test the intended live workflow: if a real-time/intraday futures investor-flow feed is available by 15:30, use same-day flow. The current workbook does not prove that availability.
- ETF price/NAV/creation flow is not used in this version; the focus is futures 수급.

## Window Diagnostic

| window                   | count   | mean     | median   | min       | max      |
| ------------------------ | ------- | -------- | -------- | --------- | -------- |
| target_fut_1530_1545_bps | 10.0000 | -70.3151 | -63.6885 | -430.1595 | 112.5245 |
| target_fut_1530_1535_bps | 10.0000 | -18.0279 | -23.6564 | -147.9915 | 95.4861  |
| target_fut_1535_1545_bps | 10.0000 | -52.3427 | -21.4961 | -393.0131 | 68.1929  |

- The 15:35-15:45 leg is material, so these tests are partly about the post-close futures auction/settlement print rather than continuous 15:30-15:45 liquidity.

## Futures Flow Strategy Ideas Tested

- Rally + investment trust: trade only when futures has rallied by 15:30 and investment-trust futures flow is net buying.
- Strong rally + investment trust: same rule but require day-to-15:30 futures rally above 100 bps.
- Rally + investment trust + foreign/institutional confirmation: require additional buyer participation.
- Rally + investment trust + securities sell: test whether apparent buy-side demand is exhausted/offset by securities selling.
- Heavy futures tape pressure after rally: use 15:25-15:30 futures price/volume pressure as a non-investor-type proxy.
- Each idea is tested as both continuation/long and fade/short where economically relevant.

## Backtest Summary

| strategy                                          | n_trades | hit_rate | avg_net_bps | median_net_bps | total_net_bps | max_drawdown_bps | worst_leave_one_out_total_net_bps | largest_win_share_of_total_net |
| ------------------------------------------------- | -------- | -------- | ----------- | -------------- | ------------- | ---------------- | --------------------------------- | ------------------------------ |
| same_day_oracle_rally_trust_securities_sell_short | 3        | 1.0000   | 87.7827     | 119.0549       | 263.3480      | 0.0000           | 144.0091                          | 0.4532                         |
| same_day_oracle_rally_trust_foreign_short         | 3        | 1.0000   | 87.7827     | 119.0549       | 263.3480      | 0.0000           | 144.0091                          | 0.4532                         |
| same_day_oracle_rally_trust_short                 | 4        | 0.7500   | 52.3162     | 72.0045        | 209.2647      | 0.0000           | 89.9258                           | 0.5703                         |
| t1_rally_prev_trust_short                         | 3        | 0.6667   | 61.4368     | 119.0549       | 184.3105      | 0.0000           | 64.9716                           | 0.6475                         |
| same_day_oracle_strong_rally_trust_short          | 3        | 0.6667   | 30.0699     | 24.9542        | 90.2098       | 0.0000           | -29.1292                          | 1.3229                         |
| t1_strong_rally_prev_trust_short                  | 2        | 0.5000   | 32.6278     | 32.6278        | 65.2556       | 0.0000           | -54.0833                          | 1.8288                         |
| same_day_oracle_rally_trust_institutional_long    | 1        | 1.0000   | 50.0833     | 50.0833        | 50.0833       | 0.0000           | 0.0000                            | 1.0000                         |
| t1_rally_prev_trust_foreign_long                  | 1        | 1.0000   | 50.0833     | 50.0833        | 50.0833       | 0.0000           | 0.0000                            | 1.0000                         |
| heavy_futures_pressure_after_rally_short          | 0        |          |             |                | 0.0000        | 0.0000           | 0.0000                            |                                |
| t1_rally_prev_trust_foreign_short                 | 1        | 0.0000   | -54.0833    | -54.0833       | -54.0833      | 0.0000           | 0.0000                            |                                |
| t1_rally_prev_trust_securities_sell_short         | 1        | 0.0000   | -54.0833    | -54.0833       | -54.0833      | 0.0000           | 0.0000                            |                                |
| t1_strong_rally_prev_trust_long                   | 2        | 0.5000   | -36.6278    | -36.6278       | -73.2556      | -123.3389        | -123.3389                         |                                |
| same_day_oracle_strong_rally_trust_long           | 3        | 0.3333   | -34.0699    | -28.9542       | -102.2098     | -152.2931        | -152.2931                         |                                |
| t1_rally_prev_trust_institutional_long            | 3        | 0.3333   | -65.4368    | -123.0549      | -196.3105     | -246.3938        | -246.3938                         |                                |
| t1_rally_prev_trust_long                          | 3        | 0.3333   | -65.4368    | -123.0549      | -196.3105     | -246.3938        | -246.3938                         |                                |
| same_day_oracle_rally_trust_long                  | 4        | 0.2500   | -56.3162    | -76.0045       | -225.2647     | -275.3480        | -275.3480                         |                                |
| same_day_oracle_rally_trust_foreign_long          | 3        | 0.0000   | -91.7827    | -123.0549      | -275.3480     | -152.0091        | -246.3938                         |                                |

## Feature Correlations To 15:30-15:45 Futures Return

| feature                                         | pearson_corr | spearman_corr | n  |
| ----------------------------------------------- | ------------ | ------------- | -- |
| fut_1525_1530_bps                               | 0.2449       | 0.6748        | 10 |
| prev_securities_net_buy_qty_ex_spread           | -0.2956      | -0.4909       | 10 |
| prev_pension_fund_net_buy_qty_ex_spread         | -0.3266      | -0.4424       | 10 |
| prev_institutional_net_buy_qty_ex_spread        | -0.2081      | -0.4061       | 10 |
| prev_investment_trust_net_buy_qty_ex_spread     | 0.3293       | 0.4061        | 10 |
| prev_foreign_net_buy_qty_ex_spread              | 0.1698       | 0.3455        | 10 |
| same_day_investment_trust_net_buy_qty_ex_spread | 0.3974       | 0.2970        | 10 |
| same_day_foreign_net_buy_qty_ex_spread          | 0.2232       | -0.2485       | 10 |
| fut_signed_volume_pressure_1525_1530            | -0.0535      | 0.2249        | 10 |
| same_day_institutional_net_buy_qty_ex_spread    | -0.3169      | 0.1879        | 10 |

## Read

- Best overall in-sample futures-flow result: `same_day_oracle_rally_trust_securities_sell_short` at 263.35 bps net over 3 trades.
- Best same-day/oracle version: `same_day_oracle_rally_trust_securities_sell_short` at 263.35 bps net over 3 trades.
- Best T-1 tradable version with this workbook alone: `t1_rally_prev_trust_short` at 184.31 bps net over 3 trades.
- Best candidate with at least 3 trades: `same_day_oracle_rally_trust_securities_sell_short` at 263.35 bps net over 3 trades.
- For that best strategy, the worst leave-one-day-out total is 144.01 bps; largest winning day share is 45.32%.
- Treat two-trade results as event diagnostics, not deployable strategy evidence.
- This is too small a sample for deployment. Treat positive results as hypotheses to retest with more post-listing days.
- The most useful next data addition is intraday futures investor-flow by type up to 15:30, especially investment trust, foreign, securities, and institutional cumulative flow.
