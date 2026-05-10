# Strategy Registry Notes

## revision_minparam_v02

Official recorded strategy for Discord reference label **전략1**.

### Alias
- 전략1

### Description
- Universe: KOSPI200
- Selection: EPS fwd 1M revision > 0 and OP fwd 1M revision > 0
- Ranking: average percentile of EPS/OP 1M forward revisions
- PTH: 0.6
- Top N: 30
- Rebalance: weekly
- Execution: next_open
- Fee: 15bps
- Slippage: 10bps

### Implementation notes
- Python strategy id: `revision_minparam_v02`
- Source file: `backtesting/strategies/revision_minparam_v02.py`
- This README entry documents the strategy alias and intended interpretation without duplicating the implementation under a second strategy name.
