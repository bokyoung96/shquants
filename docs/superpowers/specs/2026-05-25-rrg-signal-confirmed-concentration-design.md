# RRG Signal-Confirmed Concentration Design

## Objective

Define a stock-level RRG sector rotation scheme that avoids arbitrary name-count ranking as the primary alpha rule. The strategy should remain concentrated, hold at most 10 names, and select only when sector and stock evidence agree.

## Strategy Shape

The strategy is a stock strategy, not an ETF strategy. RRG state is used as a regime gate. Consensus revision is the primary confirmation signal. Flow is a fallback confirmation signal when consensus revision is unavailable. Selected names are equal-weighted.

The portfolio does not force full investment and does not force-fill 10 names. If fewer than 10 names pass the signal gates, the portfolio holds fewer names and leaves the rest in cash.

## Signal Hierarchy

### Sector Confirmation

For each sector, build cap-weighted forward revision signals:

- `sector_eps_revision`: cap-weighted sector EPS forward revision.
- `sector_op_revision`: cap-weighted sector OP forward revision.
- `sector_consensus_score`: 50% EPS revision and 50% OP revision.

Use float market cap as the preferred weight basis. Fall back to total market cap when float market cap is unavailable.

If sector consensus score is unavailable, use sector flow as fallback confirmation. The fallback is hierarchical, not blended with arbitrary weights:

1. Use sector consensus score when available.
2. Else use sector flow score when available.
3. Else mark sector confirmation missing.

### Stock Confirmation

For each stock, build:

- `stock_eps_revision`
- `stock_op_revision`
- `stock_consensus_score`: 50% EPS revision and 50% OP revision.

If stock consensus score is unavailable, use stock flow as fallback confirmation using the same hierarchy:

1. Use stock consensus score when available.
2. Else use stock flow score when available.
3. Else mark stock confirmation missing.

## Entry Rules

A stock can be newly entered only when all conditions are true:

- Its sector RRG state is `Leading` or `Improving`.
- Sector confirmation score is present and positive.
- Stock confirmation score is present and positive.

RRG alone is not sufficient for entry. A `Leading` sector with missing consensus can still qualify through positive flow confirmation.

## Exit Rules

Exit a stock when any of the following is true:

- Sector RRG state becomes `Lagging`.
- Sector confirmation score is present and non-positive.
- Stock confirmation score is present and non-positive.
- Both consensus and flow confirmation become missing for the sector or stock.

`Weakening` is not a new-entry state. Existing holdings may remain only while sector and stock confirmation remain positive.

## Selection And Concentration

The portfolio holds at most 10 stocks. The 10-name limit is a portfolio concentration constraint, not the alpha rule.

When more than 10 stocks pass the gates, rank candidates by agreement between sector and stock confirmation. Prefer a rank-agreement score such as:

```text
candidate_score = min(sector_confirm_rank, stock_confirm_rank)
```

This prevents a stock from ranking highly when only the sector or only the stock signal is strong.

When 10 or fewer stocks pass the gates, hold all passing names.

## Weighting

Use equal weighting across selected names for the first version. This keeps selection alpha separate from weighting alpha and makes evaluation easier to interpret.

## Non-Goals

- Do not add manual RRG state multipliers such as `Leading = 1.0` and `Improving = 0.7`.
- Do not use a fixed top-N list as the primary alpha rule.
- Do not blend consensus and flow with arbitrary fixed weights.
- Do not force-fill the portfolio when the signal set is sparse.

## Validation

The first implementation should compare:

- Existing archived RRG scheme.
- `use_name_cap=true` restored RRG scheme.
- This signal-confirmed concentrated scheme.

Key checks:

- Holdings count never exceeds 10.
- Strategy can hold cash when fewer than 10 names qualify.
- Entry names have positive sector and stock confirmation.
- Flow fallback is used only when consensus is unavailable.
- Equal-weighted selected holdings sum to at most 1.0.
