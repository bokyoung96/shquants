# RRG Sector Rotation Concentration Filter Design

## Context

`rrg_sector_rotation` currently buys or shorts every stock that passes the RRG
state gate and OP revision confirmation. This keeps the signal faithful to the
sector-rotation idea, but it can create too many holdings for a personal
account.

The desired behavior is not a fixed `top_n` portfolio. Selection should remain
signal-strength based: keep stocks whose OP revision strength is meaningfully
inside the preferred quantile, then use a maximum-name cap only as a practical
trading safety limit.

## Decision

Add optional concentration controls to `RrgSectorRotation`:

- `long_quantile`: optional lower bound for long OP revision strength inside
  the current long candidate set.
- `short_quantile`: optional lower bound for short revision severity inside
  the current short candidate set, using positive `short_alpha` values.
- `min_long_revision`: absolute minimum positive OP revision for long
  candidates.
- `min_short_revision`: absolute minimum negative OP revision magnitude for
  short candidates.
- `max_long_names`: optional safety cap after threshold and quantile filtering.
- `max_short_names`: optional safety cap after threshold and quantile filtering.

Default values preserve current behavior:

```text
long_quantile = None
short_quantile = None
min_long_revision = 0.0
min_short_revision = 0.0
max_long_names = None
max_short_names = None
```

The personal-account experiment should use:

```text
long_quantile = 0.70
short_quantile = 0.70
min_long_revision = 0.03
min_short_revision = 0.03
max_long_names = 20
max_short_names = 5
```

`short_quantile = 0.70` means keep the upper 30% of `short_alpha`, where
`short_alpha = -stock_op_revision` for negative OP revision names.

## Data Flow

1. Existing RRG and OP confirmation logic creates `alpha`, `short_alpha`, entry
   masks, hold masks, and tradability masks.
2. Portfolio construction builds long and short candidate scores per date.
3. Candidate scores pass through the concentration filter:
   - remove scores below the absolute minimum revision threshold;
   - remove scores below the configured quantile cutoff;
   - apply the optional max-name cap by descending score.
4. Existing rank-proportional weighting allocates `gross_long` and
   `gross_short` across the remaining names.

## Validation

Regression tests should prove:

- default parameters preserve the current broad-selection behavior;
- quantile and minimum-revision thresholds remove weak candidates;
- max-name caps are applied only after thresholds;
- invalid quantiles outside `[0, 1]` and non-positive caps are rejected;
- the strategy registry accepts the new strategy parameters.

Backtest comparison should report current baseline versus at least:

- `Q70/min3/max20x5`;
- `Q80/min3/max15x5`;
- `Q80/min5/max10x3`.

For each run, compare total return, CAGR, MDD, Sharpe, average turnover,
active days, average holdings, latest active holdings, and output directory.

## Risks

Quantile filtering can reduce diversification in sparse candidate regimes. The
maximum-name cap should be treated as an operational guardrail, not the primary
selection rule. If performance becomes too dependent on one or two short names,
the short sleeve may need a stricter minimum count or a lower `gross_short`.
