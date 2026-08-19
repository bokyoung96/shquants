# EMP008 Stock-Consensus Active Screen Design

## Goal

Test whether EMP008 improves when active stock bets require agreement across the
existing four factors. Keep the current 36-month arithmetic-mean factor alpha,
WICS neutrality, factor-plus-idiosyncratic risk model, 0.70% annual tracking
error, monthly rebalancing, transaction costs, and factor-weight candidates
unchanged.

This is a stock-level active-bet screen. It does not add a fifth factor, change
the benchmark universe, or remove benchmark holdings.

## Baseline

The existing optimizer converts factor expected returns into a stock score:

```text
stock_alpha[i] = sum(exposure[i, factor] * expected_alpha[factor])
```

It then maximizes expected active return subject to tracking-error, WICS-sector,
fully-invested, and long-only final-weight constraints. Final weights are:

```text
final_weight[i] = benchmark_weight[i] + active_weight[i]
```

The existing factor-direction policy remains in force before stock screening:

- positive-direction factors with a negative 36-month mean receive zero alpha;
- the low-direction size factor receives zero alpha when its mean is positive;
- WICS sector expected returns remain zero.

## Factor Contribution Vote

After applying the direction policy and configured factor weights, calculate the
four stock-level contributions on every rebalance date:

```text
contribution[i, factor]
    = exposure[i, factor]
    * expected_alpha[factor]
    * factor_weight[factor]
```

Only the four configured alpha factors vote. WICS dummy factors never vote.
Zero contributions abstain. This includes factors whose expected alpha has been
set to zero by the existing direction policy.

For each stock:

- `stock_alpha` is the sum of its non-sector factor contributions;
- `positive_votes` is the number of contributions above zero;
- `negative_votes` is the number of contributions below zero.

Classify the stock as follows:

1. **Positive consensus:** `stock_alpha > 0` and `positive_votes > negative_votes`.
2. **Negative consensus:** `stock_alpha < 0` and `negative_votes > positive_votes`.
3. **Uncertain:** all other cases, including tied votes, zero total alpha, and
   no nonzero votes.

This is a strict-majority rule, not a 3-of-4 hard code. It therefore remains
well-defined when one or more factor alphas are zero.

## Active-Weight Constraints

Translate the classification into optimizer bounds:

| Classification | Active-weight bound | Portfolio meaning |
| --- | --- | --- |
| Positive consensus | `active_weight >= 0` | benchmark weight or overweight |
| Negative consensus | `active_weight <= 0` | benchmark weight or underweight |
| Uncertain | `active_weight = 0` | exactly benchmark weight |

The benchmark portfolio, where every active weight is zero, is always feasible.
The existing long-only final-weight floor remains unchanged. Exact WICS sector
neutrality can cause a sector with only one consensus direction to remain at
benchmark; the implementation must not relax sector neutrality to force a bet.

## Configuration and Compatibility

Add an optional stock active-screen setting with two modes:

- `none`: current behavior and default;
- `factor_consensus`: the rules in this design.

The option must be available through `Emp008Config`, the weight and full-run
CLIs, and the factor-weight grid-search CLI. Existing runs remain behaviorally
unchanged when the option is omitted.

The expected-alpha estimator for the first experiment is `mean`. Do not combine
the screen with `mean_1se`, trimmed mean, EWMA, factor timing, or any additional
signal. A combination with `mean_1se` is a later experiment only if the screen
adds value on its own.

## Diagnostics

Record enough monthly information to distinguish a useful screen from a no-op:

- positive-consensus stock count;
- negative-consensus stock count;
- uncertain stock count and percentage;
- absolute active weight assigned against the permitted consensus direction,
  which must be within numerical tolerance of zero;
- optimizer success and existing TE, sector-residual, active-share, and turnover
  diagnostics.

The comparison report must also show whether the screen materially changed
weights. A negligible active-weight difference is reported as a no-op rather
than an improvement or failure.

## Experiment

Run the same nine WICS factor-weight candidates used by the current grid:

- period: 2019-12-30 through 2026-06-30;
- factor set: size, 12-month momentum, earnings momentum, and value;
- expected-alpha estimator: 36-month arithmetic mean;
- risk model: factor plus idiosyncratic;
- annual tracking error: 0.70%;
- monthly rebalance;
- fee 2 bp, sell tax 15 bp, and slippage 5 bp;
- WICS neutralization only.

The baseline is the existing WICS mean grid. The modified run differs only by
`stock_active_screen=factor_consensus`.

Compare candidates pairwise on:

- information ratio;
- cumulative benchmark-relative excess;
- CAGR;
- maximum drawdown;
- average turnover;
- yearly return delta;
- optimizer success;
- mean absolute target-weight difference;
- consensus class breadth.

## Decision Rule

Do not select a single favorable weight combination. Treat the screen as useful
only when improvement is broad across the nine paired candidates and is not
explained by solver failure, unused tracking-error budget, or a negligible
weight change.

At minimum, report:

- count of candidates with positive IR delta;
- median IR delta;
- median cumulative-relative-excess delta;
- median turnover delta;
- count of calendar years with positive median return delta;
- minimum optimizer success rate;
- median uncertain-stock percentage;
- median absolute target-weight difference.

Keep `none` as the default regardless of the result. Combining the screen with
`mean_1se` requires a separate paired experiment after this standalone test.

## Tests and Acceptance Criteria

1. Contribution signs and vote counts are correct for positive, negative, zero,
   and tied examples.
2. Positive consensus creates a nonnegative active-weight bound.
3. Negative consensus creates a nonpositive active-weight bound.
4. Uncertain stocks receive an exact zero active-weight bound.
5. Sector factors do not vote.
6. Zero-alpha factors abstain.
7. The benchmark active-weight vector remains feasible under every screen.
8. Existing `none` runs remain unchanged.
9. Config and all EMP008 CLI entry points accept and validate the new mode.
10. All nine WICS candidates complete with 100% optimizer success or the
    experiment is rejected as operationally invalid.
11. Run conditions and the screen mode are persisted in output metadata.
12. A paired report and PNG make the screen-versus-baseline result directly
    inspectable.

## Risks

- Hard sign bounds may leave some WICS sectors at benchmark and reduce effective
  tracking-error use.
- Consensus voting ignores contribution magnitude except through the required
  sign of total stock alpha. This is intentional for the first simple test.
- The screen may increase concentration among the remaining eligible names or
  amplify WICS subindustry bets. Report active breadth, top active weights, and
  turnover before interpreting performance.
- This is a rolling historical comparison, not an untouched out-of-sample test.
