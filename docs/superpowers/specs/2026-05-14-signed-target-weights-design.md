# Signed Target Weights Input Lane

## Purpose

Backtesting should execute portfolio weights produced by a strategy without reinterpreting the strategy. The current spec path can construct long-only, long-short, and sector-neutral weights through selected built-in constructors, but the `weighting.explicit` path is not a general final-weight input. It rejects negative values and normalizes positive values inside a selected universe.

This design adds a first-class signed target weights lane so an external or strategy-owned portfolio construction step can provide the final `date x symbol` weight matrix directly to the backtest runner.

## Current Evidence

- `backtesting.engine.core.BacktestEngine` already executes signed weights by converting `target_weight * nav / price` into target quantities, including negative quantities for shorts.
- `tests/engine/test_core.py` already covers a negative weight and borrow fee case.
- `backtesting.weighting.builders._validate_explicit_values` rejects negative explicit weights.
- `backtesting.weighting.builders._normalize` sanitizes values to positive-only weights before row normalization.
- `backtesting.specs.portfolio_shapes.build_portfolio_shape_construction` supports long-short and sector-neutral construction only through `selection.kind == "rank_top_bottom"`.

## Principles

1. Backtest executes final portfolio weights; it does not infer or rewrite strategy intent.
2. Signed weights must be preserved exactly except for explicit, user-selected execution policies.
3. Validation should check execution feasibility, not strategy correctness.
4. Silent performance-changing repairs are disallowed by default.
5. All strategy, composed-spec, and external-weight paths should converge on `PositionPlan.target_weights`.

## Non-Goals

- Do not implement Q1/Q5 selection logic in this change.
- Do not redesign the whole selection/weighting/portfolio-shape model in this change.
- Do not change existing `weighting.kind = "explicit"` behavior in place.
- Do not automatically normalize, cap, or rebalance externally supplied weights.

## Proposed Contract

Add a new spec object:

```python
@dataclass(frozen=True, slots=True)
class TargetWeightsSpec:
    kind: str  # "file" initially, "hook" later
    path: str | None = None
    hook_id: str | None = None
    missing_policy: str = "zero"
    untradable_policy: str = "fail"
    unshortable_policy: str = "fail"
```

Add it to `ExecutionSpec`:

```python
target_weights: TargetWeightsSpec | None = None
```

When `target_weights` is present, it becomes the plan source:

```text
target_weights.file
-> signed matrix reader
-> execution validation
-> PositionPlan.target_weights
-> BacktestEngine.run
```

It bypasses `selection`, `weighting`, and `portfolio_shape` because those are portfolio-construction concepts. A target weights input is already the completed portfolio.

## Initial File Format

Use a wide CSV matrix:

```csv
,A005930,A000660,A035420
2024-01-02,0.50,-0.50,0
2024-01-03,0.45,-0.45,0
```

Rules:

- First column is an ISO date index, `YYYY-MM-DD`.
- Columns are symbols.
- Values are finite numeric weights.
- Positive values are long exposure.
- Negative values are short exposure.
- Blank cells are treated as zero under `missing_policy = "zero"`.
- Duplicate dates or duplicate symbols are invalid.
- No row-level normalization is applied.

## Default Validation Policy

Default behavior is fail-fast where silent changes would alter performance:

- Reject non-finite numeric values.
- Reject negative weights unless `shorting.enabled = true`.
- Reject target weights for untradable names when a tradable mask is available and `untradable_policy = "fail"`.
- Reject short targets for unshortable names when `shorting.shortable_field` is configured and `unshortable_policy = "fail"`.
- Align target weights to market dates and symbols.
- Missing cells for known market dates and symbols become zero only under `missing_policy = "zero"`.
- Nonzero weights for symbols that are absent from the loaded market data are invalid by default.
- Dates outside the requested run window are ignored only after the file itself passes date validation.
- Compute gross and net exposure for diagnostics, but do not fail on exposure limits unless explicit limits are added later.

Allowed future policies:

- `untradable_policy = "zero"` to zero untradable targets intentionally.
- `unshortable_policy = "zero"` to zero unshortable short targets intentionally.
- Explicit exposure constraints such as `max_gross`, `target_net`, and `net_tolerance`.

## Runner Integration

Plan source priority should be:

```text
1. target_weights path, when spec.target_weights is present
2. composable spec path, when selection/weighting/portfolio_shape/position_policy is present
3. registered strategy path
```

This keeps external final weights from being accidentally passed through allocator logic.

## Reporting And Auditability

The run artifacts should make the plan source visible:

- `execution_resolution.json` should indicate `plan_source = "target_weights"`.
- `resolved_execution_spec.json` should include the target weights spec.
- Diagnostics should include observed gross and net exposure summary values.
- If a non-default policy modifies weights, the policy and affected count should be recorded.

## Acceptance Criteria

- A signed CSV target weight matrix can be used as the backtest input without `selection`, `weighting`, or `portfolio_shape`.
- Negative weights are preserved in `PositionPlan.target_weights`.
- Negative weights fail when `shorting.enabled = false`.
- The engine receives the same signed target weights that were read for known market dates and symbols, after date/symbol alignment.
- The default path does not normalize, cap, or silently zero any supplied nonzero target.
- Existing `weighting.explicit` tests continue to pass unchanged.
- Existing long-short and sector-neutral portfolio-shape tests continue to pass unchanged.

## Test Plan

Unit tests:

- Signed target weights CSV reader accepts positive, negative, zero, and blank values.
- Reader rejects duplicate dates, duplicate symbols, non-ISO dates, non-numeric values, and infinities.
- Target weights plan builder preserves signs and values.
- Target weights plan builder rejects negative values without `shorting.enabled`.
- Target weights plan builder fails on configured unshortable short targets.

Integration tests:

- Runner executes a target weights spec and produces negative quantities for short targets.
- `signal_dates` schedule works from target weight changes.
- Existing spec-composed and registered-strategy paths still choose their prior plan source.

Regression tests:

- Existing explicit allocator behavior remains long-only unless intentionally deprecated in a later change.
- Existing sector-neutral rank-top-bottom behavior remains unchanged.

## Risks

- The term `weighting.explicit` remains confusing. Mitigation: document it as non-negative allocator behavior and direct users to `target_weights.file` for final signed weights.
- Alignment policy can hide missing symbols if too permissive. Mitigation: keep silent modification limited to missing-as-zero and fail on nonzero untradable/unshortable targets.
- Later hook support could bypass validation. Mitigation: all target weight sources must pass through the same validation function before building `PositionPlan`.

## Implementation Notes

Likely files:

- `backtesting/specs/models.py` for `TargetWeightsSpec`.
- `backtesting/specs/loader.py` for JSON parsing.
- `backtesting/specs/target_weights.py` for file reading, validation, and plan building.
- `backtesting/calculation.py` for plan source priority.
- `tests/specs` or `tests/target_weights` for reader and plan tests.
- `tests/run/test_runner_specs.py` for runner-level coverage.

## Open Follow-Ups

- Add `target_weights.kind = "hook"` after file input is stable.
- Add optional exposure constraints after diagnostics prove useful.
- Later, redesign selection output to support leg-aware construction for Q1/Q5, deciles, and custom spreads inside the composed spec path.
