# EMP008 Optional Factor Timing Design

## Goal

Add an optional, auditable factor-timing stage inside EMP008. The first policy is directional factor momentum. Timing is disabled by default, and the disabled path must preserve existing EMP008 behavior and outputs.

## Scope

- EMP008 only; this is not a repository-wide timing framework.
- Support every EMP008 factor set through registry-derived factor direction metadata.
- Implement one policy: `momentum`.
- Keep the existing factor-return regression, expected-alpha estimator, risk model, optimizer, and backtest cost model unchanged.
- Do not add dependencies.

## Public Contract

`Emp008Config.factor_timing` is either `None` or an immutable `FactorTimingConfig`.

```python
FactorTimingConfig(
    policy="momentum",
    fast_lookback=6,
    slow_lookback=12,
    strong_multiplier=1.25,
    neutral_multiplier=1.00,
    weak_multiplier=0.75,
)
```

`None` is the default and bypasses every timing calculation. CLI runners expose `--factor-timing {none,momentum}`. The first version keeps lookbacks and multipliers at their documented defaults rather than adding a broad parameter grid to the CLI.

## Architecture

Create `backtesting/strategies/emp008/factor_timing.py` as the sole owner of timing configuration, validation, signal calculation, and diagnostic schemas.

The module accepts:

- monthly realized factor returns;
- registered factor directions;
- resolved base factor weights;
- the rebalance date;
- optional timing configuration.

It returns a `FactorTimingDecision` containing:

- normalized factor weights to apply;
- one diagnostic row per factor;
- the last factor-return date actually used.

`strategy.py` remains the orchestration owner. It passes only factor returns strictly earlier than the rebalance date to the timing module, applies returned weights through the existing `apply_factor_weights`, and accumulates timing diagnostics alongside target and active weights.

## Timing Rule

For each alpha factor, orient monthly factor returns so positive values mean the registered preferred direction worked:

```text
HIGH factor: directional return = raw factor return
LOW factor:  directional return = -raw factor return
```

Compute compounded directional returns over the trailing fast and slow windows:

```text
fast > 0 and slow > 0  -> strong multiplier
fast < 0 and slow < 0  -> weak multiplier
otherwise              -> neutral multiplier
```

Zero belongs to the neutral state. Apply multipliers to base weights and normalize the result to sum to one. Multipliers must be finite and strictly positive, so the timing stage cannot reverse a factor or produce an all-zero portfolio.

## Timing And Look-Ahead Contract

For a target dated `t`, timing may use only factor-return observations with dates earlier than `t`. The current `t-1 -> t` factor return is excluded from the timing decision even though it is available elsewhere in the existing risk pipeline. Diagnostics record `last_signal_date` so this contract is auditable.

If fewer than `slow_lookback` observations are available, the timing module returns normalized base weights with status `insufficient_history` for every factor. It does not partially time factors with inconsistent histories.

## Outputs

When timing is enabled, `Emp008Result` includes long-form factor-timing diagnostics and writes:

- `factor_timing.parquet`
- `factor_timing.csv`

Columns:

```text
rebalance_date
factor
direction
base_weight
fast_return
slow_return
state
multiplier
timed_weight
last_signal_date
```

When timing is disabled, diagnostics are empty and no timing files are written. Run summaries record the timing policy as `none` or `momentum`.

## Error Handling

- Reject unsupported policy names.
- Reject non-positive lookbacks and `fast_lookback > slow_lookback`.
- Reject non-finite or non-positive multipliers.
- Reject missing factor-return columns, missing directions, duplicate factors, or non-finite input returns.
- Reject non-finite or negative base weights and an all-zero base vector.
- Fall back only for insufficient history; structural data errors must fail loudly.

## Testing

- Disabled timing returns normalized base weights and creates no diagnostics.
- Strong, neutral, and weak momentum states map to the configured multipliers.
- LOW factor direction reverses raw factor returns before state classification.
- Timed weights are finite, non-negative, and sum to one.
- Changing the factor return stamped with the target rebalance date cannot change that target's timing decision.
- Insufficient history falls back to base weights with an explicit status.
- Existing EMP008 factor-weight tests remain unchanged and pass.
- A small strategy integration test verifies timing diagnostics are accumulated and written only when enabled.
- CLI tests verify default `none` behavior and explicit `momentum` configuration.

## Non-Goals

- Factor IC timing, macro timing, regime classifiers, or machine learning.
- Smoothing across prior timing decisions.
- Dynamic tracking error.
- A timing-parameter grid search.
- Changing the existing 36-month expected-alpha estimator.
