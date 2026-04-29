# Spec-Driven Selection, Weighting, and Staged Trading Design

## Goal

Make `backtesting` understand investment intent without defaulting every idea to a
`top_n` ranking strategy.

The existing `BacktestEngine` should remain the stable execution core: it receives
target weights, applies the rebalance schedule, fills trades, and computes results.
This design expands the layer before the engine so a spec file can express:

- which universe to consider,
- how securities are selected,
- how selected securities are weighted,
- whether positions are entered all at once or through staged / split trading,
- when the portfolio is re-evaluated and traded.

## Problem

The current default path is easy for a coding agent to interpret as:

```text
strategy = momentum
top_n = N
equal weight
```

That is too narrow. Many intended strategies are not "buy the top N names":

- buy every stock that passes a filter,
- buy stocks whose score is above a threshold,
- buy event-triggered names for a holding window,
- hold an index-like basket by market-cap or float-cap weight,
- enter eligible names through staged / split buying rules.

The system should encode these distinctions directly so an agent such as OpenClaw
can map natural-language requests into the right primitives instead of forcing
them into `top_n`.

## Design principles

1. **Preserve the engine.** Do not redesign `backtesting/engine/core.py` for this
   work. The engine should continue to execute target weights.
2. **Separate investment intent into layers.**
   - `selection`: what to buy or short.
   - `weighting`: how much to allocate to each selected security.
   - `position_policy`: whether to pass weights through or stage/split entries.
   - `schedule` / `execution`: when and how to trade.
3. **Make `top_n` one selection kind, not the default model.**
4. **Prefer spec files for user-facing workflows.**
5. **Keep freedom through registries and hooks.** Common cases should be
   declarative; uncommon cases should use registered Python hooks rather than
   unsafe arbitrary expressions.
6. **Maintain legacy compatibility.** Existing `strategy="momentum"` and
   `top_n` behavior should keep working.

## Proposed architecture

```text
ExecutionSpec
  ├─ UniverseSpec / existing universe handling
  ├─ Feature resolution
  ├─ SelectionSpec
  ├─ WeightingSpec
  ├─ PositionPolicySpec
  ├─ ScheduleSpec
  └─ execution settings
        ↓
Feature frames
        ↓
Selection mask
        ↓
Base target weights
        ↓
PositionPlan via pass-through or staged policy
        ↓
Existing BacktestEngine
```

## Spec model

Extend `backtesting/specs/models.py` with explicit spec objects:

```text
ConditionSpec
SelectionSpec
WeightingSpec
PositionPolicySpec
```

### SelectionSpec

Supported first-class kinds:

| Kind | Meaning |
| --- | --- |
| `rank_top_n` | Rank by a field and select N names. |
| `filter` | Select every security passing all configured filters. |
| `score_threshold` | Compute or read a score and select names above/below a threshold. |
| `event` | Select names based on an event trigger and optional holding rules. |
| `explicit` | Use a provided selection or weight source. |
| `hook` | Delegate selection to a registered Python hook. |

Example: filter-based selection.

```json
{
  "selection": {
    "kind": "filter",
    "conditions": [
      {"field": "momentum_60d", "op": ">", "value": 0},
      {"field": "market_cap", "op": ">=", "value": 100000000000},
      {"field": "avg_trading_value_20d", "op": ">=", "value": 1000000000}
    ]
  }
}
```

Example: rank-based selection.

```json
{
  "selection": {
    "kind": "rank_top_n",
    "field": "momentum_60d",
    "n": 20,
    "ascending": false
  }
}
```

### WeightingSpec

Supported first-class kinds:

| Kind | Meaning |
| --- | --- |
| `equal_weight` | Equal weight among selected names. |
| `market_cap` | Market-cap weight among selected names. |
| `float_market_cap` | Float-cap weight among selected names, with configured fallback. |
| `score` | Weight by positive score values. |
| `inverse_vol` | Weight by inverse volatility. |
| `explicit` | Use provided target weights. |
| `hook` | Delegate weighting to a registered Python hook. |

Example: filter then equal weight.

```json
{
  "selection": {
    "kind": "filter",
    "conditions": [
      {"field": "momentum_60d", "op": ">", "value": 0}
    ]
  },
  "weighting": {
    "kind": "equal_weight"
  }
}
```

Example: filter then market-cap weight.

```json
{
  "selection": {
    "kind": "filter",
    "conditions": [
      {"field": "momentum_60d", "op": ">", "value": 0}
    ]
  },
  "weighting": {
    "kind": "market_cap"
  }
}
```

### PositionPolicySpec

This brings existing staged / split trading into the spec surface.

Supported first-class kinds:

| Kind | Meaning |
| --- | --- |
| `pass_through` | Use base target weights directly. |
| `staged` | Enter or add through configured budget buckets and rule masks. |
| `hook` | Delegate position policy construction to a registered hook. |

The existing `BudgetPreservingStagedPolicy`, `BucketDefinition`, and
`StagedRuleSet` should be reused rather than reimplemented.

Example: condition-passing names are split into three entry buckets.

```json
{
  "selection": {
    "kind": "filter",
    "conditions": [
      {"field": "momentum_60d", "op": ">", "value": 0},
      {"field": "market_cap", "op": ">=", "value": 100000000000}
    ]
  },
  "weighting": {
    "kind": "equal_weight"
  },
  "position_policy": {
    "kind": "staged",
    "buckets": [
      {"id": "entry", "fraction": 0.34},
      {"id": "add_1", "fraction": 0.33},
      {"id": "add_2", "fraction": 0.33}
    ],
    "rules": {
      "entry": {"kind": "selection_passes"},
      "adds": [
        {"kind": "still_passes_after_rebalances", "count": 1},
        {"kind": "still_passes_after_rebalances", "count": 2}
      ],
      "exit": {"kind": "selection_fails"}
    }
  }
}
```

Interpretation:

- A security that newly passes selection activates the first bucket.
- If it still passes on later rebalance evaluations, later buckets activate.
- If it fails the selection condition, active buckets are released.
- The final `PositionPlan.target_weights` remains the only thing the engine sees.

## Feature registry

Add a feature layer that maps field names to frames and required datasets.

Initial registered fields should be safe, explicit, and dataset-backed:

| Field | Meaning |
| --- | --- |
| `close` | Adjusted close. |
| `open` | Adjusted open. |
| `momentum_20d` | 20-day close-to-close momentum. |
| `momentum_60d` | 60-day close-to-close momentum. |
| `market_cap` | Market capitalization. |
| `float_market_cap` | Float market capitalization. |
| `avg_trading_value_20d` | 20-day average trading value. |
| `foreign_ratio` | Foreign ownership ratio. |
| `institution_flow_20d` | 20-day institutional flow. |
| `retail_flow_20d` | 20-day retail flow. |

Arbitrary formulas should not be the first implementation surface. For freedom,
use registered features and hooks first. Formula support can be added later with
validation if needed.

## Natural-language mapping for agents

OpenClaw or another coding agent should map user phrasing like this:

| User intent | Spec interpretation |
| --- | --- |
| "상위 20개" / "top 20" | `selection.kind = rank_top_n` |
| "조건을 통과하는 종목" | `selection.kind = filter` |
| "점수 0 이상" | `selection.kind = score_threshold` |
| "분할 매매" / "분할 매수" | `position_policy.kind = staged` |
| "동일비중" | `weighting.kind = equal_weight` |
| "시총비중" | `weighting.kind = market_cap` |
| "유동시총비중" | `weighting.kind = float_market_cap` |
| "특수 이벤트" | `selection.kind = event` or `hook` |
| "내가 만든 비중 사용" | `weighting.kind = explicit` |

The rule for agents is:

```text
Do not infer top_n unless the user asks for ranking or a maximum number of names.
First identify selection, weighting, position policy, schedule, and execution.
```

## Code areas to change

Likely changes:

```text
backtesting/specs/models.py
backtesting/specs/loader.py
backtesting/specs/resolve.py
backtesting/run.py
backtesting/features/
backtesting/selection/
backtesting/weighting/
tests/specs/
tests/features/
tests/selection/
tests/weighting/
tests/policy/
```

Likely unchanged or minimally touched:

```text
backtesting/engine/core.py
backtesting/reporting/*
existing momentum strategy behavior
```

## Backward compatibility

Existing behavior should continue:

- `RunConfig(strategy="momentum", top_n=20, ...)` remains valid.
- CLI `--strategy momentum --top-n 20` remains valid.
- Existing `ExecutionSpec` files without `selection`, `weighting`, or
  `position_policy` should resolve through the legacy strategy path.

New specs should prefer:

```json
{
  "selection": {"kind": "..."},
  "weighting": {"kind": "..."},
  "position_policy": {"kind": "..."}
}
```

## Testing strategy

Add tests that prove each layer independently:

1. **Spec parsing**
   - Parses `selection`, `weighting`, and `position_policy`.
   - Legacy specs still parse.
2. **Feature resolution**
   - Each supported field declares required datasets.
   - Derived features align to market frames.
3. **Selection**
   - `rank_top_n` selects only N names.
   - `filter` selects all passing names, not top N.
   - `score_threshold` selects threshold-passing names.
4. **Weighting**
   - Equal weight sums correctly by date.
   - Market-cap and float-cap weights normalize within selected names.
5. **Position policy**
   - `pass_through` preserves base weights.
   - `staged` reuses `BudgetPreservingStagedPolicy` and emits valid bucket ledger.
6. **Runner integration**
   - New spec path produces a valid `PositionPlan`.
   - Existing momentum/top_n output remains unchanged.
   - `BacktestEngine` does not need behavioral changes.

## Acceptance criteria

The work is complete when:

- A spec can express a filter strategy that buys all passing securities.
- A spec can express a rank/top-N strategy.
- A spec can express filter selection plus equal weighting.
- A spec can express filter selection plus market-cap or float-cap weighting.
- A spec can express staged / split trading using existing staged policy behavior.
- Unknown fields, operators, selection kinds, weighting kinds, and position-policy
  kinds fail with clear errors.
- Legacy `momentum` / `top_n` tests still pass.
- The engine remains responsible only for executing target weights.

## Deferred decisions

- A general formula DSL is intentionally deferred.
- Intraday event-loop simulation is out of scope for this design.
- Full broker/order lifecycle modeling is out of scope for this design.
- The first implementation should support event-like behavior through selection
  masks, schedules, and hooks rather than replacing the engine.
