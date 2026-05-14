# OpenClaw Playbook

This guide tells OpenClaw how to turn research intent into backtesting specs, run commands, and verification checks in this repo.

## Core Rule

Backtesting owns portfolio calculation. OpenClaw may create specs, run backtests, inspect saved runs, and report results, but it must not move portfolio math into dashboards or reporting code.

## Intent Mapping

Map user intent into `ExecutionSpec` fields before choosing a legacy strategy.

| User intent | Preferred spec surface |
| --- | --- |
| Buy every name passing conditions | `selection.kind = "filter"` |
| Buy top N by score | `selection.kind = "rank_top_n"` |
| Buy top names and short bottom names by score | `selection.kind = "rank_top_bottom"` |
| Buy names above a score cutoff | `selection.kind = "score_threshold"` |
| Equal weight selected names | `weighting.kind = "equal_weight"` |
| Market-cap weight selected names | `weighting.kind = "market_cap"` |
| Float-market-cap weight selected names | `weighting.kind = "float_market_cap"` |
| Use external target weights | `weighting.kind = "explicit"` |
| Stage entries over multiple rebalances | `position_policy.kind = "staged"` |
| Long-only exposure | `portfolio_shape.kind = "long_only"` |
| Long-short exposure | `portfolio_shape.kind = "long_short"` |
| Sector-neutral exposure | `portfolio_shape.kind = "sector_neutral"` |
| Signal appears, event fires, condition turns on, or irregular signal-driven trading | Start with `selection.kind = "event"`, `"filter"`, or `"hook"`; do not default to `rank_top_n` |
| Custom selection or weighting logic | Registered `hook`, not ad hoc unsafe expressions |

Do not infer `top_n` unless the user asks for ranking, "top", "bottom", or a maximum number of names.

For long-short and sector-neutral requests, keep the split clear:
- `selection.kind = "rank_top_bottom"` chooses candidate long and short legs.
- `portfolio_shape` turns those legs into the intended exposure structure.

For Signal-Triggered Trading requests, do not infer a calendar-only strategy:
- Use `selection.kind = "event"` when the user describes dated signal events with a holding window.
- Use `selection.kind = "filter"` when the user describes a condition that can turn on and off by date.
- Use registered hooks when signal construction or trading dates need custom code.
- Treat `schedule.kind` as the single place that decides signal evaluation cadence.

Schedule interpretation:
- `schedule.kind = "named", name = "weekly"` means evaluate selection and weights only on weekly schedule dates, keep those positions until the next scheduled date, and ignore mid-week signal changes.
- `schedule.kind = "named", name = "monthly"` means evaluate only on monthly schedule dates and hold until the next scheduled date.
- `schedule.kind = "named", name = "daily"` means evaluate every trading date.
- `schedule.kind = "signal_dates"` means rebalance only when final target weights change from the previous date.
- `schedule.kind = "signal_dates"` without `evaluation` evaluates signals daily.
- `schedule.kind = "signal_dates"` with `evaluation` evaluates signals only on that nested schedule, then trades only when the evaluated target changes.
- Do not add a separate `signal_evaluation` axis unless this rule proves insufficient.

## Portfolio Shapes

A Portfolio Shape is the exposure structure of a position plan. It is declared through `portfolio_shape`, separate from the alpha signal, selection rule, weighting method, and position policy.

### Long-only

Use positive target weights that normally sum to 1.0 on invested dates.

Current first-class paths:
- Legacy registered strategies such as `trend_rank`
- Spec-driven `selection` plus non-negative `weighting`

Declarative shape:

```json
{
  "portfolio_shape": {"kind": "long_only"}
}
```

### Long-short

Use positive long weights and negative short weights. The declarative path is `selection.kind = "rank_top_bottom"` plus `portfolio_shape.kind = "long_short"`.

Declarative shape:

```json
{
  "selection": {"kind": "rank_top_bottom", "field": "momentum_60d", "top_n": 20, "bottom_n": 20},
  "portfolio_shape": {"kind": "long_short", "gross_long": 1.0, "gross_short": 1.0},
  "shorting": {"enabled": true, "borrow_fee_annual": 0.0, "cash_collateral_ratio": 1.0}
}
```

Default exposure:
- Long gross exposure: `1.0`
- Short gross exposure: `1.0`
- Net exposure: `0.0`
- Gross exposure: `2.0`

Before treating long-short as production-ready, verify:
- Target weights contain both positive and negative legs.
- Net exposure is the intended value, commonly near 0.0 for dollar-neutral tests.
- Gross exposure is explicit, commonly long gross 1.0 and short gross 1.0.
- Shorting is explicitly enabled.
- Borrow, collateral, fee, sell-tax, and slippage assumptions are documented for short trades.

### Sector-neutral

Use long and short legs balanced within each sector. The declarative path is `selection.kind = "rank_top_bottom"` plus `portfolio_shape.kind = "sector_neutral"`.

Declarative shape:

```json
{
  "selection": {"kind": "rank_top_bottom", "field": "momentum_60d", "top_n": 3, "bottom_n": 3},
  "portfolio_shape": {"kind": "sector_neutral", "group_field": "sector", "group_budget": "equal_group"},
  "shorting": {"enabled": true}
}
```

Before treating sector-neutral as production-ready, verify:
- Sector data is available and aligned to the backtest date and symbol grid.
- Undersized sectors are skipped or handled deliberately.
- Group long and short budgets match the intended neutrality rule.

Group budget options:
- `equal_group`: each qualified sector gets the same long and short gross budget.
- `proportional_selected`: sectors get long and short gross budget in proportion to selected long/short leg count.

## Performance Work

Use Backtest Timing, not wall-clock guesses alone.

Optimization order:
1. Run with profiling enabled.
2. Compare `data_load`, `plan_build`, `engine_run`, and `write_artifacts`.
3. Optimize the largest measured stage first unless the user narrows scope.
4. Preserve portfolio math and strategy meaning.
5. Re-run focused tests and capture before/after timing.

Known current direction:
- The engine loop is already array-based.
- Recent measured bottlenecks were in plan construction for benchmark overlay style strategies.

## Run Commands

Run a preset:

```powershell
uv run python -m backtesting.run --preset <preset_id>
```

Run a spec file:

```powershell
uv run python -m backtesting.run --spec path\to\spec.json
```

Run the signal-triggered example:

```powershell
uv run python -m backtesting.run --spec docs\openclaw\examples\signal-dates-filter.json
```

Run the signal-triggered weekly-evaluation example:

```powershell
uv run python -m backtesting.run --spec docs\openclaw\examples\signal-dates-weekly-evaluation.json
```

Run the long-short example:

```powershell
uv run python -m backtesting.run --spec docs\openclaw\examples\rank-top-bottom-long-short.json
```

Run the sector-neutral example:

```powershell
uv run python -m backtesting.run --spec docs\openclaw\examples\rank-top-bottom-sector-neutral.json
```

Run focused tests:

```powershell
uv run python -m pytest tests/specs tests/selection tests/weighting tests/construction tests/engine
```

Run broader backtesting verification:

```powershell
uv run python -m pytest tests/strategies tests/run tests/reporting tests/dashboard
```

## Verification Checklist

For new strategy surfaces:
- Spec parsing accepts the intended shape and rejects unknown fields clearly.
- Position plan weights align to market dates and symbols.
- `validate_position_plan` passes.
- Existing legacy strategy tests still pass.
- Saved run artifacts remain readable by reporting and dashboard code.

For long-short or sector-neutral work:
- Add construction tests for leg membership, net exposure, gross exposure, and undersized universes.
- Add runner/spec integration tests only after the declarative surface exists.
- Check reports and dashboards can display negative weights without assuming long-only semantics.

For performance work:
- Include timing evidence in the final report.
- Do not use environment-dependent millisecond thresholds as test pass/fail criteria.
- Keep timing optional so default backtests behave the same when profiling is disabled.

## Current Limits

Current declarative specs support event-like signal behavior and target-weight-change-driven signal schedules, but not every calendar/signal combination is first-class yet.

Supported now:
- `selection.kind = "event"` with `hold_days`
- `selection.kind = "filter"` for date-varying conditions
- `selection.kind = "rank_top_bottom"` for long and short candidate legs
- `portfolio_shape.kind = "long_short"` with explicit long and short gross exposure
- `portfolio_shape.kind = "sector_neutral"` with a registered grouping field such as `sector`
- `portfolio_shape.group_budget = "equal_group"` and `"proportional_selected"`
- `shorting.enabled` for negative target weights
- `shorting.borrow_fee_annual` for daily borrow fee accrual
- `shorting.shortable_field` for masking unshortable short targets through a registered boolean feature such as `shortable`
- `shorting.cash_collateral_ratio` for reserving cash against short notional before scaling buys
- `schedule.kind = "named"` with `daily`, `weekly`, or `monthly`; explicitly provided named schedules evaluate and hold positions on scheduled dates
- `schedule.kind = "custom_dates"` when dates are already known
- `schedule.kind = "signal_dates"` for rebalancing only when final target weights change
- `schedule.kind = "signal_dates"` with nested `evaluation` for weekly, monthly, daily, or custom signal evaluation cadence
- full-plan hooks that return both a position plan and a custom schedule

Shortable-mask behavior:
- If a selected short candidate is not shortable, its short target is set to `0`.
- The construction does not search for replacement short candidates in the same run; short gross exposure can shrink when shortability blocks a selected name.

Out-of-scope advanced limits:
- beta-neutral, factor-neutral, and optimizer-based constraints
- partial locate fills, recall risk, and broker-level margin liquidation
- sector-neutral variants beyond registered group fields and the supported group-budget modes

## Unknowns To Surface

Ask for clarification or record an assumption when the user request leaves these unspecified:

- Whether "signal-based" means event hold windows, date-varying filter conditions, explicit target weights, or a custom hook.
- Whether long-short should be dollar-neutral, net-long, beta-neutral, or constrained by short availability.
- Which grouping field defines neutrality when the user says sector-neutral but does not name a taxonomy.
- Whether a weekly or monthly request means scheduled signal evaluation or daily signal evaluation with less frequent execution.
- Whether transaction costs, borrow costs, sell tax, and slippage should use defaults or a research-specific assumption.

Signal-triggered schedule surface:

```json
{
  "selection": {"kind": "event", "field": "event_flag", "hold_days": 5},
  "weighting": {"kind": "equal_weight"},
  "schedule": {"kind": "signal_dates", "weight_change_tolerance": 1e-8}
}
```

`signal_dates` rules:
- Previous target weight `0` to current nonzero target weight: entry date.
- Previous nonzero target weight to current `0`: exit date.
- Previous nonzero target weight to different nonzero target weight: rebalance date.
- Target weights unchanged within `weight_change_tolerance`: no trade.
- The first date with any nonzero target weight is a rebalance date.

When a user explicitly asks for weekly or monthly signal evaluation, prefer:

```json
{
  "selection": {"kind": "filter", "conditions": []},
  "weighting": {"kind": "equal_weight"},
  "schedule": {
    "kind": "signal_dates",
    "evaluation": {"kind": "named", "name": "weekly"}
  }
}
```

## Cleanup Guidance

Generated runtime state and local build outputs may be ignored or removed when untracked. Do not delete tracked tests or planning docs as "cleanup" without a specific replacement or archival decision.
