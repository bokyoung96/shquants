# shquants Context

shquants is a quantitative research workspace where backtesting produces auditable research results and dashboards present those results without owning portfolio calculation.

## Language

**Backtesting**:
The domain core that defines research intent, loads market data, constructs target weights, executes portfolio simulation, and produces result artifacts.
_Avoid_: Dashboard, UI, presentation layer

**Report Dashboard**:
A static post-backtest report generated from saved backtest results for review or sharing.
_Avoid_: Web dashboard, app dashboard, frontend

**Research Dashboard**:
The interactive application surface that reads backtest results and presents research views.
_Avoid_: Backtesting engine, strategy runner, report dashboard

**Run Launch**:
A dashboard-initiated request to reuse an existing saved run or delegate a missing preset execution to Backtesting.
_Avoid_: Strategy calculation, portfolio simulation, dashboard-owned backtest

**Research Intent**:
The canonical description of what a backtest should evaluate before data availability and execution details are resolved.
_Avoid_: Run config, dashboard preset, ad hoc strategy parameters

**Resolved Research Intent**:
A research intent after dataset needs, schedule details, hooks, and fallbacks have been resolved for execution.
_Avoid_: Raw user input, dashboard config

**Position Plan**:
The target-weight plan produced from resolved research intent before execution costs and fills are applied.
_Avoid_: Strategy result, backtest result

**Portfolio Shape**:
The exposure structure of a position plan, such as long-only, long-short, or sector-neutral.
_Avoid_: Strategy signal, alpha model, weighting method

**Portfolio Shape Spec**:
The research-intent spec axis that declares the intended portfolio shape independently from selection, weighting, and position policy.
_Avoid_: Selection kind, weighting kind, strategy name

**Signal-Triggered Trading**:
A trading style where dated signal masks drive position entry, maintenance, and exit instead of a fixed top-N rebalance cadence.
_Avoid_: Top-N ranking, calendar-only rebalancing

**Shorting Assumptions**:
The explicit research assumptions that make negative target weights executable, including borrow fee, shortable masks, and cash collateral reserve.
_Avoid_: Broker margin engine, locate inventory simulator

**Saved Run**:
The persisted artifact bundle from one completed backtest run.
_Avoid_: Report bundle, dashboard payload, raw result object

**Backtest Calculation**:
The end-to-end computation that turns resolved research intent into a saved run, including data loading, feature construction, position planning, engine execution, and artifact writing.
_Avoid_: Dashboard rendering, report design, frontend latency

**Backtest Timing**:
Optional stage timing for a backtest calculation, used to identify performance bottlenecks without changing portfolio results.
_Avoid_: Test pass/fail threshold, frontend performance metric

**OpenClaw Playbook**:
An agent-facing guide that maps research intent into specs, commands, and verification steps for OpenClaw.
_Avoid_: Domain model, user manual, generated run artifact

## Relationships

- **Backtesting** produces saved result artifacts consumed by the **Report Dashboard** and **Research Dashboard**.
- A **Report Dashboard** belongs to **Backtesting** reporting output and is not a web UI.
- A **Research Dashboard** may launch or select runs, but portfolio math remains owned by **Backtesting**.
- **Run Launch** may invoke **Backtesting**, but strategy construction, target weights, engine execution, and performance analytics remain owned by **Backtesting**.
- **Research Intent** is resolved into **Resolved Research Intent**, which produces a **Position Plan** for the Backtesting engine.
- A **Position Plan** has a **Portfolio Shape**; the shape describes exposure structure, not how alpha was generated.
- A **Portfolio Shape Spec** is part of **Research Intent** and is resolved before the **Position Plan** is handed to the engine.
- Long/short candidate selection belongs to `selection`; `rank_top_bottom` selects the long and short candidate legs, while `portfolio_shape` declares how those legs become exposure.
- The default long-short **Portfolio Shape** is dollar-neutral: long gross exposure 1.0, short gross exposure 1.0, net exposure 0.0, and gross exposure 2.0.
- `portfolio_shape.kind = "long_short"` and `portfolio_shape.kind = "sector_neutral"` currently require `selection.kind = "rank_top_bottom"` so leg membership is explicit before exposure construction.
- `portfolio_shape.kind = "sector_neutral"` requires a registered grouping feature, currently `sector`, and balances selected long and short legs inside each sufficiently populated group.
- `portfolio_shape.group_budget` controls how sector-neutral long and short gross budgets are allocated across qualified groups; `equal_group` gives each group the same budget, while `proportional_selected` allocates by selected leg count.
- **Shorting Assumptions** are explicit through `shorting`: negative target weights require `shorting.enabled`, borrow fees accrue daily from short notional, shortable masks block unshortable short targets, and cash collateral reserves reduce buy capacity.
- **Signal-Triggered Trading** is expressed through signal-aware selection and schedule semantics; OpenClaw must not translate it into top-N ranking unless the user explicitly asks for ranking.
- `schedule.kind` owns signal evaluation cadence: named calendar schedules evaluate and hold positions on scheduled dates, while signal-driven schedules trade when signal state changes.
- `schedule.kind = "signal_dates"` means rebalance on dates where final target weights change from the previous date, including entries, exits, and material weight changes.
- `schedule.kind = "signal_dates"` may include nested `evaluation` to evaluate signals on a named or custom schedule, then rebalance only when the evaluated target changes.
- Explicitly provided named schedules freeze spec-driven target weights to scheduled evaluation dates and forward-fill positions until the next scheduled date.
- CLI run configuration and dashboard presets are adapters into **Research Intent**, not canonical domain objects.
- A **Saved Run** is produced by **Backtesting** and consumed by the **Report Dashboard** and **Research Dashboard** as a shared read contract.
- Usable-run detection and run config signatures belong to the **Saved Run** contract, not to individual dashboard services.
- **Backtest Calculation** performance is a first-class constraint for the final refactor, alongside Backtesting and Research Dashboard boundary cleanup.
- **Backtest Calculation** speed is evaluated end to end and broken down by data loading, feature and position-plan construction, engine execution, and artifact writing.
- **Backtest Calculation** optimization starts from measured timing stages; the largest measured bottleneck is optimized first unless the user deliberately narrows the scope.
- **Backtest Timing** is optional instrumentation on **Backtest Calculation**; default execution behavior must remain unchanged when timing is disabled.
- **Backtest Timing** may be saved as optional metadata on a **Saved Run**; readers must treat missing timing metadata as normal for older runs.
- The **OpenClaw Playbook** explains how agents should drive **Backtesting** without owning portfolio calculation.

## Example dialogue

> **Dev:** "Should the dashboard compute strategy weights?"
> **Domain expert:** "No. **Backtesting** owns portfolio calculation. The **Research Dashboard** only presents saved or launched results."

## Flagged ambiguities

- "dashboard" was used to mean both **Report Dashboard** and **Research Dashboard**; resolved: the final refactor treats `dashboard/` as **Research Dashboard** and `backtesting/reporting` output as **Report Dashboard**.
- "launching from the dashboard" could imply dashboard-owned backtesting; resolved: **Run Launch** is orchestration only and delegates all portfolio calculation to **Backtesting**.
- "run config" was used like the main research model; resolved: **Research Intent** is the canonical concept, currently represented by `ExecutionSpec`, while `RunConfig` remains an adapter for CLI and dashboard launch compatibility.
- "long-only", "long-short", and "sector-neutral" were resolved as **Portfolio Shape** variants rather than strategy names.
- "long-short as selection/weighting/position_policy" was rejected; resolved: **Portfolio Shape Spec** is a separate spec axis named `portfolio_shape`.
- "top/bottom ranking inside portfolio_shape" was rejected; resolved: `selection.kind = "rank_top_bottom"` owns long and short candidate selection.
- "default long-short exposure" was resolved as dollar-neutral: `gross_long = 1.0` and `gross_short = 1.0`.
- "signal-based irregular trading" was resolved as **Signal-Triggered Trading**: signal masks, not calendar cadence or top-N ranking, are the primary trading driver.
- A separate `signal_evaluation` spec axis was rejected; resolved: extend `schedule.kind` so OpenClaw has one place to decide when signals are evaluated and traded.
- `signal_dates` was resolved against final target-weight changes rather than raw signal changes so the rule works across event, filter, explicit, and hook-driven plans.
- Combining `signal_dates` with a named calendar base schedule was resolved through nested `schedule.evaluation`, keeping trade-trigger semantics and signal-evaluation cadence in the same schedule axis.
- Short borrow and collateral assumptions are first-class enough for research simulation, but broker-specific locate, recall, liquidation, and margin waterfall behavior remain out of scope.
- `SavedRun` currently lives under reporting code, but the resolved domain concept is **Saved Run**: a Backtesting result contract rather than a report-only object.
- Physical movement of `SavedRun`, `RunReader`, and `RunWriter` is deferred; existing `backtesting.reporting.*` imports remain compatible while the domain treats **Saved Run** as a Backtesting result contract.
- "calculation speed" is treated as **Backtest Calculation** speed unless narrowed further; it excludes frontend rendering and report visual redesign.
- Backtest performance work must be measurement-led: optimize the largest measured Backtest Calculation stage first while preserving portfolio math and strategy meaning.
- Timing/profiling support should use the standard library only; tests verify timing records are produced structurally, not that a stage meets an environment-dependent millisecond threshold.
- Timing metadata should be written only for explicit profiling paths such as dashboard launch or profile-enabled runner calls, not for every default backtest.
- "LLM wiki" was resolved as **OpenClaw Playbook**: a repo-local agent guide, not a separate wiki app or runtime feature.
