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

**Saved Run**:
The persisted artifact bundle from one completed backtest run.
_Avoid_: Report bundle, dashboard payload, raw result object

**Backtest Calculation**:
The end-to-end computation that turns resolved research intent into a saved run, including data loading, feature construction, position planning, engine execution, and artifact writing.
_Avoid_: Dashboard rendering, report design, frontend latency

**Backtest Timing**:
Optional stage timing for a backtest calculation, used to identify performance bottlenecks without changing portfolio results.
_Avoid_: Test pass/fail threshold, frontend performance metric

## Relationships

- **Backtesting** produces saved result artifacts consumed by the **Report Dashboard** and **Research Dashboard**.
- A **Report Dashboard** belongs to **Backtesting** reporting output and is not a web UI.
- A **Research Dashboard** may launch or select runs, but portfolio math remains owned by **Backtesting**.
- **Run Launch** may invoke **Backtesting**, but strategy construction, target weights, engine execution, and performance analytics remain owned by **Backtesting**.
- **Research Intent** is resolved into **Resolved Research Intent**, which produces a **Position Plan** for the Backtesting engine.
- CLI run configuration and dashboard presets are adapters into **Research Intent**, not canonical domain objects.
- A **Saved Run** is produced by **Backtesting** and consumed by the **Report Dashboard** and **Research Dashboard** as a shared read contract.
- Usable-run detection and run config signatures belong to the **Saved Run** contract, not to individual dashboard services.
- **Backtest Calculation** performance is a first-class constraint for the final refactor, alongside Backtesting and Research Dashboard boundary cleanup.
- **Backtest Calculation** speed is evaluated end to end and broken down by data loading, feature and position-plan construction, engine execution, and artifact writing.
- **Backtest Timing** is optional instrumentation on **Backtest Calculation**; default execution behavior must remain unchanged when timing is disabled.
- **Backtest Timing** may be saved as optional metadata on a **Saved Run**; readers must treat missing timing metadata as normal for older runs.

## Example dialogue

> **Dev:** "Should the dashboard compute strategy weights?"
> **Domain expert:** "No. **Backtesting** owns portfolio calculation. The **Research Dashboard** only presents saved or launched results."

## Flagged ambiguities

- "dashboard" was used to mean both **Report Dashboard** and **Research Dashboard**; resolved: the final refactor treats `dashboard/` as **Research Dashboard** and `backtesting/reporting` output as **Report Dashboard**.
- "launching from the dashboard" could imply dashboard-owned backtesting; resolved: **Run Launch** is orchestration only and delegates all portfolio calculation to **Backtesting**.
- "run config" was used like the main research model; resolved: **Research Intent** is the canonical concept, currently represented by `ExecutionSpec`, while `RunConfig` remains an adapter for CLI and dashboard launch compatibility.
- `SavedRun` currently lives under reporting code, but the resolved domain concept is **Saved Run**: a Backtesting result contract rather than a report-only object.
- Physical movement of `SavedRun`, `RunReader`, and `RunWriter` is deferred; existing `backtesting.reporting.*` imports remain compatible while the domain treats **Saved Run** as a Backtesting result contract.
- "calculation speed" is treated as **Backtest Calculation** speed unless narrowed further; it excludes frontend rendering and report visual redesign.
- Backtest performance work must be measurement-led: optimize the largest measured Backtest Calculation stage first while preserving portfolio math and strategy meaning.
- Timing/profiling support should use the standard library only; tests verify timing records are produced structurally, not that a stage meets an environment-dependent millisecond threshold.
- Timing metadata should be written only for explicit profiling paths such as dashboard launch or profile-enabled runner calls, not for every default backtest.
