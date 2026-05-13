# Backtesting and dashboard final refactor plan

## Goal

Tighten the boundary between Backtesting and the Research Dashboard while adding measurement-led visibility into Backtest Calculation speed.

## Scope

- Add optional Backtest Timing instrumentation around data loading, position-plan construction, engine execution, and artifact writing.
- Persist timing as optional Saved Run metadata only when profiling is enabled.
- Read missing timing metadata as normal for older Saved Runs.
- Enable profiling for dashboard-launched missing preset calculations.
- Consolidate usable-run and config-signature helpers so dashboard services share one Saved Run contract.
- Preserve existing `backtesting.reporting.*` imports for compatibility.

## Out of scope

- Moving `SavedRun`, `RunReader`, or `RunWriter` to a new physical package.
- Changing strategy math, portfolio math, or engine semantics.
- Changing Report Dashboard visual design or Research Dashboard frontend UX.
- Adding benchmark dependencies or time-threshold assertions.

## Safety checks

- Add tests before implementation.
- Verify timing tests fail before production code changes.
- Keep timing disabled by default.
- Keep older Saved Runs readable when `timing.json` is absent.
- Run focused backtesting/dashboard tests after edits.
