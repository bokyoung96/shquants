# Backtesting performance refactor plan

## Goal

Reduce repeated data loading and backtest loop overhead while preserving portfolio math and keeping names simple.

## Scope

- Add an in-process parquet frame cache with write invalidation.
- Keep returned frames isolated from caller mutation.
- Refactor the engine loop to work on arrays internally instead of repeated pandas row lookups.
- Preserve public runner, loader, and engine behavior.

## Out of scope

- New dependencies.
- Strategy math changes.
- Report/dashboard redesign.
- Cross-process cache invalidation.

## Safety checks

- Add cache tests before changing store behavior.
- Keep existing engine behavior tests green.
- Run focused data, engine, execution, and run tests after edits.
