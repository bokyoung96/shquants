# WRDS Folder Architecture Design

## Goal

Refactor the `wrds/` package so its folders communicate domain responsibility clearly while preserving the existing CLI behavior and keeping `backtesting` integration out of scope.

The refactor should make WRDS downloads feel like one coherent data-acquisition subsystem, not a set of unrelated scripts for `data`, `us`, `universe`, and `options`.

## Scope

In scope:

- Reorganize code under `wrds/` into focused packages.
- Preserve existing `wrds/run.py` command behavior and output paths.
- Keep Strategy Pattern, Dependency Injection, Registry Pattern, Protocol interfaces, and composition-first design.
- Consolidate repeated CSV save and progress/reporting behavior behind shared download/output services.
- Make WRDS market data categories visible: prices, consensus, fundamentals, and indexes.
- Keep existing tests passing, updating imports and adding regression coverage where needed.

Out of scope:

- Wiring WRDS outputs into `backtesting` execution or ingest.
- Changing the meaning, schema, or default destination of downloaded files.
- Adding new dependencies.
- Replacing local QuantiWise raw/parquet handling.

## Current Problem

The current code already reflects several requested design principles:

- `US`, `Options`, and `Universe` use injected sources/builders/strategies.
- Domain-specific registries compose default objects.
- Protocol interfaces exist in `us.py`, `options.py`, and `universe.py`.
- `wrds data` uses shared data source/table specifications from `backtesting.data`.

The weak point is folder and responsibility structure. The `wrds/` root currently contains large domain files:

- `us.py` mixes stock-name sources, FactSet sources, universe building, service methods, file saves, progress display, and CSV output.
- `options.py` mixes OptionMetrics sources, service orchestration, and raw file saves.
- `provider.py` directly writes files for the universe workflow.
- `download.py`, `backtesting.data.Pipeline`, `US.save_*`, `Options.save_raw`, and `UniverseFlow.run` all participate in saving files, but the responsibility is not centralized.

This makes `us`, `options`, and `universe` feel disconnected from other downloadable data such as WRDS CRSP prices, IBES consensus, and Compustat fundamentals.

## Target Architecture

The WRDS package should be organized by domain and shared responsibility:

```text
wrds/
  core/
    client.py
    registry.py
    io.py
    sql.py
    workflow.py

  downloads/
    service.py
    batch.py
    manifest.py

  marketdata/
    catalog.py
    specs.py
    prices.py
    consensus.py
    fundamentals.py
    indexes.py
    workflow.py

  universes/
    us/
      sources.py
      strategies.py
      service.py
      registry.py
      workflow.py
    factset/
      sources.py
      strategies.py
      service.py
      registry.py
      workflow.py

  derivatives/
    options/
      sources.py
      service.py
      registry.py
      workflow.py

  mapping.py
  provider.py
  run.py
```

`mapping.py` can remain at the package root during this refactor because it is a shared date/link cleaning helper used by multiple domains. Moving it under `core/` is optional only if the implementation remains small and behavior-preserving.

## Package Responsibilities

### `core/`

Owns cross-cutting primitives:

- `client.py`: WRDS client and client Protocols.
- `registry.py`: shared named-object registry primitives.
- `io.py`: CSV writer, saved-file records, and common output path handling.
- `sql.py`: small SQL helpers such as limit clauses and yearly date predicates.
- `workflow.py`: workflow Protocol and registry.

`core/` should not know about CRSP, IBES, Compustat, FactSet, US universes, or OptionMetrics.

### `downloads/`

Owns reusable download and save orchestration:

- simple query/table download service for `wrds query` and `wrds table`;
- batch saving of named DataFrames to CSV;
- optional manifest/result formatting for multi-file downloads;
- shared progress and printed output conventions where practical.

Domain services should return DataFrames or named file bundles. They should not directly call `to_csv` unless a narrow exception is justified by tests.

### `marketdata/`

Owns general WRDS market-data source definitions used by the `wrds data` command:

- `prices.py`: CRSP price, return, delisting, and daily/monthly security tables.
- `consensus.py`: IBES estimates, actuals, summary, and identifier tables.
- `fundamentals.py`: Compustat company, security, and fundamental tables.
- `indexes.py`: CRSP index and S&P 500 tables.
- `catalog.py`: assembles the source registry.
- `workflow.py`: executes the `wrds data ...` command.

This package makes explicit that WRDS prices and consensus are first-class downloadable data groups, analogous to the local QuantiWise raw files under `raw/`, while still keeping backtesting ingestion out of scope.

### `universes/`

Owns universe and identifier mapping workflows.

`universes/us/` handles the richer current/history/at/latest US universe workflow built from CRSP stock-name data and FactSet links.

`universes/factset/` handles the smaller legacy FactSet universe flow that currently lives in `universe.py`.

These packages should keep source acquisition, build strategies, service orchestration, registry composition, and CLI workflow glue separate.

### `derivatives/`

Owns derivative-specific WRDS data workflows.

`derivatives/options/` handles OptionMetrics raw downloads. Its source classes still fetch link, metadata, and price tables, but file output should flow through the shared `downloads/` batch saver.

### `provider.py` and `run.py`

`provider.py` should become a thin assembler:

- build the market-data source registry;
- build the workflow registry;
- expose these registries to `run.py`.

`run.py` should remain the CLI parser and dispatcher. It should not contain domain logic.

## Design Principles

Use Protocol interfaces for consumers:

- clients that can query WRDS;
- sources that return DataFrames;
- builders/strategies that transform DataFrames;
- writers that persist batches;
- workflows that run CLI commands.

Prefer composition over inheritance:

- services compose sources, strategies, and writers;
- registries compose named defaults;
- workflows compose services and output writers.

Use Strategy Pattern where behavior varies:

- universe construction strategies;
- mapping/link selection strategies if they grow beyond the current builder;
- output behavior only when it genuinely varies.

Use Registry Pattern for discoverable components:

- market data sources;
- workflows;
- domain defaults for US universe, FactSet universe, and options.

Avoid speculative abstractions:

- do not introduce broker concepts unless a real WRDS broker/execution concern appears;
- do not split one-method helpers just to mirror patterns;
- keep the public CLI stable and let tests drive import compatibility.

## Data Flow

For `wrds data`:

1. `run.py` parses selections and options.
2. `provider.py` supplies the market-data source registry.
3. `marketdata.workflow` builds a plan from `marketdata.catalog`.
4. Existing data pipeline behavior saves selected WRDS tables to CSV partitions and manifests.

For `wrds us current/history/at/latest`:

1. `run.py` dispatches to the US workflow.
2. `universes.us.workflow` composes the US service and batch writer.
3. `universes.us.service` fetches source DataFrames and applies the builder strategy.
4. `downloads.batch` writes named output files and reports saved paths.

For `wrds universe`:

1. `run.py` dispatches to the FactSet universe workflow.
2. `universes.factset.service` fetches links and applies the latest-link strategy.
3. `downloads.batch` saves `fscrsplink.csv` and `universe.csv`.

For `wrds options raw`:

1. `run.py` dispatches to the options workflow.
2. `derivatives.options.service` fetches link, metadata, and price DataFrames.
3. `downloads.batch` saves the raw OptionMetrics files.

## Compatibility Requirements

The following behavior must remain stable:

- `wrds/run.py check`
- `wrds/run.py query`
- `wrds/run.py table`
- `wrds/run.py data`
- `wrds/run.py universe`
- `wrds/run.py us current`
- `wrds/run.py us history`
- `wrds/run.py us at`
- `wrds/run.py us latest`
- `wrds/run.py options raw`

Existing output paths and filenames should remain unchanged unless a test demonstrates an intentional compatibility expectation update. This refactor should not change SQL semantics, row cleaning, date handling, default limits, or generated file names.

## Testing Strategy

Before edits:

- Run the current WRDS tests to lock behavior.

During refactor:

- Move tests only when necessary; prefer keeping the existing `wrds/tests/test_wrds.py` coverage until the package split is stable.
- Add focused tests for shared batch saving if existing assertions do not cover it.
- After each major package extraction, run the targeted WRDS tests.

After refactor:

- Run all WRDS tests.
- Run relevant backtesting data tests only to confirm the shared `backtesting.data` imports were not broken.

## Cleanup Plan

1. Lock behavior with existing WRDS tests.
2. Add `core/` and `downloads/` shared primitives without changing behavior.
3. Extract `marketdata/` from `provider.py` and preserve `wrds data` tests.
4. Extract the small FactSet universe workflow into `universes/factset/`.
5. Extract OptionMetrics into `derivatives/options/`.
6. Extract US universe into `universes/us/`.
7. Thin `provider.py` and `run.py` to assembly and dispatch.
8. Delete obsolete compatibility shims only after tests prove they are unused or intentionally replaced.

## Risks

- Import compatibility can break because existing tests import root-level modules such as `us`, `options`, and `universe`.
- The package currently appears to support direct script execution from inside `wrds/`, so relative imports must be handled carefully.
- Moving save logic into a common batch writer can accidentally change printed output order or exact filenames.
- The `backtesting.data` dependency should remain narrow; this refactor must not pull WRDS details into backtesting.

## Success Criteria

- The folder tree clearly separates core primitives, downloads, market data, universes, and derivatives.
- `US`, `Options`, and universe workflows no longer own ad hoc CSV save loops.
- Price, consensus, fundamentals, and index WRDS sources are visible under `marketdata/`.
- Strategy, DI, Registry, Protocol, and composition-first design remain visible in the code.
- Existing WRDS CLI behavior and tests pass.
