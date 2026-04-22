# Backtesting and Reporting Stabilization Design

## Goal

Stabilize the `shquants` backtesting and reporting workflow around the local raw CSV/XLSX data already present in this repository, while keeping changes minimal and avoiding broad refactors.

## Approved Scope

### In scope

- `backtesting/`
- root project configuration needed for backtesting/reporting
- local raw datasets under `raw/`
- local derived parquet artifacts under `parquet/` as untracked local outputs
- removal of repo-local Graphify code/tests from `shquants`

### Out of scope

- `analysts/` changes during this phase
- broad architectural refactors
- dashboard/UI work unless a reporting dependency forces it

## Constraints

- Prefer stabilization over redesign.
- Assume the backtesting/reporting code is mostly sound because it was written with TDD.
- Refactor only if inspection shows a concrete need.
- Keep parquet as a local untracked artifact, not committed repo data.

## Root Causes Observed

The current scoped failures fall into a few concrete buckets:

1. missing runtime dependencies required by existing code paths:
   - `scipy`
   - `openpyxl`
2. repo-local Graphify tests that depend on files not owned by this repo
3. expected local parquet artifacts not yet generated from the provided raw CSV/XLSX files
4. reporting test drift where tests instantiate `RollingMetrics` without the now-required `window` field

## Chosen Approach

Use a minimal stabilization-first pass:

1. remove Graphify-related repo-local tests/code paths from `shquants`
2. add only the missing backtesting/reporting dependencies
3. generate local parquet artifacts from the provided raw files
4. rerun only the scoped backtesting/reporting test suite
5. fix only the remaining concrete scoped failures

## Why This Approach

This matches the user's intent:

- preserve the existing code shape
- rely on the supplied raw files
- avoid touching `analysts/`
- avoid speculative refactors

## Verification Strategy

Success for this phase is:

1. scoped Graphify code removed from `shquants`
2. local parquet artifacts generated successfully from `raw/`
3. the scoped backtesting/reporting test suite passes in the root uv environment

The scoped suite excludes:

- `analysts/`
- Graphify-specific tests
- already-removed cross-repo `kis` / `crypto` tests
