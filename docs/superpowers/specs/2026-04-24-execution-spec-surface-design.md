# Execution Spec Surface Design

Date: 2026-04-24
Status: Draft for review
Owner: Codex + user collaboration

## 1. Problem statement

`shquants` can already execute backtests with:
- custom rebalance schedules passed as a `pd.Series`
- `next_open` and `close` fill semantics in the engine
- custom one-off research scripts that bypass the public CLI

The current gap is not engine capability. The gap is the **official execution surface**.

Today, `backtesting.run` exposes only a narrow CLI contract:
- schedule: `daily | weekly | monthly`
- fill mode: `close | next_open`
- registered strategy execution through a small fixed argument set

That narrow surface creates the wrong external conclusion:
> "The engine cannot do this."

What is actually true is:
> "The default CLI does not expose this path directly."

The design goal is to make advanced execution paths first-class and official so that future workflows can say:
- the simple CLI covers common cases
- presets/specs cover advanced cases
- hooks extend controlled advanced behavior safely

## 2. Core design principle

This change must **not** alter backtest math.

### Hard invariant

If two runs resolve to the same effective execution inputs, they must produce the same:
- equity series
- return series
- quantity path
- turnover path
- summary metrics

This project changes only the **execution surface** and the **audit/provenance layer**.
It does **not** change:
- engine calculations
- rebalance timing semantics
- fill semantics
- strategy formulas
- portfolio math

### Allowed differences

The new path may add only metadata artifacts such as:
- `resolved_execution_spec.json`
- `execution_resolution.json`
- explicit fallback provenance
- preset/spec provenance

Those metadata additions must not affect the computed portfolio path.

## 3. Recommended approach

Three approaches were considered:

1. Expand the existing CLI with many new flags
2. Introduce an `ExecutionSpec` layer and route all advanced execution through it
3. Split advanced execution into a separate dedicated CLI

### Recommendation: approach 2

Use a unified **ExecutionSpec** / **ResolvedExecutionSpec** contract.

Why this is preferred:
- keeps the default CLI simple for common cases
- gives advanced research a formal, official path
- makes presets, spec files, and future hook-based execution share one contract
- avoids option explosion in `backtesting.run`
- keeps complexity in a resolution layer instead of leaking it into engine logic

## 4. High-level architecture

### 4.1 Input surfaces

The system will support three official input surfaces:

1. **Legacy/simple CLI arguments**
   - Example: `uv run python -m backtesting.run --strategy momentum --start ...`
2. **Preset execution**
   - Example: `uv run python -m backtesting.run --preset kospi200_semiannual_floatcap`
3. **Spec execution**
   - Example: `uv run python -m backtesting.run --spec path/to/run_spec.yaml`

### 4.2 Resolution architecture

All three surfaces flow into the same path:

`input surface -> ExecutionSpec -> ResolvedExecutionSpec -> existing runner/engine`

This creates two layers:

#### Resolution layer
Responsible for deciding **what** should run.
- parse CLI/preset/spec input
- apply defaults
- validate the schema
- resolve datasets and fallback policy
- resolve hook IDs from a safe registry
- emit a fully resolved execution contract
- record provenance and fallback metadata

#### Execution layer
Responsible for **how** the run is computed.
- load data
- build weights or plan
- call the existing runner/engine
- write result artifacts

This boundary is important: advanced interface complexity should live in the resolution layer, not in engine math.

## 5. ExecutionSpec contract

`ExecutionSpec` is the user-facing execution contract. It describes one backtest completely enough to be validated and resolved.

### 5.1 Identity
- `name`
- `description`
- `start`
- `end`
- `capital`

### 5.2 Universe and benchmark
- `universe_id`
- optional benchmark overrides
- legacy fields may be supported during transition but the long-term direction is `universe_id`-first

### 5.3 Execution settings
- `fill_mode`: `close | next_open`
- `schedule` definition:
  - simple schedule names
  - custom dates
  - extensible rule-based schedule forms later
- `allow_fractional`
- costs:
  - `fee`
  - `sell_tax`
  - `slippage`

### 5.4 Plan / weight source
- `strategy` for existing registered strategies
- `preset_id` for provenance when a preset created the spec
- `weight_source`, with v1 supporting:
  - `strategy`
  - `dataset`
  - `file`
  - `hook`
- `hook_id` for registry-backed custom generators

### 5.5 Data and fallback policy
- requested datasets
- optional fallback policy declarations
- explicit requested vs resolved weight basis / data basis

### 5.6 Provenance and audit metadata
- `spec_source`: `cli | preset | spec_file`
- `resolved_from`
- `fallbacks_applied`
- `notes`

## 6. ResolvedExecutionSpec contract

`ResolvedExecutionSpec` is the execution-only contract created after validation and resolution.

It must be fully explicit and executable without hidden defaults.

Examples of fields that must be resolved by this stage:
- concrete dataset IDs
- resolved universe choice
- final schedule materialization strategy
- resolved weight source mode
- resolved hook target
- requested weight basis vs actual resolved basis
- explicit fallback records

The execution layer should consume a resolved contract and remain unaware of upstream ambiguity.

## 7. Preset design

### 7.1 Purpose

Presets formalize recurring research patterns that are too specific or too rich for the basic CLI but should still be first-class and officially supported.

Examples:
- `kospi200_semiannual_floatcap`
- future recurring institutional flows or staged portfolio recipes

### 7.2 Registry shape

Introduce a `PresetRegistry` that maps a stable preset ID to a factory that returns an `ExecutionSpec`.

The preset registry should:
- live inside `shquants`
- be version-controlled
- be testable
- be discoverable from the CLI

### 7.3 Why presets matter here

The current semiannual float-cap case already exists as a script. That path should graduate into a preset so it becomes an official execution mode rather than an informal workaround.

## 8. Spec file design

### 8.1 Purpose

Spec files solve the main user experience problem:
- the CLI does not need to expose every advanced execution option directly
- advanced paths remain officially supported through a structured file contract

This allows the system to answer:
> "That is not a basic CLI flag, but it is a supported spec path."

### 8.2 Format

V1 should support JSON and YAML.

Recommended CLI shape:
- `--spec path/to/file.yaml`

### 8.3 Validation requirements

Spec loading must:
- validate required fields
- reject unknown unsafe hook references
- reject incompatible schedule / weight-source combinations
- resolve or reject fallback policies deterministically
- produce user-readable validation errors

## 9. Hook registry design

### 9.1 Safety model

Hooks must be **registry-based only**.

The system must not execute arbitrary Python file paths from the spec.

That means:
- allowed: `hook_id: "kospi200_semiannual_floatcap"`
- not allowed: `python_path: "/tmp/random.py"`

### 9.2 Supported extension points in v1

The registry should be designed so that these categories can exist:
- `signal_builder`
- `plan_builder`
- `weight_provider`
- optional future `schedule_provider`

V1 does not need all extension categories fully implemented, but the contract should make room for them cleanly.

### 9.3 Why registry is the right boundary

Registry-only hooks provide:
- safe execution boundaries
- deterministic reproducibility
- easier review and testing
- better provenance in reports

## 10. Fallback policy design

Fallback must be explicit and resolved before execution begins.

### 10.1 Rule

No silent fallback.

If a run requested `float_market_cap` but only `market_cap` is available, that fact must be recorded before engine execution.

### 10.2 Example resolution metadata

A resolved execution record should be able to say:
- `requested_weight_basis: float_market_cap`
- `resolved_weight_basis: market_cap`
- `fallback_applied: true`
- `fallback_reason: missing qw_mktcap_flt parquet`

### 10.3 Engine boundary

The engine does not need to know about fallbacks.

By the time the engine is invoked, it should receive only the final resolved data/weights/plan inputs. Fallback logic belongs in the resolution layer.

## 11. Runner integration design

### 11.1 Migration target

Introduce a new `run_spec()`-style path in the runner and make the basic CLI normalize into that path.

Longer-term desired flow:
- CLI args -> spec builder -> `run_spec()`
- preset -> spec factory -> `run_spec()`
- spec file -> spec loader -> `run_spec()`

### 11.2 Backward compatibility

The basic CLI should remain available and readable for simple runs.

The user experience should improve like this:
- simple jobs stay simple
- advanced jobs become official instead of ad hoc
- existing simple workflows keep working

## 12. Reporting and artifact changes

The run output should add audit/provenance artifacts without changing portfolio math.

Recommended additions:
- `resolved_execution_spec.json`
- `execution_resolution.json`

These should capture:
- origin (`cli`, `preset`, `spec_file`)
- preset ID if used
- hook ID if used
- resolved datasets
- fallback decisions
- requested vs resolved weight basis

## 13. Verification strategy

The most important test type is **parity testing**.

### 13.1 Required parity checks

1. **Legacy CLI parity**
   - old-style CLI execution vs spec-normalized execution
   - same results

2. **Preset parity**
   - preset execution vs equivalent resolved spec execution
   - same results

3. **Custom schedule parity**
   - existing one-off script behavior vs new preset/spec path
   - same results

4. **Metadata-only difference check**
   - any difference from the previous path should be additional metadata only

### 13.2 Equality targets

For parity runs, compare at minimum:
- equity
- returns
- qty
- turnover
- summary metrics

### 13.3 Regression principle

If the effective execution inputs are unchanged, the result path must be unchanged.

That principle should be enforced both in tests and in code review.

## 14. Implementation sequence

Recommended delivery sequence:

1. Add `ExecutionSpec` / `ResolvedExecutionSpec` models
2. Add spec loader and validation
3. Add preset registry
4. Add runner `run_spec()` path
5. Route the existing CLI through spec normalization
6. Add registry-backed hook support
7. Promote the semiannual float-cap script into a preset/spec-backed official path
8. Add parity and provenance regression tests

## 15. Risks and controls

### Risk 1: accidental math drift
Control:
- parity tests against legacy paths
- explicit invariant in design and code review

### Risk 2: CLI complexity leakage
Control:
- keep advanced behavior in presets/specs
- do not keep growing raw CLI flags for every research case

### Risk 3: unsafe extension behavior
Control:
- registry-backed hooks only
- no arbitrary file execution from spec

### Risk 4: silent data substitution
Control:
- fallback decisions recorded pre-execution
- explicit provenance artifacts in run outputs

## 16. Non-goals

This design does not aim to:
- rewrite engine math
- redesign existing strategy formulas
- replace the simple CLI for common cases
- permit arbitrary user-provided Python execution from spec files

## 17. Team/Ralph execution follow-up

Once this design is approved:

1. create an implementation plan that splits the work into:
   - spec/resolution models
   - preset/spec loading path
   - runner integration
   - tests/parity verification
2. use team execution for parallelizable implementation and test lanes
3. use Ralph as the persistence + final verification loop to ensure:
   - no pending work remains
   - parity is demonstrated
   - no math drift was introduced

## 18. Final decision summary

Chosen direction:
- keep the basic CLI simple
- add official `--preset` and `--spec` execution paths
- use `ExecutionSpec` / `ResolvedExecutionSpec` as the unified contract
- support extensibility through a registry-backed hook model
- require explicit fallback provenance
- preserve existing backtest results for identical resolved inputs
