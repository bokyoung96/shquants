# KSS Index Replication Design

## Goal

Replicate an ETF tracking index end to end, starting with
`FI00.WLT.KSS` (`FnGuide AI Semiconductor TOP2 Plus Index`) as the tracer
bullet.

The desired result is not only a target-weight formula. The system must prove
the full chain:

```text
methodology evidence
-> rebalance date
-> eligible universe
-> bucket selection
-> target weights
-> official target or ETF holdings validation
```

The current FnGuide engine already calculates target weights when selected
constituents and buckets are supplied explicitly. This design adds the missing
selection and validation layers so target weights can be generated from
methodology data, not hand-filled inputs.

## Current Evidence

Current `etfs` artifacts show:

- FnGuide methodology specs: 59.
- Engine-ready specs: 12.
- Target-weight formula smoke tests passing for engine-ready specs: 12.
- Full methodology replications proven: 0.

The existing limitation is deliberate: `methodology_engine.py` receives selected
constituents or `constituents_by_bucket` as explicit inputs. It verifies
constituent counts, duplicate security codes, weight sums, cap handling, and
iterative pro-rata redistribution, but it does not construct the issuer
universe or decide which names belong in each methodology bucket.

## Principles

1. Full replication starts before weighting. A target-weight engine is correct
   only if the selected names and buckets are also reproduced.
2. Every calculation must be tied to an `as_of` or rebalance date.
3. Methodology evidence, input data, selected buckets, and final weights should
   be separately inspectable.
4. Official target weights are the preferred validation source. ETF holdings are
   fallback validation evidence, not methodology inputs.
5. The KSS implementation should create reusable interfaces for other FnGuide
   engine-ready specs, but it should not generalize beyond proven data needs.

## Non-Goals

- Do not claim all 59 FnGuide specs are replicated in the first implementation.
- Do not promote draft methodology specs without PDF evidence review.
- Do not use ETF holdings to choose constituents.
- Do not infer hidden FnGuide classifications when the required classification
  data is missing.
- Do not touch unrelated packages or sidecar code.

## Required Data

### Methodology Source Data

| Dataset | Required fields | Purpose |
| --- | --- | --- |
| `methodology_spec` | `index_code`, `index_name`, status, total constituents, bucket specs, weighting rules | Source of executable rules |
| `methodology_evidence` | PDF path or URL, evidence spans, version date | Audit trail for each rule |
| `etf_index_map` | ETF code, ETF name, tracked index code, provider | Connect ETF holdings validation to the index |

### Calendar Data

| Dataset | Required fields | Purpose |
| --- | --- | --- |
| `rebalance_calendar` | `index_code`, `review_date`, `data_cutoff_date`, `effective_date` | Drives methodology snapshots |
| `krx_trading_calendar` | date, business-day flag | Aligns market data and effective dates |
| `futures_options_expiry_calendar` | expiry date, contract type | Supports FnGuide schedules that reference expiries |
| `month_end_business_day_calendar` | month, last business day | Supports month-end based methodologies |

### Security And Eligibility Data

| Dataset | Required fields | Purpose |
| --- | --- | --- |
| `security_master` | `as_of`, `security_code`, name, market, listing status, stock type | Defines candidate securities |
| `eligibility_flags` | `as_of`, `security_code`, preferred stock, SPAC, REIT, suspended, managed, newly listed, delisted | Applies universe exclusions |
| `corporate_actions` | event date, effective date, action type, share adjustment | Handles special events and index continuity |

### Market Data

| Dataset | Required fields | Purpose |
| --- | --- | --- |
| `price_snapshot` | `as_of`, `security_code`, adjusted close | Market-cap and momentum inputs |
| `share_snapshot` | `as_of`, `security_code`, listed shares | Market-cap inputs |
| `free_float_snapshot` | `as_of`, `security_code`, free-float ratio | Float market cap |
| `market_snapshot` | `as_of`, `security_code`, market cap, float market cap | Ranking and weighting base |
| `liquidity_snapshot` | `as_of`, `security_code`, average trading value | Liquidity screening if required by methodology |

### Classification And Selection Metrics

| Dataset | Required fields | Purpose |
| --- | --- | --- |
| `classification_snapshot` | `as_of`, `security_code`, FICS sector, FICS industry, semiconductor/theme flags | KSS semiconductor universe |
| `theme_membership` | `as_of`, `index_code`, `security_code`, source, membership flag | Provider/theme membership when FICS is insufficient |
| `selection_metrics` | `as_of`, `security_code`, sales momentum, price momentum, composite momentum score | KSS momentum bucket ranking |
| `ranking_inputs` | `as_of`, `security_code`, metric name, metric value | Generic storage for later FnGuide rankings |

### Validation Data

| Dataset | Required fields | Purpose |
| --- | --- | --- |
| `official_target_weights` | `index_code`, `effective_date`, `security_code`, official target weight | Primary replication check |
| `etf_holdings_snapshot` | ETF code, holdings date, security code, holding weight | Secondary validation evidence |
| `index_levels` | `index_code`, date, level, divisor when available | Later index-level replication checks |

## KSS Methodology Flow

KSS should be implemented as a first-class methodology recipe that produces the
same input shape already accepted by `calculate_top2_plus_target_weights`.

```text
rebalance request
-> load KSS spec
-> resolve calendar dates
-> build eligible universe
-> apply semiconductor/theme classification
-> rank by float market cap
-> select top2 bucket
-> exclude top2
-> rank remaining names by composite momentum score
-> select momentum bucket of 4
-> exclude prior buckets
-> select market_cap_fill bucket of 4 by float market cap
-> pass buckets to target-weight engine
-> compare calculated weights to validation data
```

### KSS Selection Rules

The initial executable KSS rule should be:

1. Start from eligible Korean common-stock universe at the data cutoff date.
2. Keep names passing the KSS semiconductor/theme classification.
3. Compute or read float market cap.
4. Select `top2`: top 2 securities by float market cap.
5. Remove `top2` from the candidate pool.
6. Select `momentum`: top 4 by composite momentum score.
7. Remove `top2` and `momentum` from the candidate pool.
8. Select `market_cap_fill`: top 4 by float market cap.
9. Reject the run if any bucket has too few constituents.
10. Reject duplicate securities across buckets.

Tie-breakers must be explicit. The default deterministic tie-breaker should be:

```text
primary rank metric
-> float_market_cap descending
-> security_code ascending
```

If the official methodology specifies different tie-breakers, the spec evidence
must override this default.

### KSS Weighting Rules

The weighting layer should reuse the current target-weight implementation:

- `top2`: 2 names, 25% each.
- residual buckets: `momentum` and `market_cap_fill`.
- residual total weight: 50%.
- residual base: float market cap.
- residual security cap: 15%.
- redistribution: iterative pro-rata among uncapped residual names.
- required final checks:
  - 10 constituents.
  - target weights sum to 1.0 within tolerance.
  - no duplicate security codes.
  - no residual name exceeds 15%.

## Proposed Architecture

```text
etfs/fnguide/replication_data.py
  -> dataclasses or typed dictionaries for KSS input snapshots

etfs/fnguide/selection.py
  -> eligible universe filters
  -> KSS bucket selector

etfs/fnguide/replication.py
  -> orchestration from index_code/as_of to selected buckets and target weights

etfs/fnguide/validation.py
  -> compare calculated target weights with official targets or holdings

etfs/output/replication/fnguide/
  -> input requirement reports
  -> selected bucket outputs
  -> target weights
  -> validation diffs
```

The existing `methodology_engine.py` should remain the weighting authority. KSS
replication should call into it instead of duplicating cap and redistribution
logic.

## Output Contract

The KSS replication run should emit inspectable artifacts:

```text
selected_buckets.json
target_weights.json
replication_validation.json
replication_validation.md
```

`selected_buckets.json` should include:

- `index_code`
- `as_of`
- `data_cutoff_date`
- bucket name
- rank metric
- selected security codes
- metric values used for ranking
- exclusion reason for prior-bucket removals when useful

`target_weights.json` should include:

- `index_code`
- `effective_date`
- `methodology`
- target weights
- checks and metrics copied from the engine result

`replication_validation.json` should include:

- validation source type: `official_target_weights`, `etf_holdings_snapshot`, or
  `missing`
- missing securities
- extra securities
- per-security weight differences
- max absolute weight difference
- total absolute weight difference
- pass/fail status by configured tolerance

## Testing Strategy

Tests should be added before implementation changes.

Unit tests:

- KSS selector builds `top2`, `momentum`, and `market_cap_fill` buckets from a
  fixture universe.
- Prior bucket exclusions prevent duplicate securities.
- Selector fails clearly when a required metric is missing.
- Selector fails clearly when a bucket cannot be filled.
- Tie-breakers produce deterministic output.

Weighting integration tests:

- KSS selected buckets feed the existing `top2_plus` target-weight function.
- Top2 weights are exactly 25% each.
- Residual weights cap at 15% and redistribute excess.
- Final weight sum is 1.0.

Validation tests:

- Exact official target match passes.
- Missing, extra, and weight-drifted securities are reported.
- ETF holdings validation is marked as secondary evidence.
- Missing official and holdings validation prevents a full replication claim.

Pipeline tests:

- FnGuide pipeline writes KSS replication requirement artifacts.
- Pipeline skips calculation when required KSS source data is absent and reports
  the missing datasets.
- Pipeline produces target weights when a complete fixture dataset is present.

## Acceptance Criteria

The first KSS implementation is complete when:

- Required KSS data fields are documented and machine-readable.
- A fixture data snapshot can produce KSS selected buckets.
- Selected buckets feed the existing target-weight engine without manual JSON
  editing.
- The resulting target weights satisfy KSS count, cap, duplicate, and sum checks.
- A validation report distinguishes official target validation from ETF holdings
  validation.
- The replication report no longer treats KSS as only a target-weight smoke test;
  it reports KSS full replication status based on selection plus validation
  evidence.
- Existing `tests/etfs` continue to pass.

## Expansion Path

After KSS works, add other engine-ready specs by implementing their selection
recipes against the same data interfaces:

| Index | Reason to add after KSS |
| --- | --- |
| `FI00.WLT.HBM` | Similar fixed top bucket plus residual structure |
| `FI00.WLT.DSA` | Multiple fixed leader buckets |
| `FI00.WLT.NHD` | Dividend/security bucket plus float-cap residual |
| `FI00.WLT.SCT` | TOP3 plus structure with a tighter residual cap |
| `FI00.WLT.NDV` | Metric-weighted dividend index |
| `FI00.WLT.REP` | Metric-weighted keyword score index |

Draft specs should remain blocked until methodology evidence resolves selection
count, cap scope, and bucket rules.

## Risks

- FnGuide theme membership or momentum definitions may not be reconstructible
  from public data. Mitigation: make missing provider data explicit and do not
  claim full replication without it.
- ETF holdings may differ from official index targets because of cash,
  execution timing, creation/redemption activity, or tracking error. Mitigation:
  mark holdings as secondary validation evidence.
- Over-generalizing from KSS can hide method-specific rules. Mitigation: keep
  KSS as a recipe and extract shared interfaces only after at least two recipes
  use them.

## Self-Review

- No placeholders or open `TBD` items remain.
- Scope is intentionally limited to KSS as the first full-replication tracer
  bullet while preserving the larger objective of all tracked indices.
- Current target-weight logic remains the weighting authority; this design adds
  the missing data, selection, and validation layers.
- The design explicitly states that full replication cannot be claimed without
  selection evidence and official or secondary validation evidence.
