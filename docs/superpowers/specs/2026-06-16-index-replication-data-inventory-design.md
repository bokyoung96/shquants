# Index Replication Data Inventory Design

## Goal

Build a full-replication data audit surface before adding more index
replication or backtest code.

The first target is `0167A0` (`SOL AI반도체TOP2플러스`), which tracks
`FI00.WLT.KSS` (`FnGuide AI Semiconductor TOP2 Plus Index`). The design must
also support other FnGuide tracked indices by using a common inventory schema
instead of one-off index notes.

The inventory answers four questions for each index:

1. What data is required by the official methodology?
2. Which required data is already present in this repository?
3. Which data can be derived from present data without changing the official
   meaning?
4. Which data is missing and must be sourced externally before full replication
   can be claimed?

## Motivation

The current KSS replication code can execute bucket selection, target-weight
generation, and validation when a complete `kss_snapshot.json` is supplied.
That is useful infrastructure, but it is not sufficient for full replication.

Full replication must start with data authority. If official theme membership,
sales momentum, constituent snapshots, or target weights are missing, the system
must say so explicitly and must not substitute a proxy backtest while calling it
replication.

## Principles

1. Data inventory comes before implementation.
2. Proxy data and official replication evidence must never share the same
   status label.
3. Each inventory row must identify the methodology requirement it supports.
4. Each availability claim must cite a local file, generated artifact, or
   external data need.
5. A tracked index can move to replication/backtest work only after its required
   data is marked `available` or `derivable`, with official validation data
   separately identified.
6. `sidecar` is out of scope for this work.

## Inventory Status Vocabulary

Use a small fixed vocabulary so multiple indices can be compared.

| Status | Meaning |
| --- | --- |
| `available` | The required data exists locally in a directly usable form. |
| `derivable` | The required data can be created from local data without changing the official definition. |
| `missing` | The required data is not present locally. |
| `external_required` | The required data must come from FnGuide, issuer disclosures, exchange data, or another authority. |
| `methodology_blocked` | The methodology text does not define the data or formula clearly enough to implement. |
| `not_applicable` | The data class is irrelevant for this index methodology. |

`derivable` is intentionally strict. A price momentum value may be derivable
from local prices if the official lookback and treatment are known. A sales
momentum value is not derivable merely because other accounting fields exist;
the official sales definition, reporting lag, and missing-value treatment must
also be known.

## Common Inventory Schema

The machine-readable inventory should be a JSON artifact with this shape:

```json
{
  "schema_version": "1.0",
  "generated_at": "ISO-8601 timestamp",
  "provider": "fnguide",
  "indices": [
    {
      "index_code": "FI00.WLT.KSS",
      "index_name": "FnGuide AI Semiconductor TOP2 Plus Index",
      "tracked_etfs": [
        {"etf_code": "0167A0", "etf_name": "SOL AI반도체TOP2플러스"}
      ],
      "methodology_status": "methodology_verified",
      "replication_readiness": "missing_required_data",
      "requirements": [
        {
          "requirement_id": "kss.selection.sales_momentum",
          "category": "selection_metrics",
          "name": "sales_momentum",
          "required_fields": ["as_of", "security_code", "sales_momentum"],
          "methodology_reference": "selection.buckets.momentum.score.components",
          "status": "external_required",
          "local_evidence": [],
          "external_need": "FnGuide sales momentum source or exact official calculation inputs",
          "notes": "Required for composite momentum score; cannot be replaced with a proxy for full replication."
        }
      ]
    }
  ]
}
```

The Markdown report should summarize the same data in a table optimized for
review:

| Index | ETF(s) | Ready? | Available | Derivable | Missing | External required | Methodology blocked |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `FI00.WLT.KSS` | `0167A0`, `395160` | no | 5 | 2 | 1 | 5 | 0 |

## Required Requirement Categories

Each index inventory should use these categories where applicable.

### Methodology Evidence

| Requirement | Purpose |
| --- | --- |
| `methodology_spec` | Executable methodology rules extracted from provider documents. |
| `methodology_source` | PDF/page URL, version date, and evidence spans. |
| `tracked_etf_mapping` | ETF code/name to index code mapping. |

### Calendar And Rebalance

| Requirement | Purpose |
| --- | --- |
| `rebalance_calendar` | Review date, data cutoff date, weight determination date, and effective date. |
| `trading_calendar` | Business-day alignment. |
| `expiry_calendar` | Required when methodology uses futures/options expiry date `D` and `D+N`. |
| `special_rebalance_events` | Non-regular additions, deletions, or cap resets. |

### Security Master And Eligibility

| Requirement | Purpose |
| --- | --- |
| `security_master` | Security code, name, market, stock type, listing status, listing date. |
| `eligibility_flags` | Preferred share, SPAC, REIT, ETF/ETN, suspended, managed, delisted, newly listed. |
| `corporate_actions` | Share count changes, splits, mergers, symbol changes, and other index maintenance inputs. |

### Market And Liquidity Data

| Requirement | Purpose |
| --- | --- |
| `price_snapshot` | Prices used for ranking, momentum, weights, and validation. |
| `market_cap_snapshot` | Market capitalization ranking inputs. |
| `float_market_cap_snapshot` | Float-market-cap ranking and weighting inputs. |
| `trading_value_snapshot` | Average trading value and liquidity filters. |
| `free_float_snapshot` | Free-float ratio if float market cap is not already authoritative. |
| `listed_shares_snapshot` | Listed shares if market cap must be reconstructed. |

### Classification And Theme Membership

| Requirement | Purpose |
| --- | --- |
| `sector_classification` | FICS/WICS/GICS sector or industry membership. |
| `theme_membership` | Provider-defined theme membership such as AI semiconductor universe. |
| `theme_membership_effective_dates` | Historical membership snapshots by review date. |

### Selection Metrics

| Requirement | Purpose |
| --- | --- |
| `price_momentum` | Price momentum component when methodology defines it. |
| `sales_momentum` | Sales momentum component when methodology defines it. |
| `composite_score` | Official score used for ranking. |
| `tie_breakers` | Deterministic ranking rules for equal scores. |
| `official_bucket_assignments` | Optional but valuable validation of bucket selection. |

### Weighting And Validation

| Requirement | Purpose |
| --- | --- |
| `weighting_inputs` | Data used by the weighting engine: float market cap, fixed bucket weights, caps. |
| `official_target_weights` | Primary full-replication validation evidence. |
| `issuer_holdings_snapshot` | Secondary validation evidence only. |
| `index_level_series` | Optional index-level validation once constituent replication is available. |

## KSS/SOL First Inventory

`SOL AI반도체TOP2플러스` maps to `FI00.WLT.KSS`. The same index is also
tracked by `KODEX AI반도체TOP2플러스`, so the inventory should list both ETF
mappings when local artifacts identify both.

### KSS Methodology Requirements

The current methodology spec states:

- total constituents: 10
- `top2`: top 2 by market cap, fixed 25% each
- `momentum`: 4 names after excluding `top2`, ranked by composite score
- `market_cap_fill`: 4 names after excluding prior buckets, ranked by market cap
- residual buckets weighted by float market cap
- residual total weight: 50%
- residual security cap: 15%
- regular implementation months: 1, 4, 7, 10

### KSS Expected Inventory Findings

The first audit should classify local data conservatively:

| Requirement | Expected status | Evidence or gap |
| --- | --- | --- |
| `methodology_spec` | `available` | `etfs/output/extractions/fnguide/methodology_specs.json` |
| `tracked_etf_mapping` | `available` | `0167A0 SOL AI반도체TOP2플러스`, `395160 KODEX AI반도체TOP2플러스` in local outputs |
| `price_snapshot` | `available` | `parquet/qw_adj_c.parquet` |
| `float_market_cap_snapshot` | `available` | `parquet/qw_mktcap_flt.parquet` |
| `sector_classification` | `available` | `parquet/qw_wics_sec_big.parquet`, `parquet/qw_wi_sec_26_big.parquet`, mapping files |
| `price_momentum` | `derivable` | Local prices exist, but official lookback and calculation treatment must be confirmed from methodology evidence. |
| `trading_calendar` | `derivable` | Market data index can provide trading days; official holiday calendar source should be recorded if used. |
| `rebalance_calendar` | `derivable` or `external_required` | Rule says quarterly implementation months and `D+2`; exact review/effective dates need expiry calendar. |
| `theme_membership` | `external_required` | Provider-defined AI semiconductor universe cannot be replaced by broad WICS/FICS proxy for full replication. |
| `sales_momentum` | `external_required` | Official sales momentum input/formula and reporting lag are required. |
| `composite_score` | `external_required` | Depends on official sales momentum and price momentum definitions. |
| `official_bucket_assignments` | `external_required` | Needed to validate selection before weights. |
| `official_target_weights` | `external_required` | Primary full-replication validation evidence. |
| `issuer_holdings_snapshot` | `available` | Local `validation_A0167A0.xlsx` / generated validation fixtures, but secondary evidence only. |
| `corporate_actions` | `missing` | Required for long historical exact replication and index maintenance. |

The KSS inventory should therefore report `replication_readiness` as
`missing_required_data`, not `ready_for_full_replication`.

## Readiness Classification

Each index should receive one readiness status:

| Readiness | Meaning |
| --- | --- |
| `ready_for_full_replication` | All required methodology, selection, weighting, and official validation data are available or strictly derivable. |
| `ready_for_unvalidated_calculation` | Selection and weighting inputs are available, but official validation data is missing. |
| `missing_required_data` | One or more required official input datasets are missing. |
| `methodology_blocked` | The methodology is not clear enough to define required data or formulas. |
| `unsupported_methodology` | The methodology is clear, but current engine primitives cannot execute it. |

KSS/SOL should remain `missing_required_data` until official theme membership,
sales momentum/composite score, and official target weights are present.

## Expansion To Other Indices

The inventory generator should iterate over FnGuide methodology specs and create
the same audit surface for every index, not only KSS.

The first implementation should support these methodology families:

| Family | Example | Inventory behavior |
| --- | --- | --- |
| `top2_plus` | `FI00.WLT.KSS` | Require bucket selection metrics, float market cap, official target weights. |
| `fixed_plus_residual` | `FI00.WLT.HBM`, `FI00.WLT.NHD`, `FI00.WLT.SCT` | Require leader/bucket selection evidence, residual weighting input, caps. |
| `float_market_cap_weighted` | `FI00.WLT.NHM`, `FI00.WLT.NHG` | Require constituent universe, float market cap, cap rules, official targets. |
| `metric_weighted` | `FI00.WLT.NDV`, `FI00.WLT.REP` | Require metric definition and metric snapshots, plus official targets. |
| `equal_weighted` | `FI00.WLT.HMZ` | Require official constituent universe and rebalance calendar. |

Unsupported or unclear methods should still emit inventory rows. They should
not disappear from reporting merely because the engine cannot calculate them.

## Output Contract

Add a provider-level inventory output under the replication tree:

```text
etfs/output/replication/fnguide/data_inventory.json
etfs/output/replication/fnguide/data_inventory.md
```

Add an optional focused KSS/SOL view:

```text
etfs/output/replication/fnguide/kss_data_inventory.json
etfs/output/replication/fnguide/kss_data_inventory.md
```

The provider-level Markdown should include:

- summary counts by readiness status
- summary counts by requirement status
- one table row per index
- a detailed section for each index with blocking gaps
- an explicit list of indices that should not proceed to backtesting

The KSS/SOL Markdown should include:

- tracked ETF mappings
- methodology primitive summary
- available local data evidence
- external required data list
- why issuer holdings are secondary validation only
- next data acquisition actions

## Testing Strategy

Tests should be written before implementation.

Unit tests:

- KSS/SOL inventory contains `0167A0` and `FI00.WLT.KSS`.
- KSS classifies local price and float-market-cap data as `available`.
- KSS classifies theme membership, sales momentum, composite score, and official
  target weights as `external_required`.
- KSS readiness is `missing_required_data`.
- Issuer holdings do not upgrade readiness to full replication.

Provider inventory tests:

- The generator emits one inventory item for every methodology spec.
- Unsupported or methodology-blocked indices are still reported.
- Readiness counts match item statuses.
- Markdown output includes KSS gaps and the provider summary table.

Regression tests:

- A default-path official validation artifact must not make KSS ready if the
  inventory for the current run does not have the required current-run data.
- Proxy classifications or proxy scores must be labelled separately and must
  not satisfy official full-replication requirements.

## Acceptance Criteria

This work is complete when:

- A machine-readable inventory schema exists for FnGuide index replication data.
- KSS/SOL has a concrete data inventory with local evidence and missing external
  requirements.
- The inventory explicitly says KSS/SOL is not ready for full replication until
  official theme membership, sales/composite score, and official target weights
  are available.
- The provider-level report lists all FnGuide methodology specs and groups them
  by readiness.
- Existing KSS replication code remains available but is downstream of the
  inventory readiness gate.
- No `sidecar` files are modified.

## Risks

- Some local data may look similar to official methodology inputs but differ in
  authority or timing. The inventory must prefer `external_required` over
  optimistic derivation when official meaning is uncertain.
- FnGuide methodology PDFs may omit implementation details such as exact score
  calculation or tie-breakers. Those cases should become `methodology_blocked`
  or `external_required`, not guessed.
- Expanding to all indices can become noisy. The first provider report should
  prioritize clear blocker tables over detailed calculation code.

## Self-Review

- The design does not claim full replication is possible from proxy data.
- The design handles KSS/SOL first while keeping the schema reusable for other
  FnGuide indices.
- The status vocabulary distinguishes local availability, strict derivation,
  missing data, external authority needs, and methodology blockers.
- The implementation path is intentionally audit-first; backtesting and target
  generation come only after inventory readiness is proven.
