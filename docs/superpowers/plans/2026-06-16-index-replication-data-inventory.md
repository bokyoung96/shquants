# Index Replication Data Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an audit-first FnGuide index replication data inventory that proves which indices have enough official data for full replication and which are blocked by missing data.

**Architecture:** Add a new `etfs.fnguide.data_inventory` module that reads existing methodology specs and local artifact paths, emits provider-wide and KSS/SOL-focused JSON/Markdown reports, and integrates those reports into the offline pipeline before any further backtesting work. The inventory is deliberately conservative: local proxy data can be reported, but it cannot satisfy official full-replication requirements.

**Tech Stack:** Python 3, stdlib JSON/path handling, existing `etfs.paths`, existing FnGuide methodology specs, pytest, ruff.

---

## File Structure

Create:

- `etfs/fnguide/data_inventory.py`: inventory status vocabulary, requirement builders, KSS/SOL focused audit, provider-wide audit, JSON/Markdown writers, CLI.
- `tests/etfs/test_fnguide_data_inventory.py`: unit and writer tests for KSS/SOL and provider-level inventory.

Modify:

- `etfs/paths.py`: add data-inventory output path constants.
- `tests/etfs/test_output_paths.py`: cover data-inventory output constants and CLI defaults.
- `etfs/fnguide/pipeline.py`: write inventory artifacts and include them in the offline pipeline manifest.
- `tests/etfs/test_fnguide_pipeline.py`: cover pipeline inventory outputs.
- `etfs/README.md`: document the data inventory artifacts and the inventory-before-backtesting gate.

Do not touch `sidecar` or `tests/sidecar`.

---

## Task 1: Add Data Inventory Output Paths

**Files:**
- Modify: `etfs/paths.py`
- Modify: `tests/etfs/test_output_paths.py`

- [ ] **Step 1: Write the failing path test**

Append these assertions to `test_fnguide_replication_paths_are_grouped_under_replication_output` in `tests/etfs/test_output_paths.py`:

```python
    assert paths.FNGUIDE_DATA_INVENTORY_JSON.as_posix() == "etfs/output/replication/fnguide/data_inventory.json"
    assert paths.FNGUIDE_DATA_INVENTORY_MD.as_posix() == "etfs/output/replication/fnguide/data_inventory.md"
    assert paths.FNGUIDE_KSS_DATA_INVENTORY_JSON.as_posix() == "etfs/output/replication/fnguide/kss_data_inventory.json"
    assert paths.FNGUIDE_KSS_DATA_INVENTORY_MD.as_posix() == "etfs/output/replication/fnguide/kss_data_inventory.md"
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
python -m pytest tests/etfs/test_output_paths.py::test_fnguide_replication_paths_are_grouped_under_replication_output -q
```

Expected: FAIL with `AttributeError` for `FNGUIDE_DATA_INVENTORY_JSON`.

- [ ] **Step 3: Add path constants**

Add these constants to `etfs/paths.py` after the existing KSS replication constants:

```python
FNGUIDE_DATA_INVENTORY_JSON = FNGUIDE_REPLICATION_OUTPUT_DIR / "data_inventory.json"
FNGUIDE_DATA_INVENTORY_MD = FNGUIDE_REPLICATION_OUTPUT_DIR / "data_inventory.md"
FNGUIDE_KSS_DATA_INVENTORY_JSON = FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_data_inventory.json"
FNGUIDE_KSS_DATA_INVENTORY_MD = FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_data_inventory.md"
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```powershell
python -m pytest tests/etfs/test_output_paths.py::test_fnguide_replication_paths_are_grouped_under_replication_output -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add etfs/paths.py tests/etfs/test_output_paths.py
git commit -m "Add FnGuide data inventory artifact paths" -m "Full index replication needs provider-wide and KSS-focused audit outputs before any target-weight or backtest work can claim readiness." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: python -m pytest tests/etfs/test_output_paths.py::test_fnguide_replication_paths_are_grouped_under_replication_output -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

---

## Task 2: Add KSS/SOL Data Inventory Core

**Files:**
- Create: `etfs/fnguide/data_inventory.py`
- Create: `tests/etfs/test_fnguide_data_inventory.py`

- [ ] **Step 1: Write failing KSS inventory tests**

Create `tests/etfs/test_fnguide_data_inventory.py` with:

```python
import json
from pathlib import Path

from etfs.fnguide.data_inventory import (
    build_fnguide_data_inventory,
    build_kss_data_inventory,
)


def _kss_spec() -> dict[str, object]:
    return {
        "indices": [
            {
                "index_code": "FI00.WLT.KSS",
                "index_name": "FnGuide AI Semiconductor TOP2 Plus Index",
                "status": "methodology_verified",
                "etfs": [
                    {"etf_code": "395160", "etf_name": "KODEX AI반도체TOP2플러스"},
                    {"etf_code": "0167A0", "etf_name": "SOL AI반도체TOP2플러스"},
                ],
                "selection": {
                    "total_constituents": 10,
                    "buckets": [
                        {"name": "top2", "count": 2},
                        {"name": "momentum", "count": 4},
                        {"name": "market_cap_fill", "count": 4},
                    ],
                },
                "weighting": {
                    "base": "float_market_cap_weighted",
                    "residual": {
                        "applies_to_buckets": ["momentum", "market_cap_fill"],
                        "total_weight": 0.5,
                        "base": "float_market_cap",
                        "cap": 0.15,
                        "redistribution": "iterative_pro_rata",
                    },
                },
                "rebalance": {"frequency": "quarterly", "implementation_months": [1, 4, 7, 10]},
            }
        ]
    }


def _write_specs(tmp_path: Path) -> Path:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(json.dumps(_kss_spec(), ensure_ascii=False), encoding="utf-8")
    return specs_path


def test_build_kss_data_inventory_marks_official_gaps_without_upgrading_holdings(tmp_path: Path) -> None:
    specs_path = _write_specs(tmp_path)
    price_path = tmp_path / "qw_adj_c.parquet"
    float_cap_path = tmp_path / "qw_mktcap_flt.parquet"
    sector_path = tmp_path / "qw_wics_sec_big.parquet"
    holdings_path = tmp_path / "validation_A0167A0.xlsx"
    for path in [price_path, float_cap_path, sector_path, holdings_path]:
        path.write_text("", encoding="utf-8")

    inventory = build_kss_data_inventory(
        specs_path=specs_path,
        local_paths={
            "price_snapshot": price_path,
            "float_market_cap_snapshot": float_cap_path,
            "sector_classification": sector_path,
            "issuer_holdings_snapshot": holdings_path,
        },
    )

    assert inventory["index_code"] == "FI00.WLT.KSS"
    assert inventory["replication_readiness"] == "missing_required_data"
    assert {item["etf_code"] for item in inventory["tracked_etfs"]} == {"0167A0", "395160"}

    by_name = {item["name"]: item for item in inventory["requirements"]}
    assert by_name["price_snapshot"]["status"] == "available"
    assert by_name["float_market_cap_snapshot"]["status"] == "available"
    assert by_name["issuer_holdings_snapshot"]["status"] == "available"
    assert by_name["issuer_holdings_snapshot"]["satisfies_full_replication"] is False
    assert by_name["theme_membership"]["status"] == "external_required"
    assert by_name["sales_momentum"]["status"] == "external_required"
    assert by_name["composite_score"]["status"] == "external_required"
    assert by_name["official_target_weights"]["status"] == "external_required"


def test_build_fnguide_data_inventory_includes_every_spec_and_counts_readiness(tmp_path: Path) -> None:
    specs_path = _write_specs(tmp_path)

    inventory = build_fnguide_data_inventory(specs_path=specs_path, local_paths={})

    assert inventory["provider"] == "fnguide"
    assert inventory["counts"]["indices"] == 1
    assert inventory["counts"]["by_readiness"] == {"missing_required_data": 1}
    assert inventory["indices"][0]["index_code"] == "FI00.WLT.KSS"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/etfs/test_fnguide_data_inventory.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'etfs.fnguide.data_inventory'`.

- [ ] **Step 3: Add the minimal data inventory module**

Create `etfs/fnguide/data_inventory.py`:

```python
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from etfs import paths
from etfs.fnguide.replication_data import KSS_INDEX_CODE, KSS_INDEX_NAME


KSS_REQUIRED_REQUIREMENTS = [
    {
        "requirement_id": "kss.methodology_spec",
        "category": "methodology_evidence",
        "name": "methodology_spec",
        "required_fields": ["index_code", "index_name", "selection", "weighting", "status"],
        "methodology_reference": "methodology_specs.indices[]",
        "official_required": True,
    },
    {
        "requirement_id": "kss.tracked_etf_mapping",
        "category": "methodology_evidence",
        "name": "tracked_etf_mapping",
        "required_fields": ["etf_code", "etf_name", "index_code"],
        "methodology_reference": "methodology_specs.indices[].etfs",
        "official_required": True,
    },
    {
        "requirement_id": "kss.market.price_snapshot",
        "category": "market_data",
        "name": "price_snapshot",
        "required_fields": ["as_of", "security_code", "adjusted_close"],
        "methodology_reference": "selection.buckets.momentum.score.components.price_momentum",
        "official_required": True,
    },
    {
        "requirement_id": "kss.market.float_market_cap_snapshot",
        "category": "market_data",
        "name": "float_market_cap_snapshot",
        "required_fields": ["as_of", "security_code", "float_market_cap"],
        "methodology_reference": "selection.buckets.top2.rank and weighting.residual.base",
        "official_required": True,
    },
    {
        "requirement_id": "kss.classification.sector_classification",
        "category": "classification",
        "name": "sector_classification",
        "required_fields": ["as_of", "security_code", "sector_or_industry"],
        "methodology_reference": "selection.buckets.top2.universe_filter",
        "official_required": True,
    },
    {
        "requirement_id": "kss.classification.theme_membership",
        "category": "classification",
        "name": "theme_membership",
        "required_fields": ["as_of", "index_code", "security_code", "is_member"],
        "methodology_reference": "KSS semiconductor/theme universe",
        "official_required": True,
    },
    {
        "requirement_id": "kss.selection.price_momentum",
        "category": "selection_metrics",
        "name": "price_momentum",
        "required_fields": ["as_of", "security_code", "price_momentum"],
        "methodology_reference": "selection.buckets.momentum.score.components.price_momentum",
        "official_required": True,
    },
    {
        "requirement_id": "kss.selection.sales_momentum",
        "category": "selection_metrics",
        "name": "sales_momentum",
        "required_fields": ["as_of", "security_code", "sales_momentum"],
        "methodology_reference": "selection.buckets.momentum.score.components.sales_momentum",
        "official_required": True,
    },
    {
        "requirement_id": "kss.selection.composite_score",
        "category": "selection_metrics",
        "name": "composite_score",
        "required_fields": ["as_of", "security_code", "composite_score"],
        "methodology_reference": "selection.buckets.momentum.rank",
        "official_required": True,
    },
    {
        "requirement_id": "kss.validation.official_bucket_assignments",
        "category": "validation",
        "name": "official_bucket_assignments",
        "required_fields": ["index_code", "effective_date", "bucket", "security_code"],
        "methodology_reference": "selection validation",
        "official_required": True,
    },
    {
        "requirement_id": "kss.validation.official_target_weights",
        "category": "validation",
        "name": "official_target_weights",
        "required_fields": ["index_code", "effective_date", "security_code", "official_weight"],
        "methodology_reference": "target weight validation",
        "official_required": True,
    },
    {
        "requirement_id": "kss.validation.issuer_holdings_snapshot",
        "category": "validation",
        "name": "issuer_holdings_snapshot",
        "required_fields": ["etf_code", "as_of", "security_code", "holding_weight"],
        "methodology_reference": "secondary ETF holdings validation",
        "official_required": False,
    },
    {
        "requirement_id": "kss.maintenance.corporate_actions",
        "category": "maintenance",
        "name": "corporate_actions",
        "required_fields": ["event_date", "effective_date", "security_code", "action_type"],
        "methodology_reference": "index maintenance and long-run exact replication",
        "official_required": True,
    },
]


def build_kss_data_inventory(
    *,
    specs_path: Path = paths.FNGUIDE_METHODOLOGY_SPECS_JSON,
    local_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    local_paths = dict(local_paths or _default_local_paths())
    spec = _load_index_spec(specs_path, KSS_INDEX_CODE)
    requirements = [_requirement_row(item, local_paths) for item in KSS_REQUIRED_REQUIREMENTS]
    readiness = _readiness(requirements)
    return {
        "index_code": KSS_INDEX_CODE,
        "index_name": str(spec.get("index_name", KSS_INDEX_NAME)),
        "tracked_etfs": _tracked_etfs(spec),
        "methodology_status": str(spec.get("status", "")),
        "replication_readiness": readiness,
        "methodology_summary": {
            "total_constituents": _selection_total(spec),
            "buckets": _selection_buckets(spec),
            "weighting": spec.get("weighting", {}),
            "rebalance": spec.get("rebalance", {}),
        },
        "requirements": requirements,
    }


def build_fnguide_data_inventory(
    *,
    specs_path: Path = paths.FNGUIDE_METHODOLOGY_SPECS_JSON,
    local_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    specs = [item for item in payload.get("indices", []) if isinstance(item, Mapping)]
    indices = []
    for spec in specs:
        if spec.get("index_code") == KSS_INDEX_CODE:
            indices.append(build_kss_data_inventory(specs_path=specs_path, local_paths=local_paths))
        else:
            indices.append(_generic_index_inventory(spec))
    readiness_counts = Counter(str(item["replication_readiness"]) for item in indices)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "fnguide",
        "counts": {
            "indices": len(indices),
            "by_readiness": dict(sorted(readiness_counts.items())),
        },
        "indices": indices,
    }


def _default_local_paths() -> dict[str, Path]:
    return {
        "price_snapshot": Path("parquet/qw_adj_c.parquet"),
        "float_market_cap_snapshot": Path("parquet/qw_mktcap_flt.parquet"),
        "sector_classification": Path("parquet/qw_wics_sec_big.parquet"),
        "issuer_holdings_snapshot": Path("etfs/validation_A0167A0.xlsx"),
    }


def _load_index_spec(specs_path: Path, index_code: str) -> Mapping[str, object]:
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    for item in payload.get("indices", []):
        if isinstance(item, Mapping) and item.get("index_code") == index_code:
            return item
    raise ValueError(f"{index_code} not found in {specs_path}")


def _requirement_row(template: Mapping[str, object], local_paths: Mapping[str, Path]) -> dict[str, object]:
    name = str(template["name"])
    evidence_path = local_paths.get(name)
    if name in {"theme_membership", "sales_momentum", "composite_score", "official_bucket_assignments", "official_target_weights"}:
        status = "external_required"
    elif name == "corporate_actions":
        status = "missing"
    elif name == "price_momentum":
        status = "derivable" if local_paths.get("price_snapshot", Path()).exists() else "missing"
    elif evidence_path is not None and evidence_path.exists():
        status = "available"
    elif name in {"methodology_spec", "tracked_etf_mapping"}:
        status = "available"
    else:
        status = "missing"
    return {
        **dict(template),
        "status": status,
        "local_evidence": [evidence_path.as_posix()] if evidence_path is not None and evidence_path.exists() else [],
        "external_need": _external_need(name, status),
        "satisfies_full_replication": bool(template.get("official_required")) and status in {"available", "derivable"},
    }


def _external_need(name: str, status: str) -> str:
    if status != "external_required":
        return ""
    return {
        "theme_membership": "FnGuide AI semiconductor universe membership by review date",
        "sales_momentum": "FnGuide sales momentum source, formula, lag, and historical snapshots",
        "composite_score": "Official composite score or all official component inputs",
        "official_bucket_assignments": "Official rebalance constituent bucket assignments",
        "official_target_weights": "Official target weights by effective date",
    }[name]


def _readiness(requirements: list[dict[str, object]]) -> str:
    if any(item["status"] in {"external_required", "missing", "methodology_blocked"} and item.get("official_required") for item in requirements):
        return "missing_required_data"
    if any(item["status"] == "unsupported_methodology" for item in requirements):
        return "unsupported_methodology"
    return "ready_for_full_replication"


def _tracked_etfs(spec: Mapping[str, object]) -> list[dict[str, str]]:
    etfs = spec.get("etfs") or []
    if not isinstance(etfs, list):
        return []
    return [
        {"etf_code": str(item.get("etf_code", "")), "etf_name": str(item.get("etf_name", ""))}
        for item in etfs
        if isinstance(item, Mapping)
    ]


def _selection_total(spec: Mapping[str, object]) -> int | None:
    selection = spec.get("selection")
    if not isinstance(selection, Mapping):
        return None
    value = selection.get("total_constituents")
    return int(value) if value is not None else None


def _selection_buckets(spec: Mapping[str, object]) -> list[dict[str, object]]:
    selection = spec.get("selection")
    if not isinstance(selection, Mapping):
        return []
    buckets = selection.get("buckets") or []
    if not isinstance(buckets, list):
        return []
    return [
        {"name": str(item.get("name", "")), "count": item.get("count")}
        for item in buckets
        if isinstance(item, Mapping)
    ]


def _generic_index_inventory(spec: Mapping[str, object]) -> dict[str, object]:
    methodology = _methodology_family(spec)
    return {
        "index_code": str(spec.get("index_code", "")),
        "index_name": str(spec.get("index_name", "")),
        "tracked_etfs": _tracked_etfs(spec),
        "methodology_status": str(spec.get("status", "")),
        "replication_readiness": "missing_required_data" if methodology != "unsupported" else "unsupported_methodology",
        "methodology_summary": {
            "family": methodology,
            "total_constituents": _selection_total(spec),
            "buckets": _selection_buckets(spec),
            "weighting": spec.get("weighting", {}),
            "rebalance": spec.get("rebalance", {}),
        },
        "requirements": [],
    }


def _methodology_family(spec: Mapping[str, object]) -> str:
    weighting = spec.get("weighting")
    if not isinstance(weighting, Mapping):
        return "unsupported"
    base = str(weighting.get("base", ""))
    residual = weighting.get("residual")
    buckets = _selection_buckets(spec)
    bucket_names = [item["name"] for item in buckets]
    if bucket_names == ["top2", "momentum", "market_cap_fill"]:
        return "top2_plus"
    if residual:
        return "fixed_plus_residual"
    if base in {"float_market_cap_weighted", "market_cap_weighted"}:
        return "float_market_cap_weighted"
    if base == "equal_weighted":
        return "equal_weighted"
    return "unsupported"
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/etfs/test_fnguide_data_inventory.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add etfs/fnguide/data_inventory.py tests/etfs/test_fnguide_data_inventory.py
git commit -m "Add conservative FnGuide data inventory core" -m "Index replication readiness must be based on official data availability instead of proxy inputs or existing weighting-engine support." -m "Constraint: KSS remains missing_required_data until official theme, score, bucket, and target-weight evidence are available." -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: python -m pytest tests/etfs/test_fnguide_data_inventory.py -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

---

## Task 3: Add JSON And Markdown Inventory Writers

**Files:**
- Modify: `etfs/fnguide/data_inventory.py`
- Modify: `tests/etfs/test_fnguide_data_inventory.py`

- [ ] **Step 1: Write failing writer tests**

Append to `tests/etfs/test_fnguide_data_inventory.py`:

```python
from etfs.fnguide.data_inventory import (
    write_fnguide_data_inventory,
    write_kss_data_inventory,
)


def test_write_kss_data_inventory_outputs_json_and_markdown(tmp_path: Path) -> None:
    specs_path = _write_specs(tmp_path)
    json_path, md_path = write_kss_data_inventory(tmp_path, specs_path=specs_path, local_paths={})

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")

    assert json_path.name == "kss_data_inventory.json"
    assert md_path.name == "kss_data_inventory.md"
    assert payload["index_code"] == "FI00.WLT.KSS"
    assert payload["replication_readiness"] == "missing_required_data"
    assert "KSS/SOL Data Inventory" in markdown
    assert "official_target_weights" in markdown
    assert "external_required" in markdown


def test_write_fnguide_data_inventory_outputs_provider_summary(tmp_path: Path) -> None:
    specs_path = _write_specs(tmp_path)
    json_path, md_path = write_fnguide_data_inventory(tmp_path, specs_path=specs_path, local_paths={})

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")

    assert json_path.name == "data_inventory.json"
    assert md_path.name == "data_inventory.md"
    assert payload["provider"] == "fnguide"
    assert payload["counts"]["by_readiness"] == {"missing_required_data": 1}
    assert "FnGuide Data Inventory" in markdown
    assert "| `FI00.WLT.KSS` |" in markdown
```

- [ ] **Step 2: Run writer tests and verify they fail**

Run:

```powershell
python -m pytest tests/etfs/test_fnguide_data_inventory.py::test_write_kss_data_inventory_outputs_json_and_markdown tests/etfs/test_fnguide_data_inventory.py::test_write_fnguide_data_inventory_outputs_provider_summary -q
```

Expected: FAIL with `ImportError` for missing writer functions.

- [ ] **Step 3: Add writer functions and Markdown renderers**

Append to `etfs/fnguide/data_inventory.py`:

```python
def write_kss_data_inventory(
    output_dir: Path,
    *,
    specs_path: Path = paths.FNGUIDE_METHODOLOGY_SPECS_JSON,
    local_paths: Mapping[str, Path] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_kss_data_inventory(specs_path=specs_path, local_paths=local_paths)
    json_path = output_dir / "kss_data_inventory.json"
    md_path = output_dir / "kss_data_inventory.md"
    json_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_kss_inventory_markdown(inventory), encoding="utf-8")
    return json_path, md_path


def write_fnguide_data_inventory(
    output_dir: Path,
    *,
    specs_path: Path = paths.FNGUIDE_METHODOLOGY_SPECS_JSON,
    local_paths: Mapping[str, Path] | None = None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_fnguide_data_inventory(specs_path=specs_path, local_paths=local_paths)
    json_path = output_dir / "data_inventory.json"
    md_path = output_dir / "data_inventory.md"
    json_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_provider_inventory_markdown(inventory), encoding="utf-8")
    return json_path, md_path


def _kss_inventory_markdown(inventory: Mapping[str, object]) -> str:
    lines = [
        "# KSS/SOL Data Inventory",
        "",
        f"- index_code: {inventory.get('index_code', '')}",
        f"- index_name: {inventory.get('index_name', '')}",
        f"- replication_readiness: {inventory.get('replication_readiness', '')}",
        "",
        "## Tracked ETFs",
        "",
    ]
    tracked_etfs = inventory.get("tracked_etfs", [])
    if isinstance(tracked_etfs, list) and tracked_etfs:
        for item in tracked_etfs:
            if isinstance(item, Mapping):
                lines.append(f"- `{item.get('etf_code', '')}`: {item.get('etf_name', '')}")
    else:
        lines.append("- none recorded")
    lines.extend(
        [
            "",
            "## Requirements",
            "",
            "| requirement | category | status | full replication satisfied | evidence | external need |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in inventory.get("requirements", []):
        if not isinstance(item, Mapping):
            continue
        evidence = ", ".join(str(value) for value in item.get("local_evidence", []))
        lines.append(
            f"| `{item.get('name', '')}` | {item.get('category', '')} | `{item.get('status', '')}` | "
            f"{item.get('satisfies_full_replication', False)} | {evidence} | {item.get('external_need', '')} |"
        )
    return "\n".join(lines) + "\n"


def _provider_inventory_markdown(inventory: Mapping[str, object]) -> str:
    lines = [
        "# FnGuide Data Inventory",
        "",
        "## Summary",
        "",
    ]
    counts = inventory.get("counts", {})
    if isinstance(counts, Mapping):
        lines.append(f"- indices: {counts.get('indices', 0)}")
        by_readiness = counts.get("by_readiness", {})
        if isinstance(by_readiness, Mapping):
            for status, count in by_readiness.items():
                lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Indices",
            "",
            "| index_code | index_name | readiness | tracked_etfs |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in inventory.get("indices", []):
        if not isinstance(item, Mapping):
            continue
        etf_codes = []
        for etf in item.get("tracked_etfs", []):
            if isinstance(etf, Mapping) and etf.get("etf_code"):
                etf_codes.append(str(etf["etf_code"]))
        lines.append(
            f"| `{item.get('index_code', '')}` | {item.get('index_name', '')} | "
            f"`{item.get('replication_readiness', '')}` | {', '.join(etf_codes)} |"
        )
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run data-inventory tests and verify they pass**

Run:

```powershell
python -m pytest tests/etfs/test_fnguide_data_inventory.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add etfs/fnguide/data_inventory.py tests/etfs/test_fnguide_data_inventory.py
git commit -m "Write FnGuide replication data inventory reports" -m "Reviewable JSON and Markdown outputs make KSS missing-data blockers visible before anyone promotes proxy inputs to full replication." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: python -m pytest tests/etfs/test_fnguide_data_inventory.py -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

---

## Task 4: Add CLI And Pipeline Integration

**Files:**
- Modify: `etfs/fnguide/data_inventory.py`
- Modify: `tests/etfs/test_fnguide_data_inventory.py`
- Modify: `etfs/fnguide/pipeline.py`
- Modify: `tests/etfs/test_fnguide_pipeline.py`
- Modify: `tests/etfs/test_output_paths.py`

- [ ] **Step 1: Write failing CLI default test**

Add import to `tests/etfs/test_output_paths.py`:

```python
from etfs.fnguide.data_inventory import build_parser as build_data_inventory_parser
```

Append to `test_cli_defaults_write_to_grouped_output_folders`:

```python
    assert build_data_inventory_parser().parse_args([]).output_dir == paths.FNGUIDE_REPLICATION_OUTPUT_DIR.as_posix()
```

Append to `test_cli_defaults_read_from_grouped_output_inputs`:

```python
    assert build_data_inventory_parser().parse_args([]).specs == paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix()
```

- [ ] **Step 2: Write failing pipeline output test**

In `tests/etfs/test_fnguide_pipeline.py`, add fake writer calls near the other fake pipeline writers:

```python
    def fake_write_fnguide_data_inventory(output_dir: Path, *, specs_path: Path):
        calls.append(f"data_inventory:{specs_path.name}")
        return _touch(output_dir / "data_inventory.json"), _touch(output_dir / "data_inventory.md")

    def fake_write_kss_data_inventory(output_dir: Path, *, specs_path: Path):
        calls.append(f"kss_data_inventory:{specs_path.name}")
        return _touch(output_dir / "kss_data_inventory.json"), _touch(output_dir / "kss_data_inventory.md")
```

Add monkeypatches:

```python
    monkeypatch.setattr(pipeline, "write_fnguide_data_inventory", fake_write_fnguide_data_inventory)
    monkeypatch.setattr(pipeline, "write_kss_data_inventory", fake_write_kss_data_inventory)
```

Update the expected call list to include these entries after KSS requirements and before methodology replication report:

```python
        "data_inventory:methodology_specs.json",
        "kss_data_inventory:methodology_specs.json",
```

Add output assertions:

```python
    assert manifest["outputs"]["data_inventory"].endswith("data_inventory.json")
    assert manifest["outputs"]["data_inventory_md"].endswith("data_inventory.md")
    assert manifest["outputs"]["kss_data_inventory"].endswith("kss_data_inventory.json")
    assert manifest["outputs"]["kss_data_inventory_md"].endswith("kss_data_inventory.md")
```

For tests that monkeypatch `write_methodology_replication_report` but not every writer, also monkeypatch inventory writers with:

```python
    monkeypatch.setattr(
        pipeline,
        "write_fnguide_data_inventory",
        lambda output_dir, *, specs_path: (_touch(output_dir / "data_inventory.json"), _touch(output_dir / "data_inventory.md")),
    )
    monkeypatch.setattr(
        pipeline,
        "write_kss_data_inventory",
        lambda output_dir, *, specs_path: (_touch(output_dir / "kss_data_inventory.json"), _touch(output_dir / "kss_data_inventory.md")),
    )
```

- [ ] **Step 3: Run focused tests and verify they fail**

Run:

```powershell
python -m pytest tests/etfs/test_output_paths.py::test_cli_defaults_write_to_grouped_output_folders tests/etfs/test_fnguide_pipeline.py::test_run_offline_pipeline_sequences_verified_artifacts_and_skips_missing_engine_inputs -q
```

Expected: FAIL with missing parser import/function and missing pipeline writer imports.

- [ ] **Step 4: Add CLI to data inventory module**

Append to `etfs/fnguide/data_inventory.py`:

```python
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write FnGuide full-replication data inventory reports.")
    parser.add_argument("--specs", default=paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_REPLICATION_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    write_fnguide_data_inventory(output_dir, specs_path=Path(args.specs))
    write_kss_data_inventory(output_dir, specs_path=Path(args.specs))
    print(f"wrote {output_dir / 'data_inventory.json'}")
    print(f"wrote {output_dir / 'kss_data_inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Integrate writers into pipeline**

In `etfs/fnguide/pipeline.py`, add imports:

```python
from etfs.fnguide.data_inventory import write_fnguide_data_inventory, write_kss_data_inventory
```

After writing `kss_data_requirements`, add:

```python
    data_inventory_json, data_inventory_md = write_fnguide_data_inventory(
        replication_output_dir,
        specs_path=methodology_specs,
    )
    outputs["data_inventory"] = data_inventory_json.as_posix()
    outputs["data_inventory_md"] = data_inventory_md.as_posix()
    kss_inventory_json, kss_inventory_md = write_kss_data_inventory(
        replication_output_dir,
        specs_path=methodology_specs,
    )
    outputs["kss_data_inventory"] = kss_inventory_json.as_posix()
    outputs["kss_data_inventory_md"] = kss_inventory_md.as_posix()
```

- [ ] **Step 6: Run focused tests and verify they pass**

Run:

```powershell
python -m pytest tests/etfs/test_output_paths.py tests/etfs/test_fnguide_pipeline.py tests/etfs/test_fnguide_data_inventory.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add etfs/fnguide/data_inventory.py tests/etfs/test_fnguide_data_inventory.py etfs/fnguide/pipeline.py tests/etfs/test_fnguide_pipeline.py tests/etfs/test_output_paths.py
git commit -m "Wire data inventory reports into the FnGuide pipeline" -m "The offline pipeline should publish replication readiness evidence alongside KSS requirements so missing official data blocks are visible before backtesting." -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: python -m pytest tests/etfs/test_output_paths.py tests/etfs/test_fnguide_pipeline.py tests/etfs/test_fnguide_data_inventory.py -q" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

---

## Task 5: Document The Inventory Gate And Refresh Artifacts

**Files:**
- Modify: `etfs/README.md`
- Test: `tests/etfs/test_fnguide_data_inventory.py`
- Generated artifacts:
  - `etfs/output/replication/fnguide/data_inventory.json`
  - `etfs/output/replication/fnguide/data_inventory.md`
  - `etfs/output/replication/fnguide/kss_data_inventory.json`
  - `etfs/output/replication/fnguide/kss_data_inventory.md`
  - `etfs/output/engine/fnguide/offline_pipeline_manifest.json`

- [ ] **Step 1: Update README output list**

In `etfs/README.md`, add these bullets near the other replication outputs:

```markdown
- `output/replication/fnguide/data_inventory.json`, `output/replication/fnguide/data_inventory.md`: provider-wide full-replication data inventory and readiness report
- `output/replication/fnguide/kss_data_inventory.json`, `output/replication/fnguide/kss_data_inventory.md`: focused SOL/KSS data inventory showing available local inputs and missing official evidence
```

After the KSS replication paragraph, add:

```markdown
Full replication work is gated by the data inventory. KSS/SOL remains `missing_required_data` until official semiconductor theme membership, sales/composite score inputs, bucket assignments, and target weights are available. Local price, float-market-cap, sector, and ETF holdings data can be reported as evidence, but proxy classifications or ETF holdings do not upgrade a run to full replication.
```

- [ ] **Step 2: Run docs-adjacent tests**

Run:

```powershell
python -m pytest tests/etfs/test_fnguide_data_inventory.py tests/etfs/test_fnguide_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 3: Run offline pipeline to refresh artifacts**

Run:

```powershell
python -m etfs.fnguide.pipeline
```

Expected: command exits 0 and prints `wrote etfs\output\engine\fnguide\offline_pipeline_manifest.json`.

- [ ] **Step 4: Inspect generated inventory outputs**

Run:

```powershell
python -c "import json; p=json.load(open('etfs/output/replication/fnguide/kss_data_inventory.json', encoding='utf-8')); print(p['index_code']); print(p['replication_readiness']); print([r['name'] for r in p['requirements'] if r['status']=='external_required'])"
```

Expected output includes:

```text
FI00.WLT.KSS
missing_required_data
['theme_membership', 'sales_momentum', 'composite_score', 'official_bucket_assignments', 'official_target_weights']
```

Run:

```powershell
python -c "import json; p=json.load(open('etfs/output/engine/fnguide/offline_pipeline_manifest.json', encoding='utf-8')); print(p['outputs'].get('data_inventory')); print(p['outputs'].get('kss_data_inventory'))"
```

Expected output includes:

```text
etfs/output/replication/fnguide/data_inventory.json
etfs/output/replication/fnguide/kss_data_inventory.json
```

- [ ] **Step 5: Commit docs and generated artifacts**

```powershell
git add etfs/README.md etfs/output/replication/fnguide/data_inventory.json etfs/output/replication/fnguide/data_inventory.md etfs/output/replication/fnguide/kss_data_inventory.json etfs/output/replication/fnguide/kss_data_inventory.md etfs/output/engine/fnguide/offline_pipeline_manifest.json
git commit -m "Publish FnGuide replication data inventory artifacts" -m "The generated reports make SOL/KSS missing official data explicit and keep the pipeline from presenting local proxy inputs as full replication readiness." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: python -m pytest tests/etfs/test_fnguide_data_inventory.py tests/etfs/test_fnguide_pipeline.py -q; python -m etfs.fnguide.pipeline" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

---

## Task 6: Final Verification

**Files:**
- Verify all files touched in Tasks 1-5.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/etfs/test_fnguide_data_inventory.py tests/etfs/test_fnguide_pipeline.py tests/etfs/test_output_paths.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full ETF tests**

Run:

```powershell
python -m pytest tests/etfs -q
```

Expected: PASS.

- [ ] **Step 3: Run lint**

Run:

```powershell
python -m ruff check etfs tests/etfs
```

Expected: PASS.

- [ ] **Step 4: Run direct inventory CLI**

Run:

```powershell
python -m etfs.fnguide.data_inventory
```

Expected: exits 0 and prints both:

```text
wrote etfs/output/replication/fnguide/data_inventory.json
wrote etfs/output/replication/fnguide/kss_data_inventory.json
```

- [ ] **Step 5: Run offline pipeline**

Run:

```powershell
python -m etfs.fnguide.pipeline
```

Expected: exits 0 and writes `etfs/output/engine/fnguide/offline_pipeline_manifest.json`.

- [ ] **Step 6: Validate KSS/SOL readiness is still conservative**

Run:

```powershell
python -c "import json; p=json.load(open('etfs/output/replication/fnguide/kss_data_inventory.json', encoding='utf-8')); assert p['replication_readiness']=='missing_required_data'; req={r['name']: r for r in p['requirements']}; assert req['issuer_holdings_snapshot']['satisfies_full_replication'] is False; assert req['official_target_weights']['status']=='external_required'; print('kss inventory gate ok')"
```

Expected:

```text
kss inventory gate ok
```

- [ ] **Step 7: Review final diff scope**

Run:

```powershell
git status --short -- etfs tests/etfs docs/superpowers/plans/2026-06-16-index-replication-data-inventory.md sidecar tests/sidecar
```

Expected: intended `etfs`, `tests/etfs`, and plan/artifact changes only. `sidecar` may appear as pre-existing untracked files, but no `sidecar` path should be staged or committed.

- [ ] **Step 8: Commit refreshed artifacts if timestamps changed**

If verification regenerated only tracked inventory and manifest artifacts, commit those exact files:

```powershell
git add etfs/output/replication/fnguide/data_inventory.json etfs/output/replication/fnguide/data_inventory.md etfs/output/replication/fnguide/kss_data_inventory.json etfs/output/replication/fnguide/kss_data_inventory.md etfs/output/engine/fnguide/offline_pipeline_manifest.json
git commit -m "Refresh FnGuide data inventory artifacts after verification" -m "Final verification reran the inventory and offline pipeline, so checked-in artifacts should reflect the current conservative KSS readiness gate." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: python -m pytest tests/etfs -q; python -m ruff check etfs tests/etfs; python -m etfs.fnguide.data_inventory; python -m etfs.fnguide.pipeline" -m "Co-authored-by: OmX <omx@oh-my-codex.dev>"
```

---

## Self-Review

Spec coverage:

- KSS/SOL first inventory is covered by Tasks 2, 3, and 5.
- Common provider-wide inventory is covered by Tasks 2, 3, and 4.
- Readiness/status vocabulary is encoded in Task 2.
- JSON and Markdown output contract is covered by Task 3.
- Pipeline integration is covered by Task 4.
- Documentation and artifact publication are covered by Task 5.
- Conservative full-replication gate is covered by Tasks 2 and 6.
- `sidecar` is explicitly excluded from all tasks.

Placeholder scan:

- The plan contains no unfilled task steps.
- Each implementation task has concrete test snippets, code snippets, commands, expected outputs, and commit commands.

Type consistency:

- Output names are consistently `data_inventory`, `data_inventory_md`, `kss_data_inventory`, and `kss_data_inventory_md`.
- KSS readiness is consistently `missing_required_data`.
- Official data blockers are consistently `theme_membership`, `sales_momentum`, `composite_score`, `official_bucket_assignments`, and `official_target_weights`.
