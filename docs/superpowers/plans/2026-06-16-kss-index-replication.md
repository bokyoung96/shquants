# KSS Index Replication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first end-to-end FnGuide tracked-index replication lane for `FI00.WLT.KSS`, from input data requirements through bucket selection, target-weight generation, and validation reporting.

**Architecture:** Add a KSS replication lane beside the existing FnGuide methodology engine. New data-contract and selection modules produce the `constituents_by_bucket` input already accepted by `methodology_engine.py`; orchestration and validation modules write inspectable artifacts and update replication status without duplicating target-weight math.

**Tech Stack:** Python 3, dataclasses, JSON artifacts, existing `etfs.paths`, existing `etfs.fnguide.methodology_engine`, pytest, ruff.

---

## File Structure

Create or modify only `etfs`, `tests/etfs`, and this plan's documentation surface.

Files to create:

- `etfs/fnguide/replication_data.py`: KSS data requirement metadata and snapshot validation helpers.
- `etfs/fnguide/selection.py`: KSS eligible-universe filtering and bucket selection.
- `etfs/fnguide/replication.py`: End-to-end KSS orchestration and artifact writers.
- `tests/etfs/test_kss_replication_data.py`: data-contract tests.
- `tests/etfs/test_kss_selection.py`: bucket-selection tests.
- `tests/etfs/test_kss_replication.py`: orchestration, artifact, and validation tests.

Files to modify:

- `etfs/paths.py`: add `etfs/output/replication/fnguide` paths.
- `etfs/fnguide/methodology_engine.py`: include KSS full-replication status when replication artifacts exist.
- `etfs/fnguide/pipeline.py`: run KSS requirement reporting and optional fixture replication.
- `etfs/README.md`: document the KSS replication lane and artifacts.
- `tests/etfs/test_output_paths.py`: cover new paths.
- `tests/etfs/test_methodology_engine.py`: cover KSS status upgrade in the replication report.
- `tests/etfs/test_fnguide_pipeline.py`: cover pipeline outputs and skipped reasons.

Do not touch `sidecar`.

---

## Task 1: Add Replication Output Paths

**Files:**
- Modify: `etfs/paths.py`
- Test: `tests/etfs/test_output_paths.py`

- [ ] **Step 1: Write the failing path test**

Add this test to `tests/etfs/test_output_paths.py`:

```python
def test_fnguide_replication_paths_are_grouped_under_replication_output() -> None:
    assert paths.REPLICATION_OUTPUT_DIR.as_posix() == "etfs/output/replication"
    assert paths.FNGUIDE_REPLICATION_OUTPUT_DIR.as_posix() == "etfs/output/replication/fnguide"
    assert paths.FNGUIDE_KSS_REQUIREMENTS_JSON.as_posix() == "etfs/output/replication/fnguide/kss_data_requirements.json"
    assert paths.FNGUIDE_KSS_SELECTED_BUCKETS_JSON.as_posix() == "etfs/output/replication/fnguide/kss_selected_buckets.json"
    assert paths.FNGUIDE_KSS_TARGET_WEIGHTS_JSON.as_posix() == "etfs/output/replication/fnguide/kss_target_weights.json"
    assert paths.FNGUIDE_KSS_VALIDATION_JSON.as_posix() == "etfs/output/replication/fnguide/kss_replication_validation.json"
    assert paths.FNGUIDE_KSS_VALIDATION_MD.as_posix() == "etfs/output/replication/fnguide/kss_replication_validation.md"
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
python -m pytest tests/etfs/test_output_paths.py::test_fnguide_replication_paths_are_grouped_under_replication_output -q
```

Expected: FAIL with an `AttributeError` for `REPLICATION_OUTPUT_DIR`.

- [ ] **Step 3: Add path constants**

Add these constants to `etfs/paths.py` after the engine output constants:

```python
REPLICATION_OUTPUT_DIR = OUTPUT_ROOT / "replication"
FNGUIDE_REPLICATION_OUTPUT_DIR = REPLICATION_OUTPUT_DIR / "fnguide"
FNGUIDE_KSS_REQUIREMENTS_JSON = FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_data_requirements.json"
FNGUIDE_KSS_SELECTED_BUCKETS_JSON = FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_selected_buckets.json"
FNGUIDE_KSS_TARGET_WEIGHTS_JSON = FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_target_weights.json"
FNGUIDE_KSS_VALIDATION_JSON = FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_replication_validation.json"
FNGUIDE_KSS_VALIDATION_MD = FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_replication_validation.md"
```

- [ ] **Step 4: Run the test and verify it passes**

Run:

```powershell
python -m pytest tests/etfs/test_output_paths.py::test_fnguide_replication_paths_are_grouped_under_replication_output -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add etfs/paths.py tests/etfs/test_output_paths.py
git commit -m "Add FnGuide replication artifact paths" -m "KSS full-index replication needs a separate output lane from weighting-only engine artifacts so selected buckets, target weights, and validation reports remain inspectable." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: python -m pytest tests/etfs/test_output_paths.py::test_fnguide_replication_paths_are_grouped_under_replication_output -q"
```

---

## Task 2: Add KSS Data Contract And Requirement Report

**Files:**
- Create: `etfs/fnguide/replication_data.py`
- Create: `tests/etfs/test_kss_replication_data.py`

- [ ] **Step 1: Write failing data-contract tests**

Create `tests/etfs/test_kss_replication_data.py`:

```python
import json
from pathlib import Path

from etfs.fnguide.replication_data import (
    KSS_REQUIRED_DATASETS,
    build_kss_data_requirements,
    require_kss_snapshot_fields,
    write_kss_data_requirements,
)


def test_kss_required_datasets_cover_full_replication_chain() -> None:
    names = [item["name"] for item in KSS_REQUIRED_DATASETS]

    assert names == [
        "methodology_spec",
        "rebalance_calendar",
        "security_master",
        "eligibility_flags",
        "market_snapshot",
        "classification_snapshot",
        "selection_metrics",
        "official_target_weights",
        "etf_holdings_snapshot",
    ]


def test_build_kss_data_requirements_marks_missing_datasets() -> None:
    requirements = build_kss_data_requirements(available_datasets={"methodology_spec", "market_snapshot"})

    assert requirements["index_code"] == "FI00.WLT.KSS"
    assert requirements["replication_stage"] == "data_contract"
    assert requirements["available_datasets"] == ["market_snapshot", "methodology_spec"]
    assert "selection_metrics" in requirements["missing_datasets"]
    assert requirements["full_replication_ready"] is False


def test_require_kss_snapshot_fields_rejects_missing_required_fields() -> None:
    rows = [{"security_code": "A000001", "float_market_cap": 100.0}]

    errors = require_kss_snapshot_fields(rows)

    assert errors == [
        {
            "row": 0,
            "security_code": "A000001",
            "missing_fields": [
                "as_of",
                "is_eligible",
                "is_semiconductor_theme",
                "composite_momentum_score",
            ],
        }
    ]


def test_write_kss_data_requirements_outputs_json(tmp_path: Path) -> None:
    output_path = write_kss_data_requirements(tmp_path, available_datasets={"methodology_spec"})

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_path.name == "kss_data_requirements.json"
    assert payload["index_code"] == "FI00.WLT.KSS"
    assert payload["full_replication_ready"] is False
    assert "methodology_spec" in payload["available_datasets"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/etfs/test_kss_replication_data.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'etfs.fnguide.replication_data'`.

- [ ] **Step 3: Add the data-contract module**

Create `etfs/fnguide/replication_data.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


KSS_INDEX_CODE = "FI00.WLT.KSS"
KSS_INDEX_NAME = "FnGuide AI Semiconductor TOP2 Plus Index"

KSS_REQUIRED_DATASETS = [
    {
        "name": "methodology_spec",
        "required_fields": ["index_code", "index_name", "selection", "weighting", "status"],
        "purpose": "Executable KSS methodology rules and PDF-backed status.",
    },
    {
        "name": "rebalance_calendar",
        "required_fields": ["index_code", "review_date", "data_cutoff_date", "effective_date"],
        "purpose": "Date alignment for universe, selection metrics, and target weights.",
    },
    {
        "name": "security_master",
        "required_fields": ["as_of", "security_code", "name", "market", "listing_status", "stock_type"],
        "purpose": "Base Korean listed-security universe.",
    },
    {
        "name": "eligibility_flags",
        "required_fields": ["as_of", "security_code", "is_eligible"],
        "purpose": "Methodology exclusions such as non-common shares, suspended names, and managed issues.",
    },
    {
        "name": "market_snapshot",
        "required_fields": ["as_of", "security_code", "float_market_cap"],
        "purpose": "Top2 and fill-bucket ranking plus residual weighting.",
    },
    {
        "name": "classification_snapshot",
        "required_fields": ["as_of", "security_code", "is_semiconductor_theme"],
        "purpose": "KSS semiconductor or provider-theme membership.",
    },
    {
        "name": "selection_metrics",
        "required_fields": ["as_of", "security_code", "composite_momentum_score"],
        "purpose": "Momentum bucket ranking after top2 exclusion.",
    },
    {
        "name": "official_target_weights",
        "required_fields": ["index_code", "effective_date", "security_code", "official_weight"],
        "purpose": "Primary full-replication validation evidence.",
    },
    {
        "name": "etf_holdings_snapshot",
        "required_fields": ["etf_code", "as_of", "security_code", "holding_weight"],
        "purpose": "Secondary validation evidence when official targets are absent.",
    },
]

KSS_SNAPSHOT_REQUIRED_FIELDS = [
    "as_of",
    "security_code",
    "is_eligible",
    "is_semiconductor_theme",
    "float_market_cap",
    "composite_momentum_score",
]


def build_kss_data_requirements(*, available_datasets: Iterable[str] = ()) -> dict[str, object]:
    available = sorted(set(available_datasets))
    required = [str(item["name"]) for item in KSS_REQUIRED_DATASETS]
    missing = [name for name in required if name not in available]
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_code": KSS_INDEX_CODE,
        "index_name": KSS_INDEX_NAME,
        "replication_stage": "data_contract",
        "required_datasets": KSS_REQUIRED_DATASETS,
        "available_datasets": available,
        "missing_datasets": missing,
        "full_replication_ready": not missing,
    }


def require_kss_snapshot_fields(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        missing = [field for field in KSS_SNAPSHOT_REQUIRED_FIELDS if row.get(field) in {None, ""}]
        if missing:
            errors.append(
                {
                    "row": row_index,
                    "security_code": str(row.get("security_code", "")),
                    "missing_fields": missing,
                }
            )
    return errors


def write_kss_data_requirements(output_dir: Path, *, available_datasets: Iterable[str] = ()) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "kss_data_requirements.json"
    output_path.write_text(
        json.dumps(build_kss_data_requirements(available_datasets=available_datasets), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/etfs/test_kss_replication_data.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add etfs/fnguide/replication_data.py tests/etfs/test_kss_replication_data.py
git commit -m "Define KSS replication data contract" -m "Full index replication requires explicit data availability reporting before selection and weighting can claim methodology coverage." -m "Constraint: KSS remains the tracer bullet; broader FnGuide specs are not promoted by this task." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: python -m pytest tests/etfs/test_kss_replication_data.py -q"
```

---

## Task 3: Implement KSS Bucket Selection

**Files:**
- Create: `etfs/fnguide/selection.py`
- Create: `tests/etfs/test_kss_selection.py`

- [ ] **Step 1: Write failing selection tests**

Create `tests/etfs/test_kss_selection.py`:

```python
import pytest

from etfs.fnguide.selection import select_kss_buckets


def _row(code: str, float_cap: float, momentum: float, *, eligible: bool = True, theme: bool = True) -> dict[str, object]:
    return {
        "as_of": "2026-05-29",
        "security_code": code,
        "name": code,
        "is_eligible": eligible,
        "is_semiconductor_theme": theme,
        "float_market_cap": float_cap,
        "composite_momentum_score": momentum,
    }


def test_select_kss_buckets_builds_top2_momentum_and_fill() -> None:
    rows = [
        _row("A000001", 1000, 1),
        _row("A000002", 900, 2),
        _row("A000003", 800, 90),
        _row("A000004", 700, 80),
        _row("A000005", 600, 70),
        _row("A000006", 500, 60),
        _row("A000007", 400, 10),
        _row("A000008", 300, 20),
        _row("A000009", 200, 30),
        _row("A000010", 100, 40),
        _row("A000011", 50, 5),
    ]

    buckets = select_kss_buckets(rows)

    assert [item["security_code"] for item in buckets["top2"]] == ["A000001", "A000002"]
    assert [item["security_code"] for item in buckets["momentum"]] == ["A000003", "A000004", "A000005", "A000006"]
    assert [item["security_code"] for item in buckets["market_cap_fill"]] == ["A000007", "A000008", "A000009", "A000010"]


def test_select_kss_buckets_excludes_ineligible_and_non_theme_names() -> None:
    rows = [
        _row("A000001", 1000, 1),
        _row("A000002", 900, 2),
        _row("A000003", 850, 99, eligible=False),
        _row("A000004", 840, 98, theme=False),
        _row("A000005", 800, 90),
        _row("A000006", 700, 80),
        _row("A000007", 600, 70),
        _row("A000008", 500, 60),
        _row("A000009", 400, 50),
        _row("A000010", 300, 40),
        _row("A000011", 200, 30),
        _row("A000012", 100, 20),
    ]

    buckets = select_kss_buckets(rows)
    selected_codes = {item["security_code"] for members in buckets.values() for item in members}

    assert "A000003" not in selected_codes
    assert "A000004" not in selected_codes
    assert len(selected_codes) == 10


def test_select_kss_buckets_uses_deterministic_tie_breakers() -> None:
    rows = [
        _row("A000002", 1000, 1),
        _row("A000001", 1000, 1),
        _row("A000003", 800, 50),
        _row("A000004", 700, 50),
        _row("A000005", 600, 50),
        _row("A000006", 500, 50),
        _row("A000007", 400, 40),
        _row("A000008", 300, 30),
        _row("A000009", 200, 20),
        _row("A000010", 100, 10),
    ]

    buckets = select_kss_buckets(rows)

    assert [item["security_code"] for item in buckets["top2"]] == ["A000001", "A000002"]
    assert [item["security_code"] for item in buckets["momentum"]] == ["A000003", "A000004", "A000005", "A000006"]


def test_select_kss_buckets_rejects_missing_metric() -> None:
    rows = [_row(f"A{i:06d}", 1000 - i, 100 - i) for i in range(10)]
    rows[3]["composite_momentum_score"] = None

    with pytest.raises(ValueError, match="A000003 missing composite_momentum_score"):
        select_kss_buckets(rows)


def test_select_kss_buckets_rejects_insufficient_candidates() -> None:
    rows = [_row(f"A{i:06d}", 1000 - i, 100 - i) for i in range(9)]

    with pytest.raises(ValueError, match="KSS requires 10 eligible theme constituents"):
        select_kss_buckets(rows)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/etfs/test_kss_selection.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'etfs.fnguide.selection'`.

- [ ] **Step 3: Add KSS selection implementation**

Create `etfs/fnguide/selection.py`:

```python
from __future__ import annotations

from typing import Iterable, Mapping


KSS_BUCKET_COUNTS = {"top2": 2, "momentum": 4, "market_cap_fill": 4}


def select_kss_buckets(rows: Iterable[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    candidates = [_kss_candidate(row) for row in rows if _truthy(row.get("is_eligible")) and _truthy(row.get("is_semiconductor_theme"))]
    if len(candidates) < 10:
        raise ValueError(f"KSS requires 10 eligible theme constituents, got {len(candidates)}")

    top2 = _take_ranked(candidates, count=2, metric="float_market_cap")
    remaining_after_top2 = _exclude(candidates, top2)
    momentum = _take_ranked(remaining_after_top2, count=4, metric="composite_momentum_score")
    remaining_after_momentum = _exclude(remaining_after_top2, momentum)
    market_cap_fill = _take_ranked(remaining_after_momentum, count=4, metric="float_market_cap")

    return {
        "top2": [_bucket_row(item, "top2", "float_market_cap") for item in top2],
        "momentum": [_bucket_row(item, "momentum", "composite_momentum_score") for item in momentum],
        "market_cap_fill": [_bucket_row(item, "market_cap_fill", "float_market_cap") for item in market_cap_fill],
    }


def _kss_candidate(row: Mapping[str, object]) -> dict[str, object]:
    code = str(row.get("security_code", "")).strip()
    if not code:
        raise ValueError("KSS candidate security_code is required")
    for metric in ["float_market_cap", "composite_momentum_score"]:
        if row.get(metric) in {None, ""}:
            raise ValueError(f"{code} missing {metric}")
        if float(row[metric]) <= 0:
            raise ValueError(f"{code} {metric} must be positive")
    return {
        "as_of": str(row.get("as_of", "")),
        "security_code": code,
        "name": str(row.get("name", "")),
        "float_market_cap": float(row["float_market_cap"]),
        "composite_momentum_score": float(row["composite_momentum_score"]),
    }


def _take_ranked(rows: list[dict[str, object]], *, count: int, metric: str) -> list[dict[str, object]]:
    if len(rows) < count:
        raise ValueError(f"KSS {metric} bucket requires {count} constituents, got {len(rows)}")
    return sorted(
        rows,
        key=lambda row: (-float(row[metric]), -float(row["float_market_cap"]), str(row["security_code"])),
    )[:count]


def _exclude(rows: list[dict[str, object]], selected: list[dict[str, object]]) -> list[dict[str, object]]:
    selected_codes = {str(row["security_code"]) for row in selected}
    return [row for row in rows if str(row["security_code"]) not in selected_codes]


def _bucket_row(row: Mapping[str, object], bucket: str, rank_metric: str) -> dict[str, object]:
    return {
        "bucket": bucket,
        "rank_metric": rank_metric,
        "security_code": str(row["security_code"]),
        "name": str(row.get("name", "")),
        "float_market_cap": float(row["float_market_cap"]),
        "composite_momentum_score": float(row["composite_momentum_score"]),
    }


def _truthy(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes", "y"}
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/etfs/test_kss_selection.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add etfs/fnguide/selection.py tests/etfs/test_kss_selection.py
git commit -m "Select KSS methodology buckets from source snapshots" -m "Target weights cannot prove index replication unless KSS top2, momentum, and fill buckets are generated from dated source data." -m "Rejected: Feed hand-written constituents into the engine | preserves the weighting-only limitation." -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: python -m pytest tests/etfs/test_kss_selection.py -q"
```

---

## Task 4: Add KSS Replication Validation Diff

**Files:**
- Create: `tests/etfs/test_kss_replication.py`
- Create: `etfs/fnguide/replication.py`

- [ ] **Step 1: Write failing validation tests**

Create `tests/etfs/test_kss_replication.py` with these first tests:

```python
from etfs.fnguide.replication import build_replication_validation


def test_build_replication_validation_passes_exact_official_targets() -> None:
    calculated = [
        {"security_code": "A000001", "target_weight": 0.25},
        {"security_code": "A000002", "target_weight": 0.25},
    ]
    official = [
        {"security_code": "A000001", "official_weight": 0.25},
        {"security_code": "A000002", "official_weight": 0.25},
    ]

    result = build_replication_validation(
        index_code="FI00.WLT.KSS",
        as_of="2026-05-29",
        validation_source_type="official_target_weights",
        calculated_target_weights=calculated,
        validation_weights=official,
        weight_tolerance=0.0,
    )

    assert result["status"] == "passed"
    assert result["checks"] == {"constituent_membership": "passed", "weight_tolerance": "passed"}
    assert result["metrics"]["max_abs_weight_difference"] == 0.0
    assert result["differences"] == []


def test_build_replication_validation_reports_missing_extra_and_drift() -> None:
    calculated = [
        {"security_code": "A000001", "target_weight": 0.25},
        {"security_code": "A000002", "target_weight": 0.25},
    ]
    official = [
        {"security_code": "A000001", "official_weight": 0.20},
        {"security_code": "A000003", "official_weight": 0.30},
    ]

    result = build_replication_validation(
        index_code="FI00.WLT.KSS",
        as_of="2026-05-29",
        validation_source_type="official_target_weights",
        calculated_target_weights=calculated,
        validation_weights=official,
        weight_tolerance=0.01,
    )

    assert result["status"] == "failed"
    assert result["checks"] == {"constituent_membership": "failed", "weight_tolerance": "failed"}
    assert result["metrics"]["max_abs_weight_difference"] == 0.05
    assert result["metrics"]["total_abs_weight_difference"] == 0.05
    assert result["differences"] == [
        {"type": "missing_in_validation", "security_code": "A000002", "target_weight": 0.25},
        {"type": "extra_in_validation", "security_code": "A000003", "validation_weight": 0.3},
        {
            "type": "weight_difference",
            "security_code": "A000001",
            "target_weight": 0.25,
            "validation_weight": 0.2,
            "difference": 0.05,
        },
    ]


def test_build_replication_validation_marks_missing_validation_source() -> None:
    result = build_replication_validation(
        index_code="FI00.WLT.KSS",
        as_of="2026-05-29",
        validation_source_type="missing",
        calculated_target_weights=[{"security_code": "A000001", "target_weight": 1.0}],
        validation_weights=[],
        weight_tolerance=0.0,
    )

    assert result["status"] == "not_proven"
    assert result["checks"] == {"validation_source": "missing"}
    assert result["differences"] == [{"type": "validation_source_missing", "index_code": "FI00.WLT.KSS", "as_of": "2026-05-29"}]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m pytest tests/etfs/test_kss_replication.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'etfs.fnguide.replication'`.

- [ ] **Step 3: Add validation functions to replication module**

Create `etfs/fnguide/replication.py` with the validation functions first:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


def build_replication_validation(
    *,
    index_code: str,
    as_of: str,
    validation_source_type: str,
    calculated_target_weights: Iterable[Mapping[str, object]],
    validation_weights: Iterable[Mapping[str, object]],
    weight_tolerance: float,
) -> dict[str, object]:
    if validation_source_type == "missing":
        return {
            "index_code": index_code,
            "as_of": as_of,
            "validation_source_type": "missing",
            "status": "not_proven",
            "checks": {"validation_source": "missing"},
            "metrics": {},
            "differences": [{"type": "validation_source_missing", "index_code": index_code, "as_of": as_of}],
        }

    target = _weights_by_security(calculated_target_weights, weight_key="target_weight")
    validation = _weights_by_security(validation_weights, weight_key=_validation_weight_key(validation_source_type))
    target_codes = set(target)
    validation_codes = set(validation)
    differences: list[dict[str, object]] = []
    for code in sorted(target_codes - validation_codes):
        differences.append({"type": "missing_in_validation", "security_code": code, "target_weight": round(target[code], 12)})
    for code in sorted(validation_codes - target_codes):
        differences.append({"type": "extra_in_validation", "security_code": code, "validation_weight": round(validation[code], 12)})

    abs_differences: list[float] = []
    for code in sorted(target_codes & validation_codes):
        difference = round(target[code] - validation[code], 12)
        abs_difference = abs(difference)
        abs_differences.append(abs_difference)
        if abs_difference > weight_tolerance:
            differences.append(
                {
                    "type": "weight_difference",
                    "security_code": code,
                    "target_weight": round(target[code], 12),
                    "validation_weight": round(validation[code], 12),
                    "difference": difference,
                }
            )

    membership_passed = target_codes == validation_codes
    max_abs_difference = round(max(abs_differences, default=0.0), 12)
    weight_passed = max_abs_difference <= weight_tolerance
    return {
        "index_code": index_code,
        "as_of": as_of,
        "validation_source_type": validation_source_type,
        "status": "passed" if membership_passed and weight_passed else "failed",
        "checks": {
            "constituent_membership": "passed" if membership_passed else "failed",
            "weight_tolerance": "passed" if weight_passed else "failed",
        },
        "metrics": {
            "target_constituent_count": len(target_codes),
            "validation_constituent_count": len(validation_codes),
            "common_constituent_count": len(target_codes & validation_codes),
            "max_abs_weight_difference": max_abs_difference,
            "total_abs_weight_difference": round(sum(abs_differences), 12),
        },
        "differences": differences,
    }


def write_replication_validation(report: Mapping[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "kss_replication_validation.json"
    md_path = output_dir / "kss_replication_validation.md"
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": dict(report),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_validation_markdown(report), encoding="utf-8")
    return json_path, md_path


def _weights_by_security(rows: Iterable[Mapping[str, object]], *, weight_key: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for row in rows:
        code = str(row.get("security_code", "")).strip()
        if not code:
            raise ValueError("validation weight security_code is required")
        weights[code] = float(row.get(weight_key, 0.0) or 0.0)
    return weights


def _validation_weight_key(validation_source_type: str) -> str:
    if validation_source_type == "official_target_weights":
        return "official_weight"
    if validation_source_type == "etf_holdings_snapshot":
        return "holding_weight"
    raise ValueError(f"unsupported validation_source_type: {validation_source_type}")


def _validation_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# KSS Replication Validation",
        "",
        f"- index_code: {report.get('index_code', '')}",
        f"- as_of: {report.get('as_of', '')}",
        f"- validation_source_type: {report.get('validation_source_type', '')}",
        f"- status: {report.get('status', '')}",
        "",
        "## Differences",
        "",
    ]
    differences = report.get("differences", [])
    if not differences:
        lines.append("- none")
    for difference in differences:
        lines.append(f"- {difference}")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/etfs/test_kss_replication.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add etfs/fnguide/replication.py tests/etfs/test_kss_replication.py
git commit -m "Compare KSS target weights with validation evidence" -m "Full replication status needs an explicit diff between calculated target weights and official targets or secondary ETF holdings evidence." -m "Constraint: Missing validation evidence must produce not_proven rather than passed." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: python -m pytest tests/etfs/test_kss_replication.py -q"
```

---

## Task 5: Orchestrate KSS Selection Into Target Weights

**Files:**
- Modify: `etfs/fnguide/replication.py`
- Modify: `tests/etfs/test_kss_replication.py`

- [ ] **Step 1: Add failing orchestration test**

Append this test to `tests/etfs/test_kss_replication.py`:

```python
from pathlib import Path

from etfs.fnguide.replication import build_kss_replication, write_kss_replication_artifacts


def _snapshot_row(code: str, float_cap: float, momentum: float) -> dict[str, object]:
    return {
        "as_of": "2026-05-29",
        "security_code": code,
        "name": code,
        "is_eligible": True,
        "is_semiconductor_theme": True,
        "float_market_cap": float_cap,
        "composite_momentum_score": momentum,
    }


def test_build_kss_replication_selects_buckets_and_calculates_target_weights() -> None:
    rows = [
        _snapshot_row("A000001", 1000, 1),
        _snapshot_row("A000002", 900, 2),
        _snapshot_row("A000003", 800, 90),
        _snapshot_row("A000004", 700, 80),
        _snapshot_row("A000005", 600, 70),
        _snapshot_row("A000006", 500, 60),
        _snapshot_row("A000007", 400, 10),
        _snapshot_row("A000008", 300, 20),
        _snapshot_row("A000009", 200, 30),
        _snapshot_row("A000010", 100, 40),
    ]

    result = build_kss_replication(
        as_of="2026-05-29",
        effective_date="2026-06-14",
        snapshot_rows=rows,
        validation_weights=[],
        validation_source_type="missing",
    )

    assert result["index_code"] == "FI00.WLT.KSS"
    assert result["target_weight_result"]["checks"] == {"constituent_count": "passed", "weight_sum": "passed"}
    weights = {item["security_code"]: item["target_weight"] for item in result["target_weight_result"]["target_weights"]}
    assert weights["A000001"] == 0.25
    assert weights["A000002"] == 0.25
    assert result["validation"]["status"] == "not_proven"


def test_write_kss_replication_artifacts_outputs_selected_weights_and_validation(tmp_path: Path) -> None:
    rows = [_snapshot_row(f"A{i:06d}", 1000 - i, 100 - i) for i in range(10)]
    result = build_kss_replication(
        as_of="2026-05-29",
        effective_date="2026-06-14",
        snapshot_rows=rows,
        validation_weights=[],
        validation_source_type="missing",
    )

    outputs = write_kss_replication_artifacts(result, tmp_path)

    assert outputs == {
        "selected_buckets": (tmp_path / "kss_selected_buckets.json").as_posix(),
        "target_weights": (tmp_path / "kss_target_weights.json").as_posix(),
        "replication_validation": (tmp_path / "kss_replication_validation.json").as_posix(),
        "replication_validation_md": (tmp_path / "kss_replication_validation.md").as_posix(),
    }
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
python -m pytest tests/etfs/test_kss_replication.py::test_build_kss_replication_selects_buckets_and_calculates_target_weights -q
```

Expected: FAIL with `ImportError` for `build_kss_replication`.

- [ ] **Step 3: Add KSS orchestration**

Append these imports and functions to `etfs/fnguide/replication.py`:

```python
from etfs import paths
from etfs.fnguide.methodology_engine import calculate_top2_plus_target_weights, load_engine_ready_specs
from etfs.fnguide.replication_data import KSS_INDEX_CODE
from etfs.fnguide.selection import select_kss_buckets


def build_kss_replication(
    *,
    as_of: str,
    effective_date: str,
    snapshot_rows: Iterable[Mapping[str, object]],
    validation_weights: Iterable[Mapping[str, object]],
    validation_source_type: str,
    specs_path: Path = paths.FNGUIDE_METHODOLOGY_SPECS_JSON,
    weight_tolerance: float = 0.0,
) -> dict[str, object]:
    ready_specs = load_engine_ready_specs(specs_path)
    spec = ready_specs[KSS_INDEX_CODE]
    selected_buckets = select_kss_buckets(snapshot_rows)
    weights = calculate_top2_plus_target_weights(spec, selected_buckets)
    target_weights = [
        {"security_code": code, "target_weight": round(weight, 12)}
        for code, weight in weights.items()
    ]
    target_result = {
        "index_code": KSS_INDEX_CODE,
        "as_of": as_of,
        "effective_date": effective_date,
        "methodology": "top2_plus",
        "checks": {
            "constituent_count": "passed" if len(target_weights) == 10 else "failed",
            "weight_sum": "passed" if abs(round(sum(weights.values()), 12) - 1.0) <= 1e-10 else "failed",
        },
        "metrics": {
            "constituent_count": len(target_weights),
            "weight_sum": round(sum(weights.values()), 12),
        },
        "target_weights": target_weights,
    }
    validation = build_replication_validation(
        index_code=KSS_INDEX_CODE,
        as_of=as_of,
        validation_source_type=validation_source_type,
        calculated_target_weights=target_weights,
        validation_weights=validation_weights,
        weight_tolerance=weight_tolerance,
    )
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_code": KSS_INDEX_CODE,
        "as_of": as_of,
        "effective_date": effective_date,
        "selected_buckets": selected_buckets,
        "target_weight_result": target_result,
        "validation": validation,
        "full_replication_status": "proven" if validation["status"] == "passed" else "not_proven",
    }


def write_kss_replication_artifacts(result: Mapping[str, object], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "kss_selected_buckets.json"
    target_path = output_dir / "kss_target_weights.json"
    selected_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "index_code": result.get("index_code", ""),
                "as_of": result.get("as_of", ""),
                "effective_date": result.get("effective_date", ""),
                "selected_buckets": result.get("selected_buckets", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    target_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "result": result.get("target_weight_result", {}),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    validation_json, validation_md = write_replication_validation(dict(result.get("validation", {})), output_dir)
    return {
        "selected_buckets": selected_path.as_posix(),
        "target_weights": target_path.as_posix(),
        "replication_validation": validation_json.as_posix(),
        "replication_validation_md": validation_md.as_posix(),
    }
```

- [ ] **Step 4: Run orchestration tests and verify they pass**

Run:

```powershell
python -m pytest tests/etfs/test_kss_replication.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add etfs/fnguide/replication.py tests/etfs/test_kss_replication.py
git commit -m "Generate KSS target weights from selected buckets" -m "This connects the new KSS selection layer to the existing top2_plus weighting authority so target weights are produced from dated input snapshots instead of hand-filled engine JSON." -m "Constraint: Missing validation still reports not_proven." -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: python -m pytest tests/etfs/test_kss_replication.py -q"
```

---

## Task 6: Reflect KSS Full-Replication Status In Methodology Report

**Files:**
- Modify: `etfs/fnguide/methodology_engine.py`
- Modify: `tests/etfs/test_methodology_engine.py`

- [ ] **Step 1: Add failing report test**

Append this test to `tests/etfs/test_methodology_engine.py`:

```python
def test_build_methodology_replication_report_marks_kss_proven_when_artifact_proves_validation(tmp_path: Path) -> None:
    replication_path = tmp_path / "kss_replication_validation.json"
    replication_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "result": {
                    "index_code": "FI00.WLT.KSS",
                    "as_of": "2026-05-29",
                    "validation_source_type": "official_target_weights",
                    "status": "passed",
                    "checks": {"constituent_membership": "passed", "weight_tolerance": "passed"},
                    "metrics": {},
                    "differences": [],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_methodology_replication_report(
        Path("etfs/output/extractions/fnguide/methodology_specs.json"),
        kss_replication_validation_path=replication_path,
    )

    kss = next(item for item in report["items"] if item["index_code"] == "FI00.WLT.KSS")
    assert kss["full_methodology_replication_status"] == "proven"
    assert kss["full_methodology_replication_evidence"] == replication_path.as_posix()
    assert report["counts"]["full_methodology_replication_proven"] == 1
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
python -m pytest tests/etfs/test_methodology_engine.py::test_build_methodology_replication_report_marks_kss_proven_when_artifact_proves_validation -q
```

Expected: FAIL with `TypeError` because `kss_replication_validation_path` is not accepted.

- [ ] **Step 3: Add optional validation artifact support**

Change the signature in `etfs/fnguide/methodology_engine.py`:

```python
def build_methodology_replication_report(
    specs_path: Path,
    *,
    kss_replication_validation_path: Path = paths.FNGUIDE_KSS_VALIDATION_JSON,
) -> dict[str, object]:
```

Inside the function, before building `items`, add:

```python
    kss_full_replication = _load_kss_full_replication_status(kss_replication_validation_path)
```

Change the `_methodology_replication_item(...)` call to pass `kss_full_replication`:

```python
        _methodology_replication_item(
            spec,
            support_by_index.get(str(spec.get("index_code", "")), {}),
            ready_specs,
            kss_full_replication,
        )
```

Change `_methodology_replication_item` signature:

```python
def _methodology_replication_item(
    spec: Mapping[str, object],
    support_item: Mapping[str, object],
    ready_specs: Mapping[str, Mapping[str, object]],
    kss_full_replication: Mapping[str, object],
) -> dict[str, object]:
```

At the start of `_methodology_replication_item`, after `index_code`, add:

```python
    proven = index_code == "FI00.WLT.KSS" and bool(kss_full_replication.get("proven"))
```

Replace the full-replication fields in `base` with:

```python
        "full_methodology_replication_status": "proven" if proven else "not_proven",
        "full_methodology_replication_evidence": str(kss_full_replication.get("path", "")) if proven else "",
        "full_methodology_replication_blockers": []
        if proven
        else [
            "constituent universe and bucket selection are supplied as explicit engine inputs",
            "official rebalance target weights are not available for direct comparison",
        ],
```

Add helper function near other private helpers:

```python
def _load_kss_full_replication_status(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"proven": False, "path": path.as_posix()}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = _mapping(payload.get("result"))
    return {
        "proven": result.get("index_code") == "FI00.WLT.KSS"
        and result.get("validation_source_type") == "official_target_weights"
        and result.get("status") == "passed",
        "path": path.as_posix(),
    }
```

- [ ] **Step 4: Run report tests and verify they pass**

Run:

```powershell
python -m pytest tests/etfs/test_methodology_engine.py::test_build_methodology_replication_report_marks_kss_proven_when_artifact_proves_validation tests/etfs/test_methodology_engine.py::test_build_methodology_replication_report_smoke_tests_real_engine_ready_specs -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add etfs/fnguide/methodology_engine.py tests/etfs/test_methodology_engine.py
git commit -m "Require KSS validation evidence for full replication status" -m "The replication report should promote KSS only when a selection-plus-validation artifact proves official target agreement, keeping weighting-only smoke tests distinct." -m "Constraint: ETF holdings validation remains secondary evidence and does not prove full methodology replication." -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: python -m pytest tests/etfs/test_methodology_engine.py::test_build_methodology_replication_report_marks_kss_proven_when_artifact_proves_validation tests/etfs/test_methodology_engine.py::test_build_methodology_replication_report_smoke_tests_real_engine_ready_specs -q"
```

---

## Task 7: Wire KSS Artifacts Into Pipeline With Safe Skips

**Files:**
- Modify: `etfs/fnguide/pipeline.py`
- Modify: `tests/etfs/test_fnguide_pipeline.py`
- Modify: `etfs/README.md`

- [ ] **Step 1: Add failing pipeline tests**

Append this test to `tests/etfs/test_fnguide_pipeline.py`:

```python
def test_offline_pipeline_writes_kss_data_requirements_and_skips_missing_snapshot(tmp_path: Path) -> None:
    manifest = run_offline_pipeline(
        rules_path=Path("etfs/output/providers/fnguide/rules.json"),
        extraction_output_dir=tmp_path / "extractions",
        overrides_path=Path("etfs/output/extractions/fnguide/spec_overrides.json"),
        validation_inputs=[],
        validation_output_dir=tmp_path / "validation",
        engine_output_dir=tmp_path / "engine",
        engine_inputs_path=tmp_path / "engine" / "engine_inputs.json",
        replication_output_dir=tmp_path / "replication",
        kss_snapshot_path=tmp_path / "replication" / "kss_snapshot.json",
    )

    assert manifest["outputs"]["kss_data_requirements"] == (tmp_path / "replication" / "kss_data_requirements.json").as_posix()
    assert "kss_replication: kss_snapshot not found" in manifest["skipped"]
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
python -m pytest tests/etfs/test_fnguide_pipeline.py::test_offline_pipeline_writes_kss_data_requirements_and_skips_missing_snapshot -q
```

Expected: FAIL with `TypeError` because `replication_output_dir` is not accepted.

- [ ] **Step 3: Add pipeline arguments and safe skip**

Modify imports in `etfs/fnguide/pipeline.py`:

```python
from etfs.fnguide.replication import build_kss_replication, write_kss_replication_artifacts
from etfs.fnguide.replication_data import write_kss_data_requirements
```

Add parameters to `run_offline_pipeline`:

```python
    replication_output_dir: Path = paths.FNGUIDE_REPLICATION_OUTPUT_DIR,
    kss_snapshot_path: Path = paths.FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_snapshot.json",
```

After engine output creation, create the replication output dir:

```python
    replication_output_dir.mkdir(parents=True, exist_ok=True)
```

After the replication report is written, add:

```python
    kss_requirements = write_kss_data_requirements(
        replication_output_dir,
        available_datasets={"methodology_spec"} | ({"etf_holdings_snapshot"} if fixtures is not None else set()),
    )
    outputs["kss_data_requirements"] = kss_requirements.as_posix()

    if kss_snapshot_path.exists():
        snapshot_payload = json.loads(kss_snapshot_path.read_text(encoding="utf-8"))
        kss_result = build_kss_replication(
            as_of=str(snapshot_payload.get("as_of", "")),
            effective_date=str(snapshot_payload.get("effective_date", "")),
            snapshot_rows=snapshot_payload.get("rows", []),
            validation_weights=snapshot_payload.get("validation_weights", []),
            validation_source_type=str(snapshot_payload.get("validation_source_type", "missing")),
            specs_path=methodology_specs,
        )
        outputs.update(write_kss_replication_artifacts(kss_result, replication_output_dir))
    else:
        skipped.append("kss_replication: kss_snapshot not found")
```

Add parser arguments:

```python
    parser.add_argument("--replication-output-dir", default=paths.FNGUIDE_REPLICATION_OUTPUT_DIR.as_posix())
    parser.add_argument("--kss-snapshot", default=(paths.FNGUIDE_REPLICATION_OUTPUT_DIR / "kss_snapshot.json").as_posix())
```

Pass them in `main`:

```python
        replication_output_dir=Path(args.replication_output_dir),
        kss_snapshot_path=Path(args.kss_snapshot),
```

- [ ] **Step 4: Update README output list**

Add these bullets to `etfs/README.md` under FnGuide outputs:

```markdown
- `output/replication/fnguide/kss_data_requirements.json`: KSS full-replication data contract and currently available datasets
- `output/replication/fnguide/kss_selected_buckets.json`: selected KSS top2, momentum, and market-cap-fill buckets when a source snapshot is supplied
- `output/replication/fnguide/kss_target_weights.json`: KSS target weights generated from selected buckets
- `output/replication/fnguide/kss_replication_validation.json`, `output/replication/fnguide/kss_replication_validation.md`: KSS validation diff against official targets or secondary ETF holdings
```

Add this paragraph after the current replication-report paragraph:

```markdown
KSS replication is the first full-methodology tracer bullet. The pipeline always writes its data-requirement artifact and only calculates selected buckets and target weights when `output/replication/fnguide/kss_snapshot.json` supplies a dated source snapshot. Missing snapshots are reported as a skip, not as a successful replication.
```

- [ ] **Step 5: Run pipeline test and verify it passes**

Run:

```powershell
python -m pytest tests/etfs/test_fnguide_pipeline.py::test_offline_pipeline_writes_kss_data_requirements_and_skips_missing_snapshot -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add etfs/fnguide/pipeline.py tests/etfs/test_fnguide_pipeline.py etfs/README.md
git commit -m "Wire KSS replication requirements into the FnGuide pipeline" -m "The offline pipeline should expose KSS full-replication data needs even when dated source snapshots are not available for calculation." -m "Constraint: Missing KSS snapshots are skips, not successful replication evidence." -m "Confidence: high" -m "Scope-risk: moderate" -m "Tested: python -m pytest tests/etfs/test_fnguide_pipeline.py::test_offline_pipeline_writes_kss_data_requirements_and_skips_missing_snapshot -q"
```

---

## Task 8: Final Verification

**Files:**
- Verify all files touched in Tasks 1-7.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/etfs/test_kss_replication_data.py tests/etfs/test_kss_selection.py tests/etfs/test_kss_replication.py tests/etfs/test_methodology_engine.py tests/etfs/test_fnguide_pipeline.py tests/etfs/test_output_paths.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full etfs tests**

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

- [ ] **Step 4: Run offline pipeline**

Run:

```powershell
python -m etfs.fnguide.pipeline
```

Expected: command exits 0 and writes `etfs/output/engine/fnguide/offline_pipeline_manifest.json`.

- [ ] **Step 5: Inspect generated manifest for KSS skip**

Run:

```powershell
python -c "import json; p=json.load(open('etfs/output/engine/fnguide/offline_pipeline_manifest.json', encoding='utf-8')); print(p['outputs'].get('kss_data_requirements')); print([s for s in p['skipped'] if s.startswith('kss_replication:')])"
```

Expected output includes:

```text
etfs/output/replication/fnguide/kss_data_requirements.json
['kss_replication: kss_snapshot not found']
```

- [ ] **Step 6: Review final diff scope**

Run:

```powershell
git status --short -- etfs tests/etfs docs/superpowers/plans/2026-06-16-kss-index-replication.md
```

Expected: only intended `etfs`, `tests/etfs`, and plan changes appear. No `sidecar` path appears.

- [ ] **Step 7: Commit final generated docs or artifacts if needed**

If Task 7 already committed all source, test, and README changes, and Task 8 generated only expected output artifacts already tracked by the current ETF research workflow, make one final artifact commit:

```powershell
git add etfs/output/replication/fnguide etfs/output/engine/fnguide/offline_pipeline_manifest.json
git commit -m "Refresh FnGuide KSS replication artifacts" -m "The pipeline now records KSS data requirements and an explicit skip when source snapshots are absent, preserving the distinction between target-weight smoke tests and full index replication." -m "Confidence: high" -m "Scope-risk: narrow" -m "Tested: python -m pytest tests/etfs -q; python -m ruff check etfs tests/etfs; python -m etfs.fnguide.pipeline"
```

---

## Self-Review

Spec coverage:

- Required data fields are covered by Task 2.
- KSS bucket selection is covered by Task 3.
- Target-weight generation from selected buckets is covered by Task 5.
- Validation diffing is covered by Task 4.
- Replication report status is covered by Task 6.
- Pipeline artifacts and safe skip behavior are covered by Task 7.
- Full verification is covered by Task 8.

Placeholder scan:

- The plan contains no unfinished marker strings or missing implementation step.
- Each code-writing step includes concrete code.
- Each test step includes exact commands and expected outcomes.

Type consistency:

- KSS index code is consistently `FI00.WLT.KSS`.
- Selected bucket names are consistently `top2`, `momentum`, and `market_cap_fill`.
- Target-weight rows consistently use `security_code` and `target_weight`.
- Official validation rows use `official_weight`; ETF holdings rows use `holding_weight`.
