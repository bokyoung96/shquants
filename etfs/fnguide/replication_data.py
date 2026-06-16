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
