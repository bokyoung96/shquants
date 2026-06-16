from __future__ import annotations

import json
import math
from json import JSONDecodeError
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths
from etfs.fnguide.methodology_engine import (
    MethodologyNotReadyError,
    calculate_top2_plus_target_weights,
    load_engine_ready_specs,
)
from etfs.fnguide.replication_data import KSS_INDEX_CODE
from etfs.fnguide.selection import select_kss_buckets


def build_replication_validation(
    *,
    index_code: str,
    as_of: str,
    validation_source_type: str,
    calculated_target_weights: Iterable[Mapping[str, object]],
    validation_weights: Iterable[Mapping[str, object]],
    weight_tolerance: float,
) -> dict[str, object]:
    if not math.isfinite(weight_tolerance):
        raise ValueError("weight_tolerance must be finite")
    if weight_tolerance < 0:
        raise ValueError("weight_tolerance must be >= 0")

    if validation_source_type == "missing":
        return {
            "index_code": index_code,
            "as_of": as_of,
            "validation_source_type": "missing",
            "status": "not_proven",
            "checks": {"validation_source": "missing"},
            "metrics": {},
            "differences": [
                {
                    "type": "validation_source_missing",
                    "index_code": index_code,
                    "as_of": as_of,
                }
            ],
        }

    target = _weights_by_security(
        calculated_target_weights,
        weight_key="target_weight",
    )
    validation = _weights_by_security(
        validation_weights,
        weight_key=_validation_weight_key(validation_source_type),
    )
    target_codes = set(target)
    validation_codes = set(validation)
    differences: list[dict[str, object]] = []
    for code in sorted(target_codes - validation_codes):
        differences.append(
            {
                "type": "missing_in_validation",
                "security_code": code,
                "target_weight": round(target[code], 12),
            }
        )
    for code in sorted(validation_codes - target_codes):
        differences.append(
            {
                "type": "extra_in_validation",
                "security_code": code,
                "validation_weight": round(validation[code], 12),
            }
        )

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


def write_replication_validation(
    report: Mapping[str, object],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "kss_replication_validation.json"
    md_path = output_dir / "kss_replication_validation.md"
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": dict(report),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    md_path.write_text(_validation_markdown(report), encoding="utf-8")
    return json_path, md_path


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
    try:
        ready_specs = load_engine_ready_specs(specs_path)
    except (FileNotFoundError, JSONDecodeError, ValueError) as exc:
        raise MethodologyNotReadyError(
            f"{KSS_INDEX_CODE} is missing or not engine-ready in {specs_path}: {exc}"
        ) from exc
    spec = ready_specs.get(KSS_INDEX_CODE)
    if spec is None:
        raise MethodologyNotReadyError(
            f"{KSS_INDEX_CODE} is missing or not engine-ready in {specs_path}"
        )
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
            "weight_sum": "passed"
            if abs(round(sum(weights.values()), 12) - 1.0) <= 1e-10
            else "failed",
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
        "full_replication_status": "proven"
        if validation["status"] == "passed"
        and validation_source_type == "official_target_weights"
        else "not_proven",
    }


def write_kss_replication_artifacts(
    result: Mapping[str, object],
    output_dir: Path,
) -> dict[str, str]:
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
            allow_nan=False,
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
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    validation_json, validation_md = write_replication_validation(
        dict(result.get("validation", {})),
        output_dir,
    )
    return {
        "selected_buckets": selected_path.as_posix(),
        "target_weights": target_path.as_posix(),
        "replication_validation": validation_json.as_posix(),
        "replication_validation_md": validation_md.as_posix(),
    }


def _weights_by_security(
    rows: Iterable[Mapping[str, object]],
    *,
    weight_key: str,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for row in rows:
        code = str(row.get("security_code", "")).strip()
        if not code:
            raise ValueError("validation weight security_code is required")
        if code in weights:
            raise ValueError(f"duplicate security_code: {code}")

        if weight_key not in row:
            raise ValueError(f"{weight_key} is required for security_code {code}")

        raw_weight = row[weight_key]
        if raw_weight is None:
            raise ValueError(f"{weight_key} is required for security_code {code}")
        if isinstance(raw_weight, str) and not raw_weight.strip():
            raise ValueError(f"{weight_key} is required for security_code {code}")

        weight = float(raw_weight)
        if not math.isfinite(weight):
            raise ValueError(f"{weight_key} must be finite for security_code {code}")

        weights[code] = weight
    return weights


def _validation_weight_key(validation_source_type: str) -> str:
    if validation_source_type == "official_target_weights":
        return "official_weight"
    if validation_source_type == "etf_holdings_snapshot":
        return "holding_weight"
    raise ValueError(
        f"unsupported validation_source_type: {validation_source_type}"
    )


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
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| type | security_code | target_weight | validation_weight | difference | index_code | as_of |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for difference in differences:
        if not isinstance(difference, Mapping):
            raise ValueError("difference entries must be mappings")
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(difference.get("type")),
                    _markdown_cell(difference.get("security_code")),
                    _markdown_cell(difference.get("target_weight")),
                    _markdown_cell(difference.get("validation_weight")),
                    _markdown_cell(difference.get("difference")),
                    _markdown_cell(difference.get("index_code")),
                    _markdown_cell(difference.get("as_of")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value)
