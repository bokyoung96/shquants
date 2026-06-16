from __future__ import annotations

import json
import math
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
