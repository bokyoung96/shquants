from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths
from etfs.fnguide.methodology_audit import build_methodology_audit


class MethodologyNotReadyError(ValueError):
    """Raised when a methodology spec is not approved for calculation."""


def load_engine_ready_specs(path: Path = paths.FNGUIDE_METHODOLOGY_SPECS_JSON) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs = payload.get("indices")
    if not isinstance(specs, list):
        raise ValueError(f"unsupported methodology specs payload: {path}")
    ready: dict[str, dict[str, object]] = {}
    for spec in specs:
        if not isinstance(spec, Mapping):
            continue
        try:
            require_engine_ready_spec(spec)
        except MethodologyNotReadyError:
            continue
        ready[str(spec.get("index_code", ""))] = dict(spec)
    return ready


def require_engine_ready_spec(spec: Mapping[str, object]) -> Mapping[str, object]:
    audit = build_methodology_audit([spec])
    item = audit["items"][0]
    if item["engine_ready"]:
        return spec
    blockers = ", ".join(str(blocker) for blocker in item["blockers"])
    index_code = str(spec.get("index_code", ""))
    raise MethodologyNotReadyError(f"{index_code} is not engine-ready: {blockers}")


def calculate_top2_plus_target_weights(
    spec: Mapping[str, object],
    constituents_by_bucket: Mapping[str, list[Mapping[str, object]]],
) -> dict[str, float]:
    require_engine_ready_spec(spec)
    bucket_specs = _bucket_specs(spec)
    top2_spec = bucket_specs.get("top2")
    if top2_spec is None:
        raise ValueError("top2_plus methodology requires top2 bucket")
    residual = _mapping(_mapping(spec.get("weighting")).get("residual"))
    residual_bucket_names = [str(name) for name in residual.get("applies_to_buckets", [])]
    if residual_bucket_names != ["momentum", "market_cap_fill"]:
        raise ValueError("top2_plus methodology requires momentum and market_cap_fill residual buckets")

    top2_members = _bucket_members(constituents_by_bucket, "top2", _int(top2_spec.get("count")))
    residual_members: list[dict[str, object]] = []
    for bucket_name in residual_bucket_names:
        bucket_spec = bucket_specs.get(bucket_name)
        if bucket_spec is None:
            raise ValueError(f"top2_plus methodology requires {bucket_name} bucket")
        residual_members.extend(_bucket_members(constituents_by_bucket, bucket_name, _int(bucket_spec.get("count"))))

    _validate_unique_security_codes(top2_members + residual_members)

    top2_weight = _fixed_bucket_weight(top2_spec)
    weights = {_security_code(member): top2_weight for member in top2_members}
    residual_weights = _capped_pro_rata_weights(
        residual_members,
        total_weight=_float(residual.get("total_weight")),
        cap=_optional_float(residual.get("cap")),
    )
    weights.update(residual_weights)
    return weights


def calculate_equal_weight_target_weights(
    spec: Mapping[str, object],
    constituents: list[Mapping[str, object]],
) -> dict[str, float]:
    require_engine_ready_spec(spec)
    members = _constituents(constituents, _expected_total_constituents(spec))
    _validate_unique_security_codes(members)
    weight = 1.0 / len(members)
    return {_security_code(member): weight for member in members}


def calculate_capped_float_market_cap_target_weights(
    spec: Mapping[str, object],
    constituents: list[Mapping[str, object]],
) -> dict[str, float]:
    require_engine_ready_spec(spec)
    members = _constituents(constituents, _expected_total_constituents(spec))
    _validate_unique_security_codes(members)
    weighting = _mapping(spec.get("weighting"))
    return _capped_pro_rata_weights(
        members,
        total_weight=1.0,
        cap=_optional_float(weighting.get("security_cap")),
    )


def calculate_capped_metric_target_weights(
    spec: Mapping[str, object],
    constituents: list[Mapping[str, object]],
) -> dict[str, float]:
    require_engine_ready_spec(spec)
    members = _constituents(constituents, _expected_total_constituents(spec))
    _validate_unique_security_codes(members)
    weighting = _mapping(spec.get("weighting"))
    metric = str(weighting.get("metric", "")).strip()
    if not metric:
        raise ValueError("metric_weighted methodology requires weighting.metric")
    return _capped_pro_rata_weights(
        members,
        total_weight=1.0,
        cap=_optional_float(weighting.get("security_cap")),
        metric=metric,
    )


def calculate_fixed_plus_residual_target_weights(
    spec: Mapping[str, object],
    constituents_by_bucket: Mapping[str, list[Mapping[str, object]]],
) -> dict[str, float]:
    require_engine_ready_spec(spec)
    expected_total = _expected_total_constituents(spec)
    bucket_specs = _bucket_specs(spec)
    residual = _mapping(_mapping(spec.get("weighting")).get("residual"))
    residual_bucket_names = [str(name) for name in residual.get("applies_to_buckets", [])]
    if not residual_bucket_names:
        raise ValueError("fixed_plus_residual methodology requires residual.applies_to_buckets")

    residual_bucket_set = set(residual_bucket_names)
    weights: dict[str, float] = {}
    all_members: list[dict[str, object]] = []
    fixed_weight_sum = 0.0
    residual_members: list[dict[str, object]] = []
    for bucket_name, bucket_spec in bucket_specs.items():
        members = _bucket_members(constituents_by_bucket, bucket_name, _int(bucket_spec.get("count")))
        all_members.extend(members)
        if bucket_name in residual_bucket_set:
            residual_members.extend(members)
            continue
        fixed_weight = _fixed_bucket_weight(bucket_spec)
        fixed_weight_sum += fixed_weight * len(members)
        weights.update({_security_code(member): fixed_weight for member in members})

    if len(all_members) != expected_total:
        raise ValueError(f"methodology requires {expected_total} constituents, got {len(all_members)}")
    _validate_unique_security_codes(all_members)

    residual_total_weight = _float(residual.get("total_weight"))
    if abs((fixed_weight_sum + residual_total_weight) - 1.0) > 1e-10:
        raise ValueError("fixed and residual weights must sum to 1.0")

    residual_base = str(residual.get("base", ""))
    if residual_base == "equal_weighted":
        if not residual_members:
            raise ValueError("residual weighting requires at least one constituent")
        residual_weight = residual_total_weight / len(residual_members)
        weights.update({_security_code(member): residual_weight for member in residual_members})
    elif residual_base == "float_market_cap":
        weights.update(
            _capped_pro_rata_weights(
                residual_members,
                total_weight=residual_total_weight,
                cap=_optional_float(residual.get("cap")),
                metric="float_market_cap",
            )
        )
    else:
        raise ValueError(f"unsupported fixed_plus_residual residual base: {residual_base}")
    return weights


def write_target_weights(
    inputs_path: Path,
    specs_path: Path,
    output_dir: Path,
) -> Path:
    ready_specs = load_engine_ready_specs(specs_path)
    payload = json.loads(inputs_path.read_text(encoding="utf-8"))
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise ValueError(f"unsupported engine input payload: {inputs_path}")
    results = [_target_weight_result(request, ready_specs) for request in requests]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "target_weights.json"
    output_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(results),
        "results": results,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_engine_input_requirements(
    specs_path: Path,
    output_dir: Path,
) -> Path:
    ready_specs = load_engine_ready_specs(specs_path)
    requirements = [_engine_input_requirement(spec) for spec in ready_specs.values()]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "engine_input_requirements.json"
    output_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(requirements),
        "requirements": requirements,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_engine_input_template(
    specs_path: Path,
    output_dir: Path,
) -> Path:
    ready_specs = load_engine_ready_specs(specs_path)
    requests = [_engine_input_request_template(spec) for spec in ready_specs.values()]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "engine_inputs.template.json"
    output_payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "template_only": True,
        "instructions": "Copy this file to engine_inputs.json and replace placeholders with independent calculation inputs.",
        "count": len(requests),
        "requests": requests,
    }
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def build_methodology_replication_report(specs_path: Path) -> dict[str, object]:
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    specs = [_mapping(spec) for spec in payload.get("indices", [])]
    matrix = build_engine_support_matrix(specs)
    support_by_index = {str(item["index_code"]): _mapping(item) for item in matrix["items"]}
    ready_specs = load_engine_ready_specs(specs_path)
    items = [
        _methodology_replication_item(
            spec,
            support_by_index.get(str(spec.get("index_code", "")), {}),
            ready_specs,
        )
        for spec in specs
    ]
    counts = Counter(str(item["target_weight_replication_status"]) for item in items)
    full_counts = Counter(str(item["full_methodology_replication_status"]) for item in items)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "target-weight calculation from verified methodology specs and explicit engine inputs",
        "limitations": [
            "Does not prove issuer universe construction or provider-level constituent selection.",
            "Does not compare calculated targets to official rebalance target files.",
            "ETF holdings workbooks remain validation evidence, not methodology calculation inputs.",
        ],
        "counts": {
            "total_specs": len(items),
            "engine_ready": sum(item.get("engine_support_status") == "engine_ready" for item in items),
            "target_weight_replication_passed": counts.get("passed", 0),
            "target_weight_replication_failed": counts.get("failed", 0),
            "target_weight_replication_not_run": counts.get("not_run", 0),
            "full_methodology_replication_proven": full_counts.get("proven", 0),
            "full_methodology_replication_not_proven": full_counts.get("not_proven", 0),
        },
        "items": items,
    }


def write_methodology_replication_report(
    specs_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    report = build_methodology_replication_report(specs_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "methodology_replication_report.json"
    md_path = output_dir / "methodology_replication_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_methodology_replication_markdown(report), encoding="utf-8")
    return json_path, md_path


def build_engine_support_matrix(specs: Iterable[Mapping[str, object]]) -> dict[str, object]:
    items = [_engine_support_item(spec) for spec in specs]
    counts = Counter(str(item["engine_support_status"]) for item in items)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "total": len(items),
            "engine_ready": counts.get("engine_ready", 0),
            "supported_after_review": counts.get("supported_after_review", 0),
            "blocked_by_methodology_evidence": counts.get("blocked_by_methodology_evidence", 0),
            "unsupported_methodology": counts.get("unsupported_methodology", 0),
        },
        "items": items,
    }


def write_engine_support_matrix(specs_path: Path, output_dir: Path) -> tuple[Path, Path]:
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    specs = payload.get("indices", [])
    matrix = build_engine_support_matrix([_mapping(spec) for spec in specs])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "engine_support_matrix.json"
    md_path = output_dir / "engine_support_matrix.md"
    json_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_engine_support_markdown(matrix), encoding="utf-8")
    return json_path, md_path


def build_engine_promotion_candidates(specs: Iterable[Mapping[str, object]]) -> dict[str, object]:
    matrix = build_engine_support_matrix(specs)
    items = [
        _promotion_candidate_item(_mapping(item))
        for item in matrix["items"]
        if _mapping(item).get("engine_support_status") == "supported_after_review"
    ]
    counts = Counter(str(item["methodology"]) for item in items)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {"total": len(items), "by_methodology": dict(counts)},
        "items": items,
    }


def write_engine_promotion_candidates(specs_path: Path, output_dir: Path) -> tuple[Path, Path]:
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    specs = payload.get("indices", [])
    candidates = build_engine_promotion_candidates([_mapping(spec) for spec in specs])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "engine_promotion_candidates.json"
    md_path = output_dir / "engine_promotion_candidates.md"
    json_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_promotion_candidates_markdown(candidates), encoding="utf-8")
    return json_path, md_path


def _promotion_candidate_item(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "index_code": str(item.get("index_code", "")),
        "index_name": str(item.get("index_name", "")),
        "status": str(item.get("status", "")),
        "methodology": str(item.get("methodology", "")),
        "blockers": [str(blocker) for blocker in item.get("blockers", [])],
        "required_review": [
            "verify PDF evidence supports extracted total_constituents",
            "verify PDF evidence supports weighting method and cap scope",
            "promote status only after evidence review",
        ],
    }


def _promotion_candidates_markdown(candidates: Mapping[str, object]) -> str:
    counts = _mapping(candidates.get("counts"))
    lines = [
        "# FnGuide Engine Promotion Candidates",
        "",
        f"- Total candidates: {counts.get('total', 0)}",
        "",
        "| index_code | index_name | methodology | blockers |",
        "| --- | --- | --- | --- |",
    ]
    for item in candidates.get("items", []):
        row = _mapping(item)
        blockers = "<br>".join(str(blocker) for blocker in row.get("blockers", []))
        lines.append(f"| {row.get('index_code')} | {row.get('index_name')} | {row.get('methodology')} | {blockers} |")
    return "\n".join(lines) + "\n"


def _engine_support_item(spec: Mapping[str, object]) -> dict[str, object]:
    audit = build_methodology_audit([spec])
    audit_item = _mapping(audit["items"][0])
    blockers = [str(blocker) for blocker in audit_item.get("blockers", [])]
    try:
        methodology = _methodology_for_spec(spec)
    except ValueError as exc:
        return {
            "index_code": str(spec.get("index_code", "")),
            "index_name": str(spec.get("index_name", "")),
            "status": str(spec.get("status", "")),
            "methodology": "unsupported",
            "engine_support_status": "unsupported_methodology",
            "blockers": blockers,
            "engine_blockers": [str(exc)],
            "next_action": "implement methodology-specific engine only after PDF evidence is verified",
        }
    engine_blockers: list[str] = []
    try:
        _validate_supported_spec_shape(spec, methodology)
    except ValueError as exc:
        engine_blockers.append(str(exc))
    if bool(audit_item.get("engine_ready")):
        support_status = "engine_ready"
        next_action = "provide explicit engine_inputs and run target-weight calculation"
    elif engine_blockers or _has_methodology_evidence_blocker(blockers):
        support_status = "blocked_by_methodology_evidence"
        next_action = "resolve PDF/spec blockers before engine enablement"
    else:
        support_status = "supported_after_review"
        next_action = "review PDF evidence and promote status before engine enablement"
    return {
        "index_code": str(spec.get("index_code", "")),
        "index_name": str(spec.get("index_name", "")),
        "status": str(spec.get("status", "")),
        "methodology": methodology,
        "engine_support_status": support_status,
        "blockers": blockers,
        "engine_blockers": engine_blockers,
        "next_action": next_action,
    }


def _validate_supported_spec_shape(spec: Mapping[str, object], methodology: str) -> None:
    if methodology in {"equal_weighted", "float_market_cap_weighted"}:
        _expected_total_constituents(spec)
    if methodology == "metric_weighted":
        _expected_total_constituents(spec)
        weighting = _mapping(spec.get("weighting"))
        if not str(weighting.get("metric", "")).strip():
            raise ValueError("metric_weighted missing weighting.metric")
    if methodology == "fixed_plus_residual":
        _expected_total_constituents(spec)
        bucket_specs = _bucket_specs(spec)
        residual = _mapping(_mapping(spec.get("weighting")).get("residual"))
        residual_bucket_names = [str(name) for name in residual.get("applies_to_buckets", [])]
        if not residual_bucket_names:
            raise ValueError("fixed_plus_residual missing residual.applies_to_buckets")
        for name in residual_bucket_names:
            if name not in bucket_specs:
                raise ValueError(f"fixed_plus_residual missing residual bucket: {name}")
        for name, bucket_spec in bucket_specs.items():
            _int(bucket_spec.get("count"))
            if name not in residual_bucket_names:
                _fixed_bucket_weight(bucket_spec)
        residual_base = str(residual.get("base", ""))
        if residual_base not in {"equal_weighted", "float_market_cap"}:
            raise ValueError(f"unsupported fixed_plus_residual residual base: {residual_base}")
        _float(residual.get("total_weight"))
    if methodology == "top2_plus":
        bucket_specs = _bucket_specs(spec)
        for name in ["top2", "momentum", "market_cap_fill"]:
            if name not in bucket_specs:
                raise ValueError(f"top2_plus missing {name} bucket")


def _has_methodology_evidence_blocker(blockers: list[str]) -> bool:
    return any(
        blocker != "status=draft_extracted"
        and (
            "selection.total_constituents" in blocker
            or "weight cap exists" in blocker
            or "rules.selection_count differs" in blocker
            or "missing" in blocker
        )
        for blocker in blockers
    )


def _engine_support_markdown(matrix: Mapping[str, object]) -> str:
    counts = _mapping(matrix.get("counts"))
    lines = [
        "# FnGuide Engine Support Matrix",
        "",
        f"- Total specs: {counts.get('total', 0)}",
        f"- Engine ready: {counts.get('engine_ready', 0)}",
        f"- Supported after review: {counts.get('supported_after_review', 0)}",
        f"- Blocked by methodology evidence: {counts.get('blocked_by_methodology_evidence', 0)}",
        f"- Unsupported methodology: {counts.get('unsupported_methodology', 0)}",
        "",
        "| index_code | index_name | status | methodology | engine_support_status | next_action |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in matrix.get("items", []):
        row = _mapping(item)
        lines.append(
            f"| {row.get('index_code')} | {row.get('index_name')} | {row.get('status')} | "
            f"{row.get('methodology')} | {row.get('engine_support_status')} | {row.get('next_action')} |"
        )
    return "\n".join(lines) + "\n"


def _engine_input_requirement(spec: Mapping[str, object]) -> dict[str, object]:
    methodology = _methodology_for_spec(spec)
    if methodology == "equal_weighted":
        return {
            "index_code": str(spec.get("index_code", "")),
            "index_name": str(spec.get("index_name", "")),
            "methodology": methodology,
            "required_fields": ["security_code"],
            "required_count": _expected_total_constituents(spec),
        }
    if methodology == "float_market_cap_weighted":
        weighting = _mapping(spec.get("weighting"))
        return {
            "index_code": str(spec.get("index_code", "")),
            "index_name": str(spec.get("index_name", "")),
            "methodology": methodology,
            "required_fields": ["security_code", "float_market_cap"],
            "required_count": _expected_total_constituents(spec),
            "weighting": {
                "security_cap": weighting.get("security_cap"),
                "redistribution": "iterative_pro_rata",
            },
        }
    if methodology == "metric_weighted":
        weighting = _mapping(spec.get("weighting"))
        metric = str(weighting.get("metric", "")).strip()
        return {
            "index_code": str(spec.get("index_code", "")),
            "index_name": str(spec.get("index_name", "")),
            "methodology": methodology,
            "required_fields": ["security_code", metric],
            "required_count": _expected_total_constituents(spec),
            "weighting": {
                "metric": metric,
                "security_cap": weighting.get("security_cap"),
                "redistribution": "iterative_pro_rata",
            },
        }
    bucket_specs = _bucket_specs(spec)
    residual = _mapping(_mapping(spec.get("weighting")).get("residual"))
    residual_buckets = {str(name) for name in residual.get("applies_to_buckets", [])}
    if methodology == "fixed_plus_residual":
        residual_base = str(residual.get("base", ""))
        required_fields = ["security_code"]
        if residual_base == "float_market_cap":
            required_fields.append("float_market_cap")
        return {
            "index_code": str(spec.get("index_code", "")),
            "index_name": str(spec.get("index_name", "")),
            "methodology": methodology,
            "required_fields": required_fields,
            "required_buckets": [
                {
                    "name": name,
                    "count": _int(bucket_spec.get("count")),
                    "weighting": _fixed_plus_residual_bucket_weighting(name, residual_buckets, residual_base),
                }
                for name, bucket_spec in bucket_specs.items()
            ],
            "weighting": {
                "residual_total_weight": residual.get("total_weight"),
                "residual_cap": residual.get("cap"),
                "redistribution": residual.get("redistribution"),
            },
        }
    return {
        "index_code": str(spec.get("index_code", "")),
        "index_name": str(spec.get("index_name", "")),
        "methodology": "top2_plus",
        "required_fields": ["security_code", "float_market_cap"],
        "required_buckets": [
            {
                "name": name,
                "count": _int(bucket_specs[name].get("count")),
                "weighting": "residual_float_market_cap" if name in residual_buckets else "fixed",
            }
            for name in ["top2", "momentum", "market_cap_fill"]
            if name in bucket_specs
        ],
    }


def _engine_input_request_template(spec: Mapping[str, object]) -> dict[str, object]:
    requirement = _engine_input_requirement(spec)
    fields = [str(field) for field in requirement.get("required_fields", [])]
    request: dict[str, object] = {
        "index_code": str(requirement.get("index_code", "")),
        "as_of": "",
        "methodology": str(requirement.get("methodology", "")),
    }
    buckets = requirement.get("required_buckets")
    if isinstance(buckets, list) and buckets:
        request["constituents_by_bucket"] = {
            str(_mapping(bucket).get("name", "")): [_placeholder_constituent(fields) for _ in range(_int(_mapping(bucket).get("count")))]
            for bucket in buckets
        }
    else:
        request["constituents"] = [
            _placeholder_constituent(fields)
            for _ in range(_int(requirement.get("required_count")))
        ]
    return request


def _placeholder_constituent(fields: list[str]) -> dict[str, object]:
    return {field: "" if field == "security_code" else None for field in fields}


def _methodology_replication_item(
    spec: Mapping[str, object],
    support_item: Mapping[str, object],
    ready_specs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    index_code = str(spec.get("index_code", ""))
    methodology = str(support_item.get("methodology", ""))
    full_replication_blockers = [
        "constituent universe and bucket selection are supplied as explicit engine inputs",
        "official rebalance target weights are not available for direct comparison",
    ]
    base = {
        "index_code": index_code,
        "index_name": str(spec.get("index_name", "")),
        "methodology": methodology,
        "engine_support_status": str(support_item.get("engine_support_status", "")),
        "engine_blockers": [str(blocker) for blocker in support_item.get("engine_blockers", [])],
        "methodology_blockers": [str(blocker) for blocker in support_item.get("blockers", [])],
        "full_methodology_replication_status": "not_proven",
        "full_methodology_replication_evidence": "",
        "full_methodology_replication_blockers": full_replication_blockers,
    }
    if index_code not in ready_specs:
        return {
            **base,
            "target_weight_replication_status": "not_run",
            "target_weight_checks": {},
            "target_weight_metrics": {},
            "error": "",
        }
    try:
        request = _replication_smoke_request(ready_specs[index_code])
        result = _target_weight_result(request, {index_code: ready_specs[index_code]})
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "target_weight_replication_status": "failed",
            "target_weight_checks": {},
            "target_weight_metrics": {},
            "error": str(exc),
        }
    checks = _mapping(result.get("checks"))
    status = "passed" if checks and all(value == "passed" for value in checks.values()) else "failed"
    return {
        **base,
        "target_weight_replication_status": status,
        "target_weight_checks": dict(checks),
        "target_weight_metrics": dict(_mapping(result.get("metrics"))),
        "error": "" if status == "passed" else "target-weight checks failed",
    }


def _replication_smoke_request(spec: Mapping[str, object]) -> dict[str, object]:
    requirement = _engine_input_requirement(spec)
    methodology = str(requirement.get("methodology", ""))
    request: dict[str, object] = {
        "index_code": str(requirement.get("index_code", "")),
        "as_of": "replication-smoke",
        "methodology": methodology,
    }
    buckets = requirement.get("required_buckets")
    if isinstance(buckets, list) and buckets:
        fields = [str(field) for field in requirement.get("required_fields", [])]
        request["constituents_by_bucket"] = {
            str(_mapping(bucket).get("name", "")): _synthetic_constituents(
                fields,
                _int(_mapping(bucket).get("count")),
                bucket_index,
            )
            for bucket_index, bucket in enumerate(buckets, start=1)
        }
        return request
    fields = [str(field) for field in requirement.get("required_fields", [])]
    request["constituents"] = _synthetic_constituents(fields, _int(requirement.get("required_count")), 1)
    return request


def _synthetic_constituents(fields: list[str], count: int, bucket_index: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        row: dict[str, object] = {"security_code": f"A{bucket_index:02d}{index + 1:04d}"}
        for field in fields:
            if field == "security_code":
                continue
            row[field] = 1000.0
        rows.append(row)
    return rows


def _methodology_replication_markdown(report: Mapping[str, object]) -> str:
    counts = _mapping(report.get("counts"))
    lines = [
        "# FnGuide Methodology Replication Report",
        "",
        f"- Total specs: {counts.get('total_specs', 0)}",
        f"- Engine-ready specs: {counts.get('engine_ready', 0)}",
        f"- Target-weight replication passed: {counts.get('target_weight_replication_passed', 0)}",
        f"- Target-weight replication failed: {counts.get('target_weight_replication_failed', 0)}",
        f"- Full methodology replication proven: {counts.get('full_methodology_replication_proven', 0)}",
        "",
        "## Scope",
        "",
        f"- {report.get('scope', '')}",
        "",
        "## Limitations",
        "",
    ]
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "## Items",
            "",
            "| index_code | methodology | engine_support_status | target_weight_replication_status | full_methodology_replication_status |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in report.get("items", []):
        row = _mapping(item)
        lines.append(
            f"| {row.get('index_code')} | {row.get('methodology')} | {row.get('engine_support_status')} | "
            f"{row.get('target_weight_replication_status')} | {row.get('full_methodology_replication_status')} |"
        )
    return "\n".join(lines) + "\n"


def _target_weight_result(
    request: object,
    ready_specs: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    item = _mapping(request)
    index_code = str(item.get("index_code", ""))
    spec = ready_specs.get(index_code)
    if spec is None:
        raise MethodologyNotReadyError(f"{index_code} is not engine-ready")
    methodology = str(item.get("methodology", ""))
    if methodology and methodology != _methodology_for_spec(spec):
        raise ValueError(f"unsupported methodology engine: {methodology}")
    if not methodology:
        methodology = _methodology_for_spec(spec)
    weights = _calculate_target_weights(spec, item, methodology)
    weight_sum = round(sum(weights.values()), 12)
    total_constituents = _mapping(spec.get("selection")).get("total_constituents")
    target_weights = [
        {"security_code": security_code, "target_weight": round(target_weight, 12)}
        for security_code, target_weight in weights.items()
    ]
    return {
        "index_code": index_code,
        "as_of": str(item.get("as_of", "")),
        "methodology": methodology,
        "checks": {
            "constituent_count": "passed" if len(target_weights) == total_constituents else "failed",
            "weight_sum": "passed" if abs(weight_sum - 1.0) <= 1e-10 else "failed",
        },
        "metrics": {
            "constituent_count": len(target_weights),
            "weight_sum": weight_sum,
        },
        "target_weights": target_weights,
    }


def _calculate_target_weights(
    spec: Mapping[str, object],
    item: Mapping[str, object],
    methodology: str,
) -> dict[str, float]:
    if methodology == "top2_plus":
        return calculate_top2_plus_target_weights(
            spec,
            _constituents_by_bucket(item.get("constituents_by_bucket")),
        )
    if methodology == "equal_weighted":
        return calculate_equal_weight_target_weights(spec, _constituents_input(item.get("constituents")))
    if methodology == "float_market_cap_weighted":
        return calculate_capped_float_market_cap_target_weights(spec, _constituents_input(item.get("constituents")))
    if methodology == "metric_weighted":
        return calculate_capped_metric_target_weights(spec, _constituents_input(item.get("constituents")))
    if methodology == "fixed_plus_residual":
        return calculate_fixed_plus_residual_target_weights(
            spec,
            _constituents_by_bucket(item.get("constituents_by_bucket")),
        )
    raise ValueError(f"unsupported methodology engine: {methodology}")


def _methodology_for_spec(spec: Mapping[str, object]) -> str:
    selection = _mapping(spec.get("selection"))
    buckets = selection.get("buckets")
    weighting_base = str(_mapping(spec.get("weighting")).get("base", ""))
    if weighting_base == "fixed_plus_residual":
        return "fixed_plus_residual"
    if isinstance(buckets, list) and any(str(_mapping(bucket).get("name", "")) == "top2" for bucket in buckets):
        return "top2_plus"
    if weighting_base == "equal_weighted":
        return "equal_weighted"
    if weighting_base in {"float_market_cap_weighted", "market_cap_weighted"}:
        return "float_market_cap_weighted"
    if weighting_base == "metric_weighted":
        return "metric_weighted"
    raise ValueError(f"unsupported methodology engine: {weighting_base}")


def _bucket_specs(spec: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    selection = _mapping(spec.get("selection"))
    buckets = selection.get("buckets")
    if not isinstance(buckets, list):
        raise ValueError("methodology spec selection.buckets must be a list")
    return {str(_mapping(bucket).get("name", "")): _mapping(bucket) for bucket in buckets}


def _constituents_input(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ValueError("engine input constituents must be a list")
    return [_mapping(member) for member in value]


def _constituents(value: list[Mapping[str, object]], expected_count: int) -> list[dict[str, object]]:
    if len(value) != expected_count:
        raise ValueError(f"methodology requires {expected_count} constituents, got {len(value)}")
    return [dict(member) for member in value]


def _constituents_by_bucket(value: object) -> dict[str, list[Mapping[str, object]]]:
    if not isinstance(value, Mapping):
        raise ValueError("engine input constituents_by_bucket must be a mapping")
    constituents: dict[str, list[Mapping[str, object]]] = {}
    for bucket_name, members in value.items():
        if not isinstance(members, list):
            raise ValueError(f"{bucket_name} constituents must be a list")
        constituents[str(bucket_name)] = [_mapping(member) for member in members]
    return constituents


def _expected_total_constituents(spec: Mapping[str, object]) -> int:
    selection = _mapping(spec.get("selection"))
    value = selection.get("total_constituents")
    if value is None:
        raise ValueError("methodology spec selection.total_constituents is required")
    return _int(value)


def _bucket_members(
    constituents_by_bucket: Mapping[str, list[Mapping[str, object]]],
    bucket_name: str,
    expected_count: int,
) -> list[dict[str, object]]:
    members = constituents_by_bucket.get(bucket_name, [])
    if len(members) != expected_count:
        raise ValueError(f"{bucket_name} requires {expected_count} constituents, got {len(members)}")
    return [dict(member) for member in members]


def _fixed_bucket_weight(bucket_spec: Mapping[str, object]) -> float:
    weight = _mapping(bucket_spec.get("weight"))
    if weight.get("type") != "fixed":
        raise ValueError("fixed bucket weight must be fixed")
    return _float(weight.get("value"))


def _fixed_plus_residual_bucket_weighting(name: str, residual_buckets: set[str], residual_base: str) -> str:
    if name not in residual_buckets:
        return "fixed"
    if residual_base == "equal_weighted":
        return "residual_equal_weighted"
    if residual_base == "float_market_cap":
        return "residual_float_market_cap"
    raise ValueError(f"unsupported fixed_plus_residual residual base: {residual_base}")


def _capped_pro_rata_weights(
    members: list[Mapping[str, object]],
    *,
    total_weight: float,
    cap: float | None,
    metric: str = "float_market_cap",
) -> dict[str, float]:
    if not members:
        raise ValueError("residual weighting requires at least one constituent")
    if total_weight < 0:
        raise ValueError("residual total_weight must be non-negative")
    if cap is not None and cap * len(members) + 1e-12 < total_weight:
        raise ValueError("residual cap is too low for the residual total_weight")

    remaining = {str(_security_code(member)): _float(member.get(metric)) for member in members}
    weights: dict[str, float] = {}
    remaining_weight = total_weight
    while remaining:
        metric_total = sum(remaining.values())
        if metric_total <= 0:
            raise ValueError(f"{metric} must be positive for weighted constituents")
        proposed = {code: remaining_weight * metric_value / metric_total for code, metric_value in remaining.items()}
        if cap is None:
            weights.update(proposed)
            break
        capped_codes = [code for code, weight in proposed.items() if weight > cap + 1e-12]
        if not capped_codes:
            weights.update(proposed)
            break
        for code in capped_codes:
            weights[code] = cap
            remaining_weight -= cap
            del remaining[code]
    return weights


def _validate_unique_security_codes(members: list[Mapping[str, object]]) -> None:
    seen: set[str] = set()
    for member in members:
        code = _security_code(member)
        if code in seen:
            raise ValueError(f"duplicate security_code: {code}")
        seen.add(code)


def _security_code(member: Mapping[str, object]) -> str:
    code = str(member.get("security_code", "")).strip()
    if not code:
        raise ValueError("constituent security_code is required")
    return code


def _float(value: object) -> float:
    number = float(value)
    if number <= 0:
        raise ValueError("numeric methodology inputs must be positive")
    return number


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _float(value)


def _int(value: object) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError("bucket count must be positive")
    return number


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate FnGuide target weights from engine-ready methodology specs.")
    parser.add_argument("--inputs", default=paths.FNGUIDE_ENGINE_INPUTS_JSON.as_posix())
    parser.add_argument("--specs", default=paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_ENGINE_OUTPUT_DIR.as_posix())
    parser.add_argument("--write-requirements", action="store_true")
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--write-replication-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_replication_report:
        json_path, md_path = write_methodology_replication_report(Path(args.specs), Path(args.output_dir))
        print(f"wrote {json_path} and {md_path}")
        return 0
    if args.write_template:
        output_path = write_engine_input_template(Path(args.specs), Path(args.output_dir))
    elif args.write_requirements:
        output_path = write_engine_input_requirements(Path(args.specs), Path(args.output_dir))
    else:
        output_path = write_target_weights(Path(args.inputs), Path(args.specs), Path(args.output_dir))
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
