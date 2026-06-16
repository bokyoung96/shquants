from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths
from etfs.fnguide.methodology_extraction import ExtractedField, MethodologyExtraction


@dataclass(frozen=True, slots=True)
class DraftMethodologySpec:
    index_code: str
    index_name: str
    provider: str
    products: list[dict[str, object]]
    status: str
    source: dict[str, object]
    rebalance: dict[str, object]
    selection: dict[str, object]
    weighting: dict[str, object]
    evidence_fields: dict[str, object]
    open_questions: list[str]


@dataclass(frozen=True, slots=True)
class MethodologySpec:
    index_code: str
    index_name: str
    provider: str
    products: list[dict[str, object]]
    status: str
    source: dict[str, object]
    rebalance: dict[str, object]
    selection: dict[str, object]
    weighting: dict[str, object]
    validation: dict[str, object]
    review: dict[str, object]
    evidence_fields: dict[str, object]
    open_questions: list[str]


def load_methodology_extractions(path: Path) -> list[MethodologyExtraction]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"unsupported methodology extraction payload: {path}")
    return [_extraction_from_mapping(item) for item in payload["items"]]


def build_draft_specs(extractions: Iterable[MethodologyExtraction]) -> list[DraftMethodologySpec]:
    specs_by_key: dict[str, DraftMethodologySpec] = {}
    products_by_key: dict[str, list[dict[str, object]]] = {}
    for extraction in extractions:
        key = extraction.index_code or extraction.index_name
        products_by_key.setdefault(key, []).append(_product_from_extraction(extraction))
        if key in specs_by_key:
            continue
        specs_by_key[key] = _build_draft_spec(extraction, products=[])
    return [
        _replace_products(spec, _dedupe_products(products_by_key.get(key, [])))
        for key, spec in specs_by_key.items()
    ]


def write_draft_specs(extractions_path: Path, output_dir: Path) -> Path:
    specs = build_draft_specs(load_methodology_extractions(extractions_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "draft_specs.json"
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "fnguide",
        "count": len(specs),
        "indices": [asdict(spec) for spec in specs],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def load_draft_specs(path: Path) -> list[DraftMethodologySpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("indices"), list):
        raise ValueError(f"unsupported draft spec payload: {path}")
    return [_draft_spec_from_mapping(item) for item in payload["indices"]]


def load_spec_overrides(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("indices"), list):
        raise ValueError(f"unsupported spec override payload: {path}")
    return [dict(_mapping(item)) for item in payload["indices"]]


def apply_spec_overrides(
    draft_specs: Iterable[DraftMethodologySpec],
    overrides: Iterable[Mapping[str, object]],
) -> list[MethodologySpec]:
    overrides_by_code = {str(item.get("index_code", "")): item for item in overrides}
    specs: list[MethodologySpec] = []
    for draft in draft_specs:
        override = overrides_by_code.get(draft.index_code, {})
        override_payload = dict(_mapping(override.get("overrides")))
        specs.append(
            MethodologySpec(
                index_code=draft.index_code,
                index_name=draft.index_name,
                provider=draft.provider,
                products=draft.products,
                status=str(override.get("status", draft.status)),
                source=_deep_merge(draft.source, _mapping(override_payload.get("source"))),
                rebalance=_deep_merge(draft.rebalance, _mapping(override_payload.get("rebalance"))),
                selection=_deep_merge(draft.selection, _mapping(override_payload.get("selection"))),
                weighting=_deep_merge(draft.weighting, _mapping(override_payload.get("weighting"))),
                validation=dict(_mapping(override_payload.get("validation"))),
                review=dict(_mapping(override.get("review"))),
                evidence_fields=draft.evidence_fields,
                open_questions=[str(item) for item in override.get("open_questions", draft.open_questions)],
            )
        )
    return specs


def write_methodology_specs(
    draft_specs_path: Path,
    output_dir: Path,
    *,
    overrides_path: Path | None = None,
) -> Path:
    specs = apply_spec_overrides(
        load_draft_specs(draft_specs_path),
        load_spec_overrides(overrides_path or output_dir / "spec_overrides.json"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "methodology_specs.json"
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "fnguide",
        "count": len(specs),
        "indices": [asdict(spec) for spec in specs],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path


def _build_draft_spec(extraction: MethodologyExtraction, *, products: list[dict[str, object]]) -> DraftMethodologySpec:
    fields = extraction.fields
    return DraftMethodologySpec(
        index_code=extraction.index_code,
        index_name=extraction.index_name,
        provider=extraction.provider,
        products=products,
        status="draft_extracted",
        source=extraction.source,
        rebalance=_rebalance_from_fields(fields),
        selection=_selection_from_fields(fields),
        weighting=_weighting_from_fields(fields),
        evidence_fields={name: _field_value(field) for name, field in sorted(fields.items())},
        open_questions=list(extraction.open_questions),
    )


def _replace_products(spec: DraftMethodologySpec, products: list[dict[str, object]]) -> DraftMethodologySpec:
    return DraftMethodologySpec(
        index_code=spec.index_code,
        index_name=spec.index_name,
        provider=spec.provider,
        products=products,
        status=spec.status,
        source=spec.source,
        rebalance=spec.rebalance,
        selection=spec.selection,
        weighting=spec.weighting,
        evidence_fields=spec.evidence_fields,
        open_questions=spec.open_questions,
    )


def _product_from_extraction(extraction: MethodologyExtraction) -> dict[str, object]:
    return {"etf_code": extraction.etf_code, "etf_name": extraction.etf_name}


def _dedupe_products(products: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[str] = set()
    for product in products:
        key = str(product.get("etf_code", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"etf_code": key, "etf_name": str(product.get("etf_name", ""))})
    return deduped


def _rebalance_from_fields(fields: Mapping[str, ExtractedField]) -> dict[str, object]:
    return {
        "frequency": _value(fields, "rebalance.frequency", ""),
        "implementation_months": _value(fields, "rebalance.implementation_months", []),
        "implementation_timing": _value(fields, "rebalance.implementation_timing", ""),
    }


def _selection_from_fields(fields: Mapping[str, ExtractedField]) -> dict[str, object]:
    total = _value(fields, "selection.total_constituents", None)
    min_constituents = _value(fields, "selection.min_constituents", None)
    max_constituents = _value(fields, "selection.max_constituents", None)
    variable_count = _variable_count_from_fields(fields)
    if total is None:
        selection = {"total_constituents": None, "buckets": []}
        if variable_count:
            selection["variable_count"] = variable_count
        if min_constituents is not None:
            selection["min_constituents"] = min_constituents
        if max_constituents is not None:
            selection["max_constituents"] = max_constituents
        return selection
    if "selection.buckets.top2.count" not in fields and "selection.buckets.top2.weight" not in fields:
        selection = {"total_constituents": total, "buckets": []}
        if variable_count:
            selection["variable_count"] = variable_count
        if min_constituents is not None:
            selection["min_constituents"] = min_constituents
        if max_constituents is not None:
            selection["max_constituents"] = max_constituents
        return selection
    return {
        "total_constituents": total,
        "buckets": [
            {
                "name": "top2",
                "count": _value(fields, "selection.buckets.top2.count", 2),
                "universe_filter": {"fics_sector": "semiconductor"},
                "rank": [{"field": "market_cap", "direction": "desc"}],
                "weight": {"type": "fixed", "value": _value(fields, "selection.buckets.top2.weight", 0.25)},
            },
            {
                "name": "momentum",
                "count": _value(fields, "selection.buckets.momentum.count", None),
                "exclude_prior_buckets": True,
                "score": {
                    "type": "sum",
                    "components": [
                        {"field": "sales_momentum", "weight": 1.0},
                        {"field": "price_momentum", "weight": 1.0},
                    ],
                },
                "rank": [{"field": "composite_score", "direction": "desc"}],
            },
            {
                "name": "market_cap_fill",
                "count": _value(fields, "selection.buckets.market_cap_fill.count", None),
                "exclude_prior_buckets": True,
                "rank": [{"field": "market_cap", "direction": "desc"}],
            },
        ],
    }


def _variable_count_from_fields(fields: Mapping[str, ExtractedField]) -> dict[str, object]:
    method = _value(fields, "selection.variable_count.method", None)
    threshold = _value(fields, "selection.variable_count.threshold", None)
    if method is None:
        return {}
    variable_count = {"method": method}
    if threshold is not None:
        variable_count["threshold"] = threshold
    return variable_count


def _weighting_from_fields(fields: Mapping[str, ExtractedField]) -> dict[str, object]:
    security_cap = _value(fields, "weighting.security_cap", None)
    if "selection.buckets.top2.count" not in fields and "selection.buckets.top2.weight" not in fields:
        weighting = {
            "base": _value(fields, "weighting.base", ""),
            "residual": {},
        }
        if security_cap is not None:
            weighting["security_cap"] = security_cap
        return weighting
    top2_count = _value(fields, "selection.buckets.top2.count", 0)
    top2_weight = _value(fields, "selection.buckets.top2.weight", 0.0)
    return {
        "base": _value(fields, "weighting.base", ""),
        "residual": {
            "applies_to_buckets": ["momentum", "market_cap_fill"],
            "total_weight": 1.0 - top2_count * top2_weight,
            "base": "float_market_cap",
            "cap": _value(fields, "weighting.residual.cap", None),
            "redistribution": "iterative_pro_rata",
        },
    }


def _value(fields: Mapping[str, ExtractedField], name: str, default: object) -> object:
    field = fields.get(name)
    return field.value if field else default


def _field_value(field: ExtractedField) -> dict[str, object]:
    return {
        "value": field.value,
        "confidence": field.confidence,
        "evidence": [asdict(item) for item in field.evidence],
    }


def _extraction_from_mapping(item: Mapping[str, object]) -> MethodologyExtraction:
    fields = {
        name: ExtractedField(
            value=_mapping(value).get("value"),
            confidence=str(_mapping(value).get("confidence", "")),
            evidence=[],
        )
        for name, value in _mapping(item.get("fields")).items()
    }
    return MethodologyExtraction(
        etf_code=str(item.get("etf_code", "")),
        etf_name=str(item.get("etf_name", "")),
        index_code=str(item.get("index_code", "")),
        index_name=str(item.get("index_name", "")),
        provider=str(item.get("provider", "")),
        extraction_status=str(item.get("extraction_status", "")),
        source=dict(_mapping(item.get("source"))),
        sections={str(key): dict(_mapping(value)) for key, value in _mapping(item.get("sections")).items()},
        fields=fields,
        open_questions=[str(value) for value in item.get("open_questions", [])],
    )


def _draft_spec_from_mapping(item: Mapping[str, object]) -> DraftMethodologySpec:
    return DraftMethodologySpec(
        index_code=str(item.get("index_code", "")),
        index_name=str(item.get("index_name", "")),
        provider=str(item.get("provider", "")),
        products=[dict(_mapping(value)) for value in item.get("products", [])],
        status=str(item.get("status", "")),
        source=dict(_mapping(item.get("source"))),
        rebalance=dict(_mapping(item.get("rebalance"))),
        selection=dict(_mapping(item.get("selection"))),
        weighting=dict(_mapping(item.get("weighting"))),
        evidence_fields=dict(_mapping(item.get("evidence_fields"))),
        open_questions=[str(value) for value in item.get("open_questions", [])],
    )


def _deep_merge(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(_mapping(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build draft and canonical execution specs from methodology extractions.")
    parser.add_argument("--extractions", default=paths.FNGUIDE_EXTRACTIONS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_EXTRACTION_OUTPUT_DIR.as_posix())
    parser.add_argument("--overrides", default=paths.FNGUIDE_SPEC_OVERRIDES_JSON.as_posix())
    parser.add_argument("--canonical", action="store_true", help="Also write methodology_specs.json with overrides applied.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    draft_path = write_draft_specs(Path(args.extractions), Path(args.output_dir))
    if args.canonical:
        spec_path = write_methodology_specs(draft_path, Path(args.output_dir), overrides_path=Path(args.overrides))
        print(f"wrote {draft_path} and {spec_path}")
    else:
        print(f"wrote {draft_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
