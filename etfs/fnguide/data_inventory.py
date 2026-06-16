from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from etfs import paths
from etfs.fnguide.replication_data import KSS_INDEX_CODE, KSS_INDEX_NAME


def build_kss_data_inventory(
    *,
    specs_path: Path,
    local_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    spec = _require_index_spec(specs_path, index_code=KSS_INDEX_CODE)
    paths = (
        _normalize_local_paths(local_paths, base_dir=specs_path.parent)
        if local_paths is not None
        else _default_local_paths()
    )
    requirements = [
        _local_requirement(
            "price_snapshot",
            path=paths.get("price_snapshot"),
            usage="calculation_input",
            note="Local price history supports direct price-momentum derivation.",
        ),
        _derivable_requirement(
            "price_momentum",
            source_path=paths.get("price_snapshot"),
            usage="calculation_input",
            note="Derive price momentum from the local price snapshot once the rebalance window is fixed.",
        ),
        _local_requirement(
            "float_market_cap_snapshot",
            path=paths.get("float_market_cap_snapshot"),
            usage="calculation_input",
            note="Free-float market cap is needed for top-2 ranking and residual weighting.",
        ),
        _local_requirement(
            "semiconductor_classification_snapshot",
            path=paths.get("sector_classification"),
            usage="calculation_input",
            note="Classification snapshot used to build the semiconductor selection universe; this is not an official constituent list.",
        ),
        _missing_requirement(
            "sales_momentum",
            usage="calculation_input",
            note="Sales momentum input required by the KSS momentum bucket selection.",
        ),
        _missing_requirement(
            "composite_score",
            usage="calculation_input",
            note="Composite score or enough component inputs to calculate it are required for ranking.",
        ),
        _local_requirement(
            "issuer_holdings_snapshot",
            path=paths.get("issuer_holdings_snapshot"),
            usage="validation_evidence",
            note="ETF issuer holdings are useful proxy evidence but cannot prove full index replication on their own.",
            satisfies_full_replication=False,
        ),
        _local_requirement(
            "corporate_actions",
            path=paths.get("corporate_actions"),
            usage="calculation_input",
            note="Corporate action history is needed to keep the security universe and weights aligned through rebalance dates.",
        ),
        _missing_requirement(
            "official_bucket_assignments",
            usage="validation_evidence",
            note="Official bucket assignments validate the calculated bucket output; they are not calculation inputs.",
        ),
        _missing_requirement(
            "official_target_weights",
            usage="validation_evidence",
            note="Official target weights validate calculated weights; they are not required to calculate unknown constituents.",
        ),
    ]
    calculation_readiness = _calculation_readiness(requirements)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_code": KSS_INDEX_CODE,
        "index_name": str(spec.get("index_name") or KSS_INDEX_NAME),
        "product_names": _product_names(spec),
        "tracked_etfs": _tracked_etfs(spec),
        "methodology_status": str(spec.get("status", "")),
        "replication_calculation_readiness": calculation_readiness,
        "replication_proven": False,
        "methodology_summary": _methodology_summary(spec),
        "requirements": requirements,
    }


def build_fnguide_data_inventory(
    *,
    specs_path: Path,
    local_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    specs = _load_specs(specs_path)
    items: list[dict[str, object]] = []
    for spec in specs:
        index_code = str(spec.get("index_code", ""))
        if index_code == KSS_INDEX_CODE:
            kss_inventory = build_kss_data_inventory(specs_path=specs_path, local_paths=local_paths)
            items.append({**kss_inventory, "status": str(spec.get("status", ""))})
            continue
        items.append(_generic_index_inventory(spec))
    readiness_counts = Counter(str(item["replication_calculation_readiness"]) for item in items)
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "fnguide",
        "count": len(items),
        "counts": {
            "indices": len(items),
            "by_calculation_readiness": dict(sorted(readiness_counts.items())),
        },
        "indices": items,
    }


def write_kss_data_inventory(
    output_dir: Path,
    *,
    specs_path: Path = paths.FNGUIDE_METHODOLOGY_SPECS_JSON,
    local_paths: Mapping[str, Path] | None = None,
) -> tuple[Path, Path]:
    payload = build_kss_data_inventory(specs_path=specs_path, local_paths=local_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / paths.FNGUIDE_KSS_DATA_INVENTORY_JSON.name
    markdown_path = output_dir / paths.FNGUIDE_KSS_DATA_INVENTORY_MD.name
    _write_json(json_path, payload)
    markdown_path.write_text(_render_kss_data_inventory_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def write_fnguide_data_inventory(
    output_dir: Path,
    *,
    specs_path: Path = paths.FNGUIDE_METHODOLOGY_SPECS_JSON,
    local_paths: Mapping[str, Path] | None = None,
) -> tuple[Path, Path]:
    payload = build_fnguide_data_inventory(specs_path=specs_path, local_paths=local_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / paths.FNGUIDE_DATA_INVENTORY_JSON.name
    markdown_path = output_dir / paths.FNGUIDE_DATA_INVENTORY_MD.name
    _write_json(json_path, payload)
    markdown_path.write_text(_render_fnguide_data_inventory_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write FnGuide full-replication data inventory reports.")
    parser.add_argument("--specs", default=paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_REPLICATION_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    provider_json, _provider_md = write_fnguide_data_inventory(output_dir, specs_path=Path(args.specs))
    kss_json, _kss_md = write_kss_data_inventory(output_dir, specs_path=Path(args.specs))
    print(f"wrote {provider_json}")
    print(f"wrote {kss_json}")
    return 0


def _require_index_spec(specs_path: Path, *, index_code: str) -> Mapping[str, object]:
    for spec in _load_specs(specs_path):
        if str(spec.get("index_code", "")) == index_code:
            return spec
    raise ValueError(f"index spec not found: {index_code}")


def _load_specs(specs_path: Path) -> list[Mapping[str, object]]:
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("indices"), list):
        raise ValueError(f"unsupported methodology specs payload: {specs_path}")
    return [item for item in payload["indices"] if isinstance(item, Mapping)]


def _product_names(spec: Mapping[str, object]) -> list[str]:
    products = spec.get("products", [])
    if not isinstance(products, list):
        return []
    return [str(product.get("etf_name", "")) for product in products if isinstance(product, Mapping)]


def _tracked_etfs(spec: Mapping[str, object]) -> list[dict[str, str]]:
    products = spec.get("products", [])
    if not isinstance(products, list):
        return []
    return [
        {
            "etf_code": str(product.get("etf_code", "")),
            "etf_name": str(product.get("etf_name", "")),
        }
        for product in products
        if isinstance(product, Mapping)
    ]


def _generic_index_inventory(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "index_code": str(spec.get("index_code", "")),
        "index_name": str(spec.get("index_name", "")),
        "product_names": _product_names(spec),
        "tracked_etfs": _tracked_etfs(spec),
        "status": str(spec.get("status", "")),
        "methodology_status": str(spec.get("status", "")),
        "replication_calculation_readiness": "inventory_required",
        "replication_proven": False,
        "methodology_summary": _methodology_summary(spec),
        "requirements": [],
    }


def _methodology_summary(spec: Mapping[str, object]) -> dict[str, object]:
    return {
        "total_constituents": _selection_total(spec),
        "buckets": _selection_buckets(spec),
        "weighting": spec.get("weighting", {}),
        "rebalance": spec.get("rebalance", {}),
    }


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
        {
            "name": str(bucket.get("name", "")),
            "count": bucket.get("count"),
        }
        for bucket in buckets
        if isinstance(bucket, Mapping)
    ]


def _normalize_local_paths(local_paths: Mapping[str, Path], *, base_dir: Path) -> dict[str, Path]:
    return {
        name: path if path.is_absolute() else base_dir / path
        for name, path in local_paths.items()
    }


def _default_local_paths() -> dict[str, Path]:
    return {
        "price_snapshot": Path("parquet/qw_adj_c.parquet"),
        "float_market_cap_snapshot": Path("parquet/qw_mktcap_flt.parquet"),
        "sector_classification": Path("parquet/qw_wics_sec_big.parquet"),
        "issuer_holdings_snapshot": Path("etfs/validation_A0167A0.xlsx"),
    }


def _local_requirement(
    name: str,
    *,
    path: Path | None,
    usage: str,
    note: str,
    satisfies_full_replication: bool | None = None,
) -> dict[str, object]:
    requirement: dict[str, object] = {
        "name": name,
        "usage": usage,
        "status": "available" if path and path.is_file() else "missing",
        "note": note,
    }
    if satisfies_full_replication is not None:
        requirement["satisfies_full_replication"] = satisfies_full_replication
    return requirement


def _derivable_requirement(name: str, *, source_path: Path | None, usage: str, note: str) -> dict[str, object]:
    return {
        "name": name,
        "usage": usage,
        "status": "derivable" if source_path and source_path.is_file() else "missing",
        "note": note,
    }


def _missing_requirement(name: str, *, usage: str, note: str) -> dict[str, object]:
    return {"name": name, "usage": usage, "status": "missing", "note": note}


def _calculation_readiness(requirements: list[dict[str, object]]) -> str:
    calculation_inputs = [item for item in requirements if item.get("usage") == "calculation_input"]
    if any(str(item.get("status", "")) == "missing" for item in calculation_inputs):
        return "missing_calculation_inputs"
    if any(str(item.get("status", "")) not in {"available", "derivable"} for item in calculation_inputs):
        return "missing_calculation_inputs"
    return "ready"


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_kss_data_inventory_markdown(payload: Mapping[str, object]) -> str:
    title = _markdown_cell(payload.get("index_name", KSS_INDEX_NAME))
    index_code = _markdown_cell(payload.get("index_code", KSS_INDEX_CODE))
    readiness = _markdown_cell(payload.get("replication_calculation_readiness", ""))
    proven = _markdown_cell(payload.get("replication_proven", False))
    methodology_status = _markdown_cell(payload.get("methodology_status", ""))
    requirements = payload.get("requirements", [])
    lines = [
        f"# {title} data inventory",
        "",
        f"- Index: {index_code}",
        f"- Calculation readiness: {readiness}",
        f"- Replication proven: {proven}",
        f"- Methodology status: {methodology_status}",
        "",
        "## Requirements",
        "",
        "| Requirement | Usage | Status | Notes |",
        "| --- | --- | --- | --- |",
    ]
    if isinstance(requirements, list):
        for requirement in requirements:
            if not isinstance(requirement, Mapping):
                continue
            lines.append(
                "| {name} | {usage} | {status} | {note} |".format(
                    name=_markdown_cell(requirement.get("name", "")),
                    usage=_markdown_cell(requirement.get("usage", "")),
                    status=_markdown_cell(requirement.get("status", "")),
                    note=_markdown_cell(requirement.get("note", "")),
                )
            )
    return "\n".join(lines) + "\n"


def _render_fnguide_data_inventory_markdown(payload: Mapping[str, object]) -> str:
    counts = payload.get("counts", {})
    by_calculation_readiness = counts.get("by_calculation_readiness", {}) if isinstance(counts, Mapping) else {}
    indices = payload.get("indices", [])
    lines = [
        "# FnGuide data inventory",
        "",
        f"- Provider: {str(payload.get('provider', ''))}",
        f"- Index count: {counts.get('indices', payload.get('count', 0)) if isinstance(counts, Mapping) else payload.get('count', 0)}",
        "",
        "## Readiness summary",
        "",
    ]
    if isinstance(by_calculation_readiness, Mapping):
        for readiness, count in by_calculation_readiness.items():
            lines.append(f"- {readiness}: {count}")
    lines.extend(
        [
            "",
            "## Indices",
            "",
            "| Index code | Index name | Tracked ETFs | Calculation readiness | Proven | Status |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    if isinstance(indices, list):
        for item in indices:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "| {code} | {name} | {tracked_etfs} | {readiness} | {proven} | {status} |".format(
                    code=_markdown_cell(item.get("index_code", "")),
                    name=_markdown_cell(item.get("index_name", "")),
                    tracked_etfs=_markdown_cell(_tracked_etfs_markdown(item)),
                    readiness=_markdown_cell(item.get("replication_calculation_readiness", "")),
                    proven=_markdown_cell(item.get("replication_proven", False)),
                    status=_markdown_cell(item.get("status", item.get("methodology_status", ""))),
                )
            )
    return "\n".join(lines) + "\n"


def _tracked_etfs_markdown(item: Mapping[str, object]) -> str:
    tracked_etfs = item.get("tracked_etfs", [])
    if not isinstance(tracked_etfs, list):
        return ""
    values = []
    for tracked_etf in tracked_etfs:
        if not isinstance(tracked_etf, Mapping):
            continue
        etf_code = str(tracked_etf.get("etf_code", "")).strip()
        etf_name = str(tracked_etf.get("etf_name", "")).strip()
        values.append(" ".join(value for value in [etf_code, etf_name] if value))
    return "; ".join(values)


def _markdown_cell(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
