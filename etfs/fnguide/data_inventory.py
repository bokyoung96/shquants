from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from etfs import paths


def build_fnguide_data_inventory(*, specs_path: Path) -> dict[str, object]:
    items = [_generic_index_inventory(spec) for spec in _load_specs(specs_path)]
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


def write_fnguide_data_inventory(
    output_dir: Path,
    *,
    specs_path: Path = paths.FNGUIDE_METHODOLOGY_SPECS_JSON,
) -> tuple[Path, Path]:
    payload = build_fnguide_data_inventory(specs_path=specs_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / paths.FNGUIDE_DATA_INVENTORY_JSON.name
    markdown_path = output_dir / paths.FNGUIDE_DATA_INVENTORY_MD.name
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_fnguide_data_inventory_markdown(payload), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write FnGuide methodology data inventory reports.")
    parser.add_argument("--specs", default=paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider_json, _provider_md = write_fnguide_data_inventory(Path(args.output_dir), specs_path=Path(args.specs))
    print(f"wrote {provider_json}")
    return 0


def _load_specs(specs_path: Path) -> list[Mapping[str, object]]:
    payload = json.loads(specs_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("indices"), list):
        raise ValueError(f"unsupported methodology specs payload: {specs_path}")
    return [item for item in payload["indices"] if isinstance(item, Mapping)]


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
        {"etf_code": str(product.get("etf_code", "")), "etf_name": str(product.get("etf_name", ""))}
        for product in products
        if isinstance(product, Mapping)
    ]


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
    return [{"name": str(bucket.get("name", "")), "count": bucket.get("count")} for bucket in buckets if isinstance(bucket, Mapping)]


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
    values: list[str] = []
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
