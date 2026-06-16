from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from etfs.fnguide.replication_data import KSS_INDEX_CODE, KSS_INDEX_NAME


def build_kss_data_inventory(
    *,
    specs_path: Path,
    local_paths: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    spec = _require_index_spec(specs_path, index_code=KSS_INDEX_CODE)
    paths = local_paths or {}
    requirements = [
        _local_requirement(
            "price_snapshot",
            path=paths.get("price_snapshot"),
            note="Local price history supports direct price-momentum derivation.",
        ),
        _derivable_requirement(
            "price_momentum",
            source_path=paths.get("price_snapshot"),
            note="Derive price momentum from the local price snapshot once the rebalance window is fixed.",
        ),
        _local_requirement(
            "float_market_cap_snapshot",
            path=paths.get("float_market_cap_snapshot"),
            note="Free-float market cap is needed for top-2 ranking and residual weighting.",
        ),
        _local_requirement(
            "sector_classification",
            path=paths.get("sector_classification"),
            note="Local sector labels can support semiconductor screening, but not provider theme confirmation.",
        ),
        _external_requirement(
            "theme_membership",
            note="Full replication still needs FnGuide or provider-confirmed theme membership evidence.",
        ),
        _external_requirement(
            "sales_momentum",
            note="The methodology depends on official sales-momentum inputs that are not inferable from current local files.",
        ),
        _external_requirement(
            "composite_score",
            note="Composite ranking inputs remain provider-controlled unless the official scoring formula and values are supplied.",
        ),
        _local_requirement(
            "issuer_holdings_snapshot",
            path=paths.get("issuer_holdings_snapshot"),
            note="ETF issuer holdings are useful proxy evidence but cannot prove full index replication on their own.",
            satisfies_full_replication=False,
        ),
        _local_requirement(
            "corporate_actions",
            path=paths.get("corporate_actions"),
            note="Corporate action history is needed to keep the security universe and weights aligned through rebalance dates.",
        ),
        _external_requirement(
            "official_bucket_assignments",
            note="Official bucket assignments are required to confirm which names land in top-2, momentum, and fill buckets.",
        ),
        _external_requirement(
            "official_target_weights",
            note="Official target weights remain the primary replication and validation evidence for full fidelity.",
        ),
    ]
    return {
        "schema_version": "1.0",
        "index_code": KSS_INDEX_CODE,
        "index_name": str(spec.get("index_name") or KSS_INDEX_NAME),
        "product_names": _product_names(spec),
        "replication_readiness": _replication_readiness(requirements),
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
            items.append(
                {
                    "index_code": kss_inventory["index_code"],
                    "index_name": kss_inventory["index_name"],
                    "product_names": kss_inventory["product_names"],
                    "status": str(spec.get("status", "")),
                    "replication_readiness": kss_inventory["replication_readiness"],
                    "requirements": kss_inventory["requirements"],
                }
            )
            continue
        items.append(
            {
                "index_code": index_code,
                "index_name": str(spec.get("index_name", "")),
                "product_names": _product_names(spec),
                "status": str(spec.get("status", "")),
                "replication_readiness": "inventory_required",
            }
        )
    return {
        "schema_version": "1.0",
        "provider": "fnguide",
        "count": len(items),
        "indices": items,
    }


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


def _local_requirement(
    name: str,
    *,
    path: Path | None,
    note: str,
    satisfies_full_replication: bool | None = None,
) -> dict[str, object]:
    requirement: dict[str, object] = {
        "name": name,
        "status": "available" if path and path.exists() else "missing",
        "note": note,
    }
    if satisfies_full_replication is not None:
        requirement["satisfies_full_replication"] = satisfies_full_replication
    return requirement


def _derivable_requirement(name: str, *, source_path: Path | None, note: str) -> dict[str, object]:
    return {
        "name": name,
        "status": "derivable" if source_path and source_path.exists() else "missing",
        "note": note,
    }


def _external_requirement(name: str, *, note: str) -> dict[str, object]:
    return {"name": name, "status": "external_required", "note": note}


def _replication_readiness(requirements: list[dict[str, object]]) -> str:
    blocking_statuses = {"missing", "external_required"}
    if any(str(item.get("status", "")) in blocking_statuses for item in requirements):
        return "missing_required_data"
    if any(str(item.get("status", "")) not in {"available", "derivable"} for item in requirements):
        return "missing_required_data"
    return "ready"
