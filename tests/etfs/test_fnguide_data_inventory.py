import json
from pathlib import Path

from etfs.fnguide.data_inventory import (
    build_fnguide_data_inventory,
    build_kss_data_inventory,
)


def test_build_kss_data_inventory_marks_local_and_external_requirements(tmp_path: Path) -> None:
    specs_path = _write_specs(
        tmp_path / "methodology_specs.json",
        indices=[
            {
                "index_code": "FI00.WLT.KSS",
                "index_name": "FnGuide AI Semiconductor TOP2 Plus Index",
                "provider": "fnguide",
                "products": [{"etf_code": "466920", "etf_name": "SOL AI Semiconductor ETF"}],
                "status": "methodology_verified",
                "source": {},
                "rebalance": {},
                "selection": {},
                "weighting": {},
                "validation": {},
            },
            {
                "index_code": "FI00.OTHER",
                "index_name": "FnGuide Other Index",
                "provider": "fnguide",
                "products": [{"etf_code": "000001", "etf_name": "Other ETF"}],
                "status": "methodology_verified",
                "source": {},
                "rebalance": {},
                "selection": {},
                "weighting": {},
                "validation": {},
            },
        ],
    )
    local_paths = {
        "price_snapshot": _touch(tmp_path / "price_snapshot.csv"),
        "float_market_cap_snapshot": _touch(tmp_path / "float_market_cap_snapshot.csv"),
        "sector_classification": _touch(tmp_path / "sector_classification.csv"),
        "issuer_holdings_snapshot": _touch(tmp_path / "issuer_holdings_snapshot.csv"),
    }

    inventory = build_kss_data_inventory(specs_path=specs_path, local_paths=local_paths)
    requirements = {item["name"]: item for item in inventory["requirements"]}

    assert inventory["schema_version"] == "1.0"
    assert inventory["index_code"] == "FI00.WLT.KSS"
    assert inventory["replication_readiness"] == "missing_required_data"
    assert inventory["product_names"] == ["SOL AI Semiconductor ETF"]
    assert requirements["price_snapshot"]["status"] == "available"
    assert requirements["float_market_cap_snapshot"]["status"] == "available"
    assert requirements["sector_classification"]["status"] == "available"
    assert requirements["issuer_holdings_snapshot"]["status"] == "available"
    assert requirements["price_momentum"]["status"] == "derivable"
    assert requirements["theme_membership"]["status"] == "external_required"
    assert requirements["sales_momentum"]["status"] == "external_required"
    assert requirements["composite_score"]["status"] == "external_required"
    assert requirements["official_bucket_assignments"]["status"] == "external_required"
    assert requirements["official_target_weights"]["status"] == "external_required"
    assert requirements["corporate_actions"]["status"] == "missing"
    assert requirements["issuer_holdings_snapshot"]["satisfies_full_replication"] is False


def test_build_fnguide_data_inventory_includes_kss_and_other_indices(tmp_path: Path) -> None:
    specs_path = _write_specs(
        tmp_path / "methodology_specs.json",
        indices=[
            {
                "index_code": "FI00.WLT.KSS",
                "index_name": "FnGuide AI Semiconductor TOP2 Plus Index",
                "provider": "fnguide",
                "products": [{"etf_code": "466920", "etf_name": "SOL AI Semiconductor ETF"}],
                "status": "methodology_verified",
                "source": {},
                "rebalance": {},
                "selection": {},
                "weighting": {},
                "validation": {},
            },
            {
                "index_code": "FI00.OTHER",
                "index_name": "FnGuide Other Index",
                "provider": "fnguide",
                "products": [{"etf_code": "000001", "etf_name": "Other ETF"}],
                "status": "draft_extracted",
                "source": {},
                "rebalance": {},
                "selection": {},
                "weighting": {},
                "validation": {},
            },
        ],
    )

    inventory = build_fnguide_data_inventory(specs_path=specs_path)
    items = {item["index_code"]: item for item in inventory["indices"]}

    assert inventory["provider"] == "fnguide"
    assert inventory["schema_version"] == "1.0"
    assert inventory["count"] == 2
    assert items["FI00.WLT.KSS"]["replication_readiness"] == "missing_required_data"
    assert items["FI00.OTHER"]["index_name"] == "FnGuide Other Index"
    assert items["FI00.OTHER"]["product_names"] == ["Other ETF"]
    assert items["FI00.OTHER"]["status"] == "draft_extracted"
    assert items["FI00.OTHER"]["replication_readiness"] in {"inventory_required", "not_audited"}


def _write_specs(path: Path, *, indices: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "provider": "fnguide",
                "count": len(indices),
                "indices": indices,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _touch(path: Path) -> Path:
    path.write_text("", encoding="utf-8")
    return path
