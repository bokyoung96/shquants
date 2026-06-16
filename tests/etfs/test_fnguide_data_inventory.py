import json
from pathlib import Path

from etfs.fnguide.data_inventory import (
    build_fnguide_data_inventory,
    build_kss_data_inventory,
    write_fnguide_data_inventory,
    write_kss_data_inventory,
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
    assert inventory["tracked_etfs"] == [{"etf_code": "466920", "etf_name": "SOL AI Semiconductor ETF"}]
    assert inventory["methodology_status"] == "methodology_verified"
    assert inventory["methodology_summary"] == {
        "total_constituents": None,
        "buckets": [],
        "weighting": {},
        "rebalance": {},
    }
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
    assert inventory["counts"] == {"indices": 2, "by_readiness": {"inventory_required": 1, "missing_required_data": 1}}
    assert items["FI00.WLT.KSS"]["replication_readiness"] == "missing_required_data"
    assert items["FI00.WLT.KSS"]["tracked_etfs"] == [{"etf_code": "466920", "etf_name": "SOL AI Semiconductor ETF"}]
    assert "requirements" in items["FI00.WLT.KSS"]
    assert items["FI00.OTHER"]["index_name"] == "FnGuide Other Index"
    assert items["FI00.OTHER"]["product_names"] == ["Other ETF"]
    assert items["FI00.OTHER"]["tracked_etfs"] == [{"etf_code": "000001", "etf_name": "Other ETF"}]
    assert items["FI00.OTHER"]["status"] == "draft_extracted"
    assert items["FI00.OTHER"]["replication_readiness"] == "inventory_required"
    assert items["FI00.OTHER"]["requirements"] == []


def test_build_kss_data_inventory_resolves_relative_paths_from_specs_directory_and_ignores_directories(
    tmp_path: Path,
    monkeypatch,
) -> None:
    specs_path = _write_specs(
        tmp_path / "methodology_specs.json",
        indices=[
            {
                "index_code": "FI00.WLT.KSS",
                "index_name": "FnGuide AI Semiconductor TOP2 Plus Index",
                "provider": "fnguide",
                "products": [],
                "status": "methodology_verified",
                "source": {},
                "rebalance": {},
                "selection": {},
                "weighting": {},
                "validation": {},
            },
        ],
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _touch(data_dir / "price_snapshot.csv")
    (data_dir / "float_market_cap_snapshot.csv").mkdir()
    monkeypatch.chdir(tmp_path.parent)

    inventory = build_kss_data_inventory(
        specs_path=specs_path,
        local_paths={
            "price_snapshot": Path("data/price_snapshot.csv"),
            "float_market_cap_snapshot": Path("data/float_market_cap_snapshot.csv"),
        },
    )
    requirements = {item["name"]: item for item in inventory["requirements"]}

    assert requirements["price_snapshot"]["status"] == "available"
    assert requirements["price_momentum"]["status"] == "derivable"
    assert requirements["float_market_cap_snapshot"]["status"] == "missing"


def test_write_kss_data_inventory_writes_json_and_markdown(tmp_path: Path) -> None:
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
        ],
    )
    output_dir = tmp_path / "artifacts" / "replication"

    json_path, markdown_path = write_kss_data_inventory(output_dir, specs_path=specs_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert json_path == output_dir / "kss_data_inventory.json"
    assert markdown_path == output_dir / "kss_data_inventory.md"
    assert payload["index_code"] == "FI00.WLT.KSS"
    assert payload["replication_readiness"] == "missing_required_data"
    assert "FI00.WLT.KSS" in markdown
    assert "missing_required_data" in markdown
    assert "official_target_weights" in markdown
    assert "issuer_holdings_snapshot" in markdown


def test_write_kss_data_inventory_escapes_markdown_table_cells(tmp_path: Path) -> None:
    specs_path = _write_specs(
        tmp_path / "methodology_specs.json",
        indices=[
            {
                "index_code": "FI00.WLT.KSS",
                "index_name": "FnGuide AI | Semiconductor\nTOP2 Plus Index",
                "provider": "fnguide",
                "products": [],
                "status": "methodology_verified",
                "source": {},
                "rebalance": {},
                "selection": {},
                "weighting": {},
                "validation": {},
            },
        ],
    )

    _, markdown_path = write_kss_data_inventory(tmp_path / "artifacts", specs_path=specs_path)

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "FnGuide AI \\| Semiconductor TOP2 Plus Index" in markdown


def test_write_fnguide_data_inventory_writes_json_and_markdown(tmp_path: Path) -> None:
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
    output_dir = tmp_path / "artifacts" / "provider"

    json_path, markdown_path = write_fnguide_data_inventory(output_dir, specs_path=specs_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert json_path == output_dir / "data_inventory.json"
    assert markdown_path == output_dir / "data_inventory.md"
    assert payload["counts"]["by_readiness"] == {"inventory_required": 1, "missing_required_data": 1}
    assert "FnGuide" in markdown
    assert "missing_required_data" in markdown
    assert "FI00.WLT.KSS" in markdown
    assert "466920 SOL AI Semiconductor ETF" in markdown


def test_write_fnguide_data_inventory_escapes_markdown_cells_and_tracks_etfs(tmp_path: Path) -> None:
    specs_path = _write_specs(
        tmp_path / "methodology_specs.json",
        indices=[
            {
                "index_code": "FI00.WLT.KSS",
                "index_name": "FnGuide AI | Semiconductor\nTOP2 Plus Index",
                "provider": "fnguide",
                "products": [{"etf_code": "466920", "etf_name": "SOL | AI\nSemiconductor ETF"}],
                "status": "methodology_verified",
                "source": {},
                "rebalance": {},
                "selection": {},
                "weighting": {},
                "validation": {},
            },
        ],
    )

    _, markdown_path = write_fnguide_data_inventory(tmp_path / "artifacts", specs_path=specs_path)

    markdown = markdown_path.read_text(encoding="utf-8")
    assert "FnGuide AI \\| Semiconductor TOP2 Plus Index" in markdown
    assert "466920 SOL \\| AI Semiconductor ETF" in markdown


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
