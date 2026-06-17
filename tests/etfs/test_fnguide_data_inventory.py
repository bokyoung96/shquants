import json
from pathlib import Path

from etfs.fnguide.data_inventory import build_fnguide_data_inventory, write_fnguide_data_inventory


def test_build_fnguide_data_inventory_treats_all_indices_generically(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps(
            {
                "indices": [
                    {
                        "index_code": "FI00.EXAMPLE.A",
                        "index_name": "Example A",
                        "status": "methodology_verified",
                        "products": [{"etf_code": "111111", "etf_name": "ETF A"}],
                        "selection": {"total_constituents": 2, "buckets": [{"name": "top2", "count": 2}]},
                        "weighting": {"base": "equal_weighted"},
                        "rebalance": {"frequency": "quarterly"},
                    },
                    {
                        "index_code": "FI00.EXAMPLE.B",
                        "index_name": "Example B",
                        "status": "draft_extracted",
                        "products": [{"etf_code": "222222", "etf_name": "ETF B"}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    inventory = build_fnguide_data_inventory(specs_path=specs_path)

    assert inventory["provider"] == "fnguide"
    assert inventory["count"] == 2
    assert inventory["counts"]["by_calculation_readiness"] == {"inventory_required": 2}
    items = {item["index_code"]: item for item in inventory["indices"]}
    assert items["FI00.EXAMPLE.A"]["tracked_etfs"] == [{"etf_code": "111111", "etf_name": "ETF A"}]
    assert items["FI00.EXAMPLE.A"]["replication_calculation_readiness"] == "inventory_required"
    assert items["FI00.EXAMPLE.A"]["replication_proven"] is False
    assert items["FI00.EXAMPLE.A"]["requirements"] == []
    assert items["FI00.EXAMPLE.A"]["methodology_summary"] == {
        "total_constituents": 2,
        "buckets": [{"name": "top2", "count": 2}],
        "weighting": {"base": "equal_weighted"},
        "rebalance": {"frequency": "quarterly"},
    }


def test_write_fnguide_data_inventory_writes_json_and_markdown(tmp_path: Path) -> None:
    specs_path = tmp_path / "methodology_specs.json"
    specs_path.write_text(
        json.dumps(
            {
                "indices": [
                    {
                        "index_code": "FI00.EXAMPLE",
                        "index_name": "Example | Index\nName",
                        "status": "methodology_verified",
                        "products": [{"etf_code": "111111", "etf_name": "ETF | Name"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    json_path, markdown_path = write_fnguide_data_inventory(tmp_path / "artifacts", specs_path=specs_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert json_path.name == "data_inventory.json"
    assert markdown_path.name == "data_inventory.md"
    assert payload["indices"][0]["index_code"] == "FI00.EXAMPLE"
    assert "FI00.EXAMPLE" in markdown
    assert "Example \\| Index Name" in markdown
    assert "111111 ETF \\| Name" in markdown
