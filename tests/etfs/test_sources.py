import json
from pathlib import Path

from etfs.sources import build_source_records, write_source_inventory


def _family_item(**overrides):
    item = {
        "code": "111111",
        "name": "KODEX 200",
        "product_family": "domestic_broad_market",
        "coverage_provider": "",
        "provider_status": "",
        "expansion_lane": "future_product_family",
    }
    item.update(overrides)
    return item


def test_build_source_records_infers_named_methodology_sources() -> None:
    records = build_source_records(
        [
            _family_item(name="TIGER 미국S&P500"),
            _family_item(name="ACE MSCI멕시코(합성)"),
            _family_item(name="KODEX 미국나스닥100"),
            _family_item(name="HANARO Fn K-푸드"),
            _family_item(name="HANARO 원자력iSelect"),
            _family_item(name="TIGER CD금리투자KIS(합성)"),
            _family_item(name="KODEX 코스닥150"),
            _family_item(name="KODEX 반도체", coverage_provider="fnguide", provider_status="downloaded"),
            _family_item(name="Unknown ETF", product_family="other_or_unclassified"),
        ]
    )

    assert [record.source_candidate for record in records] == [
        "sp_global",
        "msci",
        "nasdaq",
        "fnguide",
        "iselect",
        "kis",
        "krx",
        "fnguide",
        "",
    ]
    assert records[0].confidence == "medium"
    assert records[7].confidence == "high"
    assert records[8].next_action == "manual_source_research"


def test_build_source_records_uses_family_probe_fallbacks() -> None:
    records = build_source_records(
        [
            _family_item(name="Theme ETF", product_family="domestic_group_or_theme"),
            _family_item(name="Factor ETF", product_family="domestic_factor_dividend_or_value"),
            _family_item(name="Bond ETF", product_family="fixed_income_cash_commodity_or_derivative"),
            _family_item(name="Global ETF", product_family="foreign_or_global"),
            _family_item(name="TDF ETF", product_family="asset_allocation_or_tdf"),
            _family_item(name="REIT ETF", product_family="real_estate_or_infrastructure"),
            _family_item(name="Broad ETF", product_family="domestic_broad_market"),
        ]
    )

    assert [record.source_candidate for record in records] == [
        "issuer_or_domestic_index_provider",
        "issuer_or_domestic_index_provider",
        "fixed_income_or_commodity_provider",
        "global_index_provider",
        "issuer_or_domestic_index_provider",
        "issuer_or_domestic_index_provider",
        "krx",
    ]
    assert all(record.confidence == "low" for record in records)
    assert records[0].next_action == "research_issuer_or_domestic_index_provider_sources"


def test_write_source_inventory_uses_simple_file_names(tmp_path: Path) -> None:
    families_path = tmp_path / "families.json"
    families_path.write_text(
        json.dumps(
            {
                "items": [
                    _family_item(name="KODEX 코스닥150"),
                    _family_item(name="Unknown ETF", product_family="other_or_unclassified"),
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_path, json_path, md_path = write_source_inventory(families_path, tmp_path)

    assert [csv_path.name, json_path.name, md_path.name] == ["sources.csv", "sources.json", "sources.md"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["count"] == 2
    assert payload["source_counts"] == {"krx": 1, "unknown": 1}
    assert "KODEX 코스닥150" in csv_path.read_text(encoding="utf-8-sig")
    assert "ETF methodology source candidates" in md_path.read_text(encoding="utf-8")
