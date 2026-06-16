import json
from pathlib import Path

from etfs.families import build_family_records, write_family_inventory


def test_build_family_records_marks_fnguide_and_discovery_lanes() -> None:
    universe_items = [
        {"code": "111111", "name": "Fn sector", "is_domestic_sector": True, "reason": "domestic_sector_keyword"},
        {"code": "222222", "name": "Unknown sector", "is_domestic_sector": True, "reason": "domestic_sector_keyword"},
        {"code": "333333", "name": "Foreign ETF", "is_domestic_sector": False, "reason": "foreign_exposure"},
        {"code": "444444", "name": "Fn missing", "is_domestic_sector": True, "reason": "domestic_sector_keyword"},
    ]
    fnguide_items = [
        {"code": "111111", "status": "downloaded", "provider": "fnguide"},
        {"code": "444444", "status": "not_found", "provider": "fnguide"},
    ]

    records = build_family_records(universe_items, fnguide_items)

    assert records[0].coverage_provider == "fnguide"
    assert records[0].product_family == "domestic_sector"
    assert records[0].expansion_lane == "fnguide_reference"
    assert records[0].next_action == "continue_fnguide_data_pipeline"
    assert records[1].coverage_provider == ""
    assert records[1].product_family == "domestic_sector"
    assert records[1].expansion_lane == "provider_discovery"
    assert records[1].next_action == "identify_index_provider_and_methodology_source"
    assert records[2].product_family == "foreign_or_global"
    assert records[2].expansion_lane == "future_product_family"
    assert records[3].next_action == "resolve_fnguide_methodology_gap"


def test_build_family_records_classifies_non_sector_product_families() -> None:
    records = build_family_records(
        [
            {"code": "111111", "name": "Broad", "is_domestic_sector": False, "reason": "broad_market"},
            {"code": "222222", "name": "Bond", "is_domestic_sector": False, "reason": "non_equity_or_derivative"},
            {"code": "333333", "name": "Other", "is_domestic_sector": False, "reason": "no_sector_keyword"},
        ],
        [],
    )

    assert [record.product_family for record in records] == [
        "domestic_broad_market",
        "fixed_income_cash_commodity_or_derivative",
        "other_or_unclassified",
    ]
    assert [record.expansion_lane for record in records] == [
        "future_product_family",
        "future_product_family",
        "future_product_family",
    ]


def test_build_family_records_refines_no_sector_keyword_product_families() -> None:
    records = build_family_records(
        [
            {"code": "111111", "name": "ACE MSCI멕시코(합성)", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "222222", "name": "ACE TDF2030액티브 적격", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "333333", "name": "ACE 고배당주", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "444444", "name": "ACE 리츠부동산인프라액티브", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "555555", "name": "ACE 삼성그룹동일가중", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "666666", "name": "ACE KPOP포커스", "is_domestic_sector": False, "reason": "no_sector_keyword"},
        ],
        [],
    )

    assert [record.product_family for record in records] == [
        "foreign_or_global",
        "asset_allocation_or_tdf",
        "domestic_factor_dividend_or_value",
        "real_estate_or_infrastructure",
        "domestic_group_or_theme",
        "domestic_group_or_theme",
    ]


def test_build_family_records_refines_broad_cash_and_theme_names() -> None:
    records = build_family_records(
        [
            {"code": "111111", "name": "ACE 200TR", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "222222", "name": "KIWOOM KRX100", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "333333", "name": "ACE 코스피", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "444444", "name": "KODEX CD1년금리플러스액티브(합성)", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "555555", "name": "1Q 은액티브", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "666666", "name": "HANARO Fn K-뉴딜디지털플러스", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "777777", "name": "HANARO K-뷰티", "is_domestic_sector": False, "reason": "no_sector_keyword"},
        ],
        [],
    )

    assert [record.product_family for record in records] == [
        "domestic_broad_market",
        "domestic_broad_market",
        "domestic_broad_market",
        "fixed_income_cash_commodity_or_derivative",
        "fixed_income_cash_commodity_or_derivative",
        "domestic_group_or_theme",
        "domestic_group_or_theme",
    ]


def test_build_family_records_refines_factor_allocation_and_derivative_names() -> None:
    records = build_family_records(
        [
            {"code": "111111", "name": "HANARO 200", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "222222", "name": "KODEX TRF3070", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "333333", "name": "KODEX 멀티팩터", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "444444", "name": "KODEX 모멘텀Plus", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "555555", "name": "KODEX 국채선물10년", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "666666", "name": "KODEX 금액티브", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "777777", "name": "KODEX K콘텐츠", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "888888", "name": "KODEX 웹툰&드라마", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "999999", "name": "KIWOOM 독일DAX", "is_domestic_sector": False, "reason": "no_sector_keyword"},
        ],
        [],
    )

    assert [record.product_family for record in records] == [
        "domestic_broad_market",
        "asset_allocation_or_tdf",
        "domestic_factor_dividend_or_value",
        "domestic_factor_dividend_or_value",
        "fixed_income_cash_commodity_or_derivative",
        "fixed_income_cash_commodity_or_derivative",
        "domestic_group_or_theme",
        "domestic_group_or_theme",
        "foreign_or_global",
    ]


def test_build_family_records_refines_remaining_unclassified_names() -> None:
    records = build_family_records(
        [
            {"code": "111111", "name": "KoAct 코스닥액티브", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "222222", "name": "MIDAS 중소형액티브", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "333333", "name": "SOL 국제금", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "444444", "name": "RISE 유로스탁스50(H)", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "555555", "name": "RISE 주식혼합", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "666666", "name": "WON 전단채플러스액티브", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "777777", "name": "TIGER BBIG", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "888888", "name": "UNicorn 포스트IPO액티브", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "999999", "name": "TIGER 로우볼", "is_domestic_sector": False, "reason": "no_sector_keyword"},
        ],
        [],
    )

    assert [record.product_family for record in records] == [
        "domestic_broad_market",
        "domestic_broad_market",
        "fixed_income_cash_commodity_or_derivative",
        "foreign_or_global",
        "asset_allocation_or_tdf",
        "fixed_income_cash_commodity_or_derivative",
        "domestic_group_or_theme",
        "domestic_group_or_theme",
        "domestic_factor_dividend_or_value",
    ]


def test_build_family_records_refines_last_domestic_strategy_names() -> None:
    records = build_family_records(
        [
            {"code": "111111", "name": "RISE 내수주플러스", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "222222", "name": "RISE 동학개미", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "333333", "name": "마이티 다이나믹퀀트액티브", "is_domestic_sector": False, "reason": "no_sector_keyword"},
            {"code": "444444", "name": "에셋플러스 코리아플랫폼액티브", "is_domestic_sector": False, "reason": "no_sector_keyword"},
        ],
        [],
    )

    assert [record.product_family for record in records] == [
        "domestic_group_or_theme",
        "domestic_group_or_theme",
        "domestic_factor_dividend_or_value",
        "domestic_group_or_theme",
    ]


def test_write_family_inventory_uses_simple_file_names(tmp_path: Path) -> None:
    universe_path = tmp_path / "universe.json"
    fnguide_path = tmp_path / "pdfs.json"
    universe_path.write_text(
        json.dumps(
            {
                "items": [
                    {"code": "111111", "name": "Fn sector", "is_domestic_sector": True, "reason": "domestic_sector_keyword"},
                    {"code": "222222", "name": "Unknown sector", "is_domestic_sector": True, "reason": "domestic_sector_keyword"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fnguide_path.write_text(
        json.dumps({"items": [{"code": "111111", "status": "downloaded", "provider": "fnguide"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path, json_path, md_path = write_family_inventory(universe_path, fnguide_path, tmp_path)

    assert [csv_path.name, json_path.name, md_path.name] == ["families.csv", "families.json", "families.md"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["count"] == 2
    assert payload["lane_counts"] == {"fnguide_reference": 1, "provider_discovery": 1}
    assert "provider_discovery" in csv_path.read_text(encoding="utf-8-sig")
    assert "ETF index family inventory" in md_path.read_text(encoding="utf-8")
