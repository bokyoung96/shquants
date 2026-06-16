import json
from pathlib import Path

from etfs.nasdaq.methodology import build_nasdaq_records, write_nasdaq_manifest


def _source_item(**overrides):
    item = {
        "code": "111111",
        "name": "TIGER 미국나스닥100",
        "product_family": "foreign_or_global",
        "source_candidate": "nasdaq",
        "confidence": "medium",
    }
    item.update(overrides)
    return item


def test_build_nasdaq_records_keeps_only_nasdaq_candidates() -> None:
    records = build_nasdaq_records(
        [
            _source_item(name="TIGER 미국나스닥100"),
            _source_item(name="TIGER 미국S&P500", source_candidate="sp_global"),
        ]
    )

    assert len(records) == 1
    assert records[0].index_name_candidates == ["Nasdaq-100"]
    assert records[0].methodology_probe == "nasdaq_index_methodology_library"


def test_build_nasdaq_records_infers_common_nasdaq_underlyings() -> None:
    records = build_nasdaq_records(
        [
            _source_item(name="TIGER 미국나스닥100"),
            _source_item(name="TIGER 미국나스닥넥스트100"),
            _source_item(name="TIGER 미국나스닥바이오"),
            _source_item(name="KODEX 미국클린에너지나스닥"),
            _source_item(name="KIWOOM 미국방어배당성장나스닥"),
            _source_item(name="TIGER 미국필라델피아반도체나스닥"),
            _source_item(name="Unknown Nasdaq"),
        ]
    )

    assert [record.index_name_candidates for record in records] == [
        ["Nasdaq-100"],
        ["Nasdaq Next Generation 100"],
        ["Nasdaq Biotechnology"],
        ["Nasdaq Clean Edge Green Energy"],
        ["Nasdaq US Dividend Achievers"],
        ["PHLX Semiconductor Sector"],
        [],
    ]
    assert "Methodology_NDX.pdf" in records[0].methodology_url_candidates[0]
    assert records[-1].next_action == "manual_nasdaq_index_mapping"


def test_build_nasdaq_records_maps_generic_nasdaq_growth_and_tech_names_to_nasdaq_100() -> None:
    records = build_nasdaq_records(
        [
            _source_item(name="KODEX 미국나스닥AI테크액티브"),
            _source_item(name="KoAct 미국나스닥성장기업액티브"),
            _source_item(name="KoAct 미국나스닥채권혼합50액티브"),
            _source_item(name="PLUS 미국나스닥테크"),
        ]
    )

    assert [record.index_name_candidates for record in records] == [
        ["Nasdaq-100"],
        ["Nasdaq-100"],
        ["Nasdaq-100"],
        ["Nasdaq-100"],
    ]


def test_write_nasdaq_manifest_uses_simple_file_names(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(
        json.dumps({"items": [_source_item(name="TIGER 미국나스닥100")]}, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path, json_path, md_path = write_nasdaq_manifest(sources_path, tmp_path)

    assert [csv_path.name, json_path.name, md_path.name] == ["nasdaq.csv", "nasdaq.json", "nasdaq.md"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "nasdaq"
    assert payload["count"] == 1
    assert payload["index_candidate_counts"] == {"Nasdaq-100": 1}
    assert "Nasdaq-100" in csv_path.read_text(encoding="utf-8-sig")
    assert "Nasdaq methodology probe manifest" in md_path.read_text(encoding="utf-8")
