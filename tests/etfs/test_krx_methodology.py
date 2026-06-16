import json
from pathlib import Path

from etfs.krx.methodology import build_krx_records, write_krx_manifest


def _source_item(**overrides):
    item = {
        "code": "111111",
        "name": "KODEX 200",
        "product_family": "domestic_broad_market",
        "source_candidate": "krx",
        "confidence": "medium",
    }
    item.update(overrides)
    return item


def test_build_krx_records_keeps_only_krx_source_candidates() -> None:
    records = build_krx_records(
        [
            _source_item(name="KODEX 200"),
            _source_item(name="KODEX 반도체", source_candidate="fnguide"),
        ]
    )

    assert len(records) == 1
    assert records[0].code == "111111"
    assert records[0].index_name_candidates == ["KOSPI 200"]
    assert records[0].methodology_probe == "krx_index_data_system"


def test_build_krx_records_infers_common_krx_underlyings() -> None:
    records = build_krx_records(
        [
            _source_item(name="KODEX 200선물인버스2X"),
            _source_item(name="KODEX 코스닥150"),
            _source_item(name="KODEX KRX300"),
            _source_item(name="ACE KRX금현물", product_family="fixed_income_cash_commodity_or_derivative"),
            _source_item(name="KODEX 코스피100"),
            _source_item(name="KODEX KTOP30"),
            _source_item(name="Unknown KRX"),
        ]
    )

    assert [record.index_name_candidates for record in records] == [
        ["KOSPI 200"],
        ["KOSDAQ 150"],
        ["KRX 300"],
        ["KRX Gold Spot"],
        ["KOSPI 100"],
        ["KTOP 30"],
        [],
    ]
    assert records[-1].next_action == "manual_krx_index_mapping"


def test_build_krx_records_infers_remaining_krx_name_patterns() -> None:
    records = build_krx_records(
        [
            _source_item(name="KODEX 200exTOP"),
            _source_item(name="KoAct 코스닥액티브"),
            _source_item(name="MIDAS 중소형액티브"),
            _source_item(name="SOL KRX기후변화솔루션", product_family="domestic_group_or_theme"),
            _source_item(name="SOL 코스닥TOP10"),
        ]
    )

    assert [record.index_name_candidates for record in records] == [
        ["KOSPI 200"],
        ["KOSDAQ"],
        ["KOSPI Small Cap"],
        ["KRX Climate Change Solutions"],
        ["KOSDAQ Top 10"],
    ]


def test_write_krx_manifest_uses_simple_file_names(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(
        json.dumps({"items": [_source_item(name="KODEX 코스닥150")]}, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path, json_path, md_path = write_krx_manifest(sources_path, tmp_path)

    assert [csv_path.name, json_path.name, md_path.name] == ["krx.csv", "krx.json", "krx.md"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "krx"
    assert payload["count"] == 1
    assert payload["index_candidate_counts"] == {"KOSDAQ 150": 1}
    assert "KOSDAQ 150" in csv_path.read_text(encoding="utf-8-sig")
    assert "KRX methodology probe manifest" in md_path.read_text(encoding="utf-8")
