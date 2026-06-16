import json
from pathlib import Path

from etfs.spglobal.methodology import build_spglobal_records, write_spglobal_manifest


def _source_item(**overrides):
    item = {
        "code": "111111",
        "name": "TIGER 미국S&P500",
        "product_family": "foreign_or_global",
        "source_candidate": "sp_global",
        "confidence": "medium",
    }
    item.update(overrides)
    return item


def test_build_spglobal_records_keeps_only_spglobal_candidates() -> None:
    records = build_spglobal_records(
        [
            _source_item(name="TIGER 미국S&P500"),
            _source_item(name="KODEX 미국나스닥100", source_candidate="nasdaq"),
        ]
    )

    assert len(records) == 1
    assert records[0].index_name_candidates == ["S&P 500"]
    assert records[0].methodology_probe == "spglobal_methodology_library"


def test_build_spglobal_records_infers_common_spglobal_underlyings() -> None:
    records = build_spglobal_records(
        [
            _source_item(name="TIGER 미국S&P500동일가중"),
            _source_item(name="HK S&P코리아로우볼"),
            _source_item(name="KODEX 미국S&P500금융"),
            _source_item(name="KODEX 미국S&P500성장주"),
            _source_item(name="KODEX 미국S&P500배당귀족커버드콜"),
            _source_item(name="S&P글로벌인프라"),
            _source_item(name="Unknown S&P"),
        ]
    )

    assert [record.index_name_candidates for record in records] == [
        ["S&P 500 Equal Weight"],
        ["S&P Korea Low Volatility"],
        ["S&P 500 Sector"],
        ["S&P 500 Growth"],
        ["S&P 500 Dividend Aristocrats"],
        ["S&P Global Infrastructure"],
        [],
    ]
    assert "methodology-sp-us-indices.pdf" in records[0].methodology_url_candidates[0]
    assert records[-1].next_action == "manual_spglobal_index_mapping"


def test_build_spglobal_records_infers_remaining_spglobal_patterns() -> None:
    records = build_spglobal_records(
        [
            _source_item(name="HANARO 글로벌럭셔리S&P(합성)"),
            _source_item(name="KODEX 미국S&P바이오(합성)"),
            _source_item(name="KODEX 미국스마트모빌리티S&P"),
            _source_item(name="RISE 미국S&P배당킹"),
            _source_item(name="RISE 미국S&P원유생산기업(합성 H)"),
            _source_item(name="SOL 유럽탄소배출권선물S&P(H)"),
        ]
    )

    assert [record.index_name_candidates for record in records] == [
        ["S&P Global Luxury"],
        ["S&P Biotechnology Select Industry"],
        ["S&P Kensho Smart Transportation"],
        ["S&P Dividend Kings"],
        ["S&P Oil & Gas Exploration & Production Select Industry"],
        ["S&P GSCI Carbon Emission Allowances"],
    ]
    assert all(record.next_action == "probe_spglobal_methodology_library" for record in records)


def test_write_spglobal_manifest_uses_simple_file_names(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(
        json.dumps({"items": [_source_item(name="TIGER 미국S&P500")]}, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path, json_path, md_path = write_spglobal_manifest(sources_path, tmp_path)

    assert [csv_path.name, json_path.name, md_path.name] == ["spglobal.csv", "spglobal.json", "spglobal.md"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "sp_global"
    assert payload["count"] == 1
    assert payload["index_candidate_counts"] == {"S&P 500": 1}
    assert "S&P 500" in csv_path.read_text(encoding="utf-8-sig")
    assert "S&P Global methodology probe manifest" in md_path.read_text(encoding="utf-8")
