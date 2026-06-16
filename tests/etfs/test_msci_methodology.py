import json
from pathlib import Path

from etfs.msci.methodology import build_msci_records, write_msci_manifest


def _source_item(**overrides):
    item = {
        "code": "111111",
        "name": "KODEX MSCI Korea",
        "product_family": "foreign_or_global",
        "source_candidate": "msci",
        "confidence": "medium",
    }
    item.update(overrides)
    return item


def test_build_msci_records_keeps_only_msci_candidates() -> None:
    records = build_msci_records(
        [
            _source_item(name="KODEX MSCI Korea"),
            _source_item(name="TIGER 미국S&P500", source_candidate="sp_global"),
        ]
    )

    assert len(records) == 1
    assert records[0].index_name_candidates == ["MSCI Korea"]
    assert records[0].methodology_probe == "msci_methodology_library"


def test_build_msci_records_infers_common_msci_underlyings() -> None:
    records = build_msci_records(
        [
            _source_item(name="ACE MSCI멕시코(합성)"),
            _source_item(name="ACE MSCI인도네시아(합성)"),
            _source_item(name="ACE MSCI필리핀(합성)"),
            _source_item(name="ACE 러시아MSCI(합성)"),
            _source_item(name="KODEX MSCI Korea TR"),
            _source_item(name="KODEX MSCI KOREA ESG유니버설"),
            _source_item(name="TIGER MSCI KOREA ESG리더스"),
            _source_item(name="KODEX MSCI선진국"),
            _source_item(name="PLUS 신흥국MSCI(합성 H)"),
            _source_item(name="TIGER 미국MSCI리츠(합성 H)"),
            _source_item(name="HANARO 글로벌워터MSCI(합성)"),
            _source_item(name="KODEX 차이나2차전지MSCI(합성)"),
            _source_item(name="Unknown MSCI"),
        ]
    )

    assert [record.index_name_candidates for record in records] == [
        ["MSCI Mexico"],
        ["MSCI Indonesia"],
        ["MSCI Philippines"],
        ["MSCI Russia"],
        ["MSCI Korea"],
        ["MSCI Korea ESG Universal"],
        ["MSCI Korea ESG Leaders"],
        ["MSCI World"],
        ["MSCI Emerging Markets"],
        ["MSCI US REIT"],
        ["MSCI ACWI IMI Water ESG Filtered"],
        ["MSCI China"],
        [],
    ]
    assert records[0].methodology_url_candidates[0].startswith("https://www.msci.com/")
    assert records[-1].next_action == "manual_msci_index_mapping"


def test_build_msci_records_maps_em_futures_and_global_names() -> None:
    records = build_msci_records(
        [
            _source_item(name="KODEX MSCI EM선물(H)"),
            _source_item(name="TIGER 이머징마켓MSCI레버리지(합성 H)"),
            _source_item(name="PLUS 글로벌MSCI(합성 H)"),
            _source_item(name="RISE 중국MSCI China(H)"),
        ]
    )

    assert [record.index_name_candidates for record in records] == [
        ["MSCI Emerging Markets"],
        ["MSCI Emerging Markets"],
        ["MSCI ACWI"],
        ["MSCI China"],
    ]


def test_write_msci_manifest_uses_simple_file_names(tmp_path: Path) -> None:
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(
        json.dumps({"items": [_source_item(name="KODEX MSCI Korea")]}, ensure_ascii=False),
        encoding="utf-8",
    )

    csv_path, json_path, md_path = write_msci_manifest(sources_path, tmp_path)

    assert [csv_path.name, json_path.name, md_path.name] == ["msci.csv", "msci.json", "msci.md"]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "msci"
    assert payload["count"] == 1
    assert payload["index_candidate_counts"] == {"MSCI Korea": 1}
    assert "MSCI Korea" in csv_path.read_text(encoding="utf-8-sig")
    assert "MSCI methodology probe manifest" in md_path.read_text(encoding="utf-8")
