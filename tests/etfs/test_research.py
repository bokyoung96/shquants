from etfs.research import (
    EtfClassification,
    EtfListing,
    classify_etf_listing,
    fetch_naver_etf_list,
    filter_domestic_sector_etfs,
    parse_naver_etf_payload,
    write_outputs,
)


def test_parse_naver_etf_payload_prefers_valid_utf8_over_wrong_header_encoding() -> None:
    payload = '{"result":{"etfItemList":[{"itemcode":"091160","itemname":"KODEX 반도체"}]}}'

    rows = parse_naver_etf_payload(payload.encode("utf-8"), "EUC-KR")

    assert rows == [{"itemcode": "091160", "itemname": "KODEX 반도체"}]


def test_parse_naver_etf_payload_handles_euc_kr_json() -> None:
    payload = '{"result":{"etfItemList":[{"itemcode":"091160","itemname":"KODEX 반도체"}]}}'

    rows = parse_naver_etf_payload(payload.encode("euc-kr"), "EUC-KR")

    assert rows == [{"itemcode": "091160", "itemname": "KODEX 반도체"}]


def test_domestic_sector_classifier_keeps_korean_sector_equity_etfs() -> None:
    assert classify_etf_listing(EtfListing(code="091160", name="KODEX 반도체")).is_domestic_sector
    assert classify_etf_listing(EtfListing(code="305720", name="KODEX 2차전지산업")).is_domestic_sector
    assert classify_etf_listing(EtfListing(code="102970", name="KODEX 증권")).is_domestic_sector
    assert classify_etf_listing(EtfListing(code="363580", name="KODEX 200IT TR")).is_domestic_sector
    assert classify_etf_listing(EtfListing(code="284980", name="RISE 200금융")).is_domestic_sector


def test_domestic_sector_classifier_excludes_foreign_bond_and_broad_market_etfs() -> None:
    excluded = [
        EtfListing(code="360750", name="TIGER 미국S&P500"),
        EtfListing(code="069500", name="KODEX 200"),
        EtfListing(code="434960", name="DAISHIN343 K200"),
        EtfListing(code="148020", name="RISE 200"),
        EtfListing(code="0117L0", name="KODEX 26-12 금융채(AA-이상)액티브"),
        EtfListing(code="0131W0", name="1Q 단기특수은행채액티브"),
        EtfListing(code="229200", name="KODEX 코스닥150"),
        EtfListing(code="458730", name="TIGER CD금리투자KIS(합성)"),
        EtfListing(code="0087F0", name="ACE 차이나AI빅테크TOP2+액티브"),
    ]

    assert [classify_etf_listing(item).is_domestic_sector for item in excluded] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_filter_domestic_sector_etfs_returns_normalized_sorted_rows() -> None:
    payload = [
        {"itemcode": "9160", "itemname": "KODEX 반도체"},
        {"itemcode": "102970", "itemname": "KODEX 증권"},
        {"itemcode": "069500", "itemname": "KODEX 200"},
        {"itemcode": "360750", "itemname": "TIGER 미국S&P500"},
        {"itemcode": "363580", "itemname": "KODEX 200IT TR"},
        {"itemcode": "", "itemname": "blank"},
    ]

    rows = filter_domestic_sector_etfs(payload)

    assert rows == [
        {"code": "363580", "name": "KODEX 200IT TR"},
        {"code": "009160", "name": "KODEX 반도체"},
        {"code": "102970", "name": "KODEX 증권"},
    ]


def test_fetch_naver_etf_list_normalizes_api_rows() -> None:
    class FakeResponse:
        content = '{"result":{"etfItemList":[{"itemcode":"9160","itemname":"KODEX 반도체"}]}}'.encode()
        encoding = "utf-8"

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def get(self, url):
            assert "etfItemList" in url
            return FakeResponse()

    assert fetch_naver_etf_list(FakeClient()) == [EtfListing(code="009160", name="KODEX 반도체")]


def test_write_outputs_uses_simple_file_names(tmp_path) -> None:
    listing = EtfListing(code="009160", name="KODEX 반도체")
    classification = EtfClassification(listing=listing, is_domestic_sector=True, reason="domestic_sector_keyword")

    paths = write_outputs([listing], [classification], tmp_path)

    assert [path.name for path in paths] == ["all.csv", "sector.csv", "universe.json"]
