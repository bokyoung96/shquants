from etfs.rebalance_research import (
    SearchResult,
    build_queries,
    classify_rebalance_schedule,
    extract_rebalance_excerpt,
    filter_domestic_sector_etfs,
    is_domestic_source_result,
    parse_naver_etf_payload,
    parse_duckduckgo_results,
)


def test_parse_duckduckgo_results_decodes_redirect_links():
    html = """
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Findex.example%2Fmethod.pdf&amp;rut=abc">
      KRX 반도체 지수 방법론
    </a>
    <a class="result__snippet">정기변경 및 구성종목 기준</a>
    """

    results = parse_duckduckgo_results(html)

    assert results[0].title == "KRX 반도체 지수 방법론"
    assert results[0].url == "https://index.example/method.pdf"
    assert "정기변경" in results[0].snippet


def test_extract_rebalance_excerpt_prefers_rebalance_paragraphs():
    text = """
    지수 개요
    이 지수는 국내 반도체 관련 종목으로 구성됩니다.

    정기변경
    구성종목은 매년 2회, 6월과 12월 선물옵션 만기일 다음 매매일에 변경합니다.
    수시변경은 상장폐지, 관리종목 지정 등 특별 사유가 발생하는 경우 실시합니다.

    산출 방법
    유동시가총액 가중방식으로 산출합니다.
    """

    excerpt = extract_rebalance_excerpt(text)

    assert "정기변경" in excerpt
    assert "매년 2회" in excerpt
    assert "수시변경" in excerpt
    assert "유동시가총액" not in excerpt


def test_classify_rebalance_schedule_from_korean_text():
    text = "정기변경은 매년 4회 3, 6, 9, 12월에 실시하며 구성종목을 교체합니다."

    assert classify_rebalance_schedule(text) == "quarterly"


def test_filter_domestic_sector_etfs_excludes_overseas_and_broad_market():
    payload = [
        {"itemcode": "091160", "itemname": "KODEX 반도체"},
        {"itemcode": "102970", "itemname": "KODEX 증권"},
        {"itemcode": "069500", "itemname": "KODEX 200"},
        {"itemcode": "360750", "itemname": "TIGER 미국S&P500"},
        {"itemcode": "364970", "itemname": "TIGER KRX바이오K-뉴딜"},
        {"itemcode": "0087F0", "itemname": "ACE 차이나AI빅테크TOP2+액티브"},
        {"itemcode": "0117L0", "itemname": "KODEX 26-12 금융채(AA-이상)액티브"},
        {"itemcode": "298770", "itemname": "KODEX 한국대만IT프리미어"},
        {"itemcode": "434960", "itemname": "DAISHIN343 K200"},
        {"itemcode": "363580", "itemname": "KODEX 200IT TR"},
        {"itemcode": "284980", "itemname": "RISE 200금융"},
    ]

    names = [row["name"] for row in filter_domestic_sector_etfs(payload)]

    assert names == [
        "KODEX 200IT TR",
        "KODEX 반도체",
        "KODEX 증권",
        "RISE 200금융",
        "TIGER KRX바이오K-뉴딜",
    ]


def test_parse_naver_etf_payload_handles_euc_kr_json():
    payload = '{"result":{"etfItemList":[{"itemcode":"091160","itemname":"KODEX 반도체"}]}}'

    rows = parse_naver_etf_payload(payload.encode("euc-kr"), "EUC-KR")

    assert rows == [{"itemcode": "091160", "itemname": "KODEX 반도체"}]


def test_source_filter_rejects_overseas_search_hits():
    result = SearchResult(
        title="미국 반도체 ETF 리밸런싱",
        url="https://example.com/us-semiconductor-etf",
        snippet="국내 ETF와 미국 ETF 비교",
    )

    assert not is_domestic_source_result(result)


def test_build_queries_stays_korean_for_domestic_pdf_search():
    queries = build_queries("KODEX 반도체")

    assert len(queries) == 3
    assert all("underlying" not in query.lower() for query in queries)
    assert any("투자설명서" in query and "PDF" in query for query in queries)
