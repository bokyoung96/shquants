import json
from pathlib import Path

from etfs.fnguide.index_methodology import (
    IndexMethodologyFactory,
    PdfMethodologyDocument,
    extract_methodology_rules,
    write_index_methodologies,
)
from etfs.fnguide.rules import MethodologyRules


def test_extract_methodology_rules_detects_quarterly_schedule_and_float_cap_weighting() -> None:
    text = """
    FnGuide Sample Sector Index Methodology Book
    Updated April 2026
    2.3 개별 종목의 지수 편입 비중 산정 방법
    구성종목은 유동시가총액 가중방식으로 편입한다. 단, 개별 종목의 비중상한은 25%로 한다.
    2.4 종목 선정일 및 개편 일정
    정기변경은 매년 4회 3월, 6월, 9월 및 12월 선물옵션 만기일 익영업일에 실시한다.
    2.5 수시변경 및 처리방식
    """

    rules = extract_methodology_rules(text)

    assert rules.index_name == "FnGuide Sample Sector Index"
    assert rules.updated == "April 2026"
    assert rules.review_frequency == "quarterly"
    assert rules.review_months == [3, 6, 9, 12]
    assert rules.weighting_scheme == "float_market_cap_weighted"
    assert rules.weight_cap == "25%"
    assert rules.has_free_float_adjustment is True
    assert "schedule" in rules.evidence


def test_extract_methodology_rules_detects_semiannual_equal_weight_and_keyword_filter() -> None:
    text = """
    FnGuide Keyword Theme Index Methodology Book
    Updated November 2025
    2.2 종목구성 방법
    키워드 검색 종목 선정 방법을 사용하며 최근 거래대금과 시가총액 요건을 충족한 종목을 선정한다.
    2.3 개별 종목의 지수 편입 비중 산정 방법
    최종 편입종목은 동일가중 방식으로 산정한다.
    2.4 종목 선정일 및 개편 일정
    정기변경은 매년 2회 6월 및 12월에 수행한다.
    """

    rules = extract_methodology_rules(text)

    assert rules.review_frequency == "semiannual"
    assert rules.review_months == [6, 12]
    assert rules.weighting_scheme == "equal_weighted"
    assert rules.has_keyword_filter is True
    assert rules.has_liquidity_screen is True
    assert rules.has_market_cap_screen is True


def test_extract_methodology_rules_detects_score_weighted_methodology() -> None:
    text = """
    FnGuide Secondary Battery Industry Index Methodology Book
    2.3 개별 종목의 지수 편입 비중 산정 방법
    스코어와 유동주식비율을 반영하여 종목의 비중을 결정한다.
    이때 종목 스코어의 표준화 값을 지수포함가중치로 사용한다.
    """

    rules = extract_methodology_rules(text)

    assert rules.weighting_scheme == "score_weighted"
    assert rules.has_free_float_adjustment is True


def test_extract_methodology_rules_uses_main_section_before_appendix_duplicate() -> None:
    text = (
        "Contents 2.4 종목 선정일 및 개편 일정 5 "
        + ("x " * 1400)
        + "2.4 종목 선정일 및 개편 일정 "
        + "정기변경은 매년 2회 6월 및 12월 선물옵션 만기일 이후에 수행한다. "
        + "2.5 수시변경 및 처리방식 "
        + ("x " * 1400)
        + "Appendix D 개편 일정에 맞춰 개편을 실시합니다."
    )

    rules = extract_methodology_rules(text)

    assert rules.review_frequency == "semiannual"
    assert rules.review_months == [6, 12]


def test_extract_methodology_rules_detects_index_series_and_esg_market_cap_weighting() -> None:
    text = """
    Maekyung FnGuide Index Series Methodology Book
    Updated August 2021
    편입비중의 결정
    지수 시리즈는 아래와 같이 비중을 적용한다.
    Wi = MVt / Sum(MVt)
    MVt : 기준가 기준 시가총액
    정기변경 시 개별 종목의 지수내 편입 비중은 10%를 초과할 수 없다.
    """

    rules = extract_methodology_rules(text)

    assert rules.index_name == "Maekyung FnGuide Index Series"
    assert rules.weighting_scheme == "market_cap_weighted"
    assert rules.weight_cap == "10%"


def test_factory_reads_manifest_records_through_reader(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "code": "123456",
                        "name": "SAMPLE ETF",
                        "status": "downloaded",
                        "source_url": "https://example.com/sample.pdf",
                        "page_url": "https://example.com",
                        "file_path": str(pdf_path),
                        "sha256": "abc",
                        "bytes": 9,
                        "query": "fnindex_catalog",
                        "error": "",
                    },
                    {
                        "code": "999999",
                        "name": "MISSING ETF",
                        "status": "not_found",
                        "source_url": "",
                        "page_url": "",
                        "file_path": "",
                        "sha256": "",
                        "bytes": 0,
                        "query": "",
                        "error": "no candidate",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def reader(path: Path) -> PdfMethodologyDocument:
        assert path == pdf_path
        return PdfMethodologyDocument(
            path=path,
            page_count=3,
            text="FnGuide Sample Index Methodology Book\nUpdated April 2026\n유동시가총액 가중",
        )

    factory = IndexMethodologyFactory.from_manifest(manifest_path, pdf_reader=reader)

    assert factory.get("123456").pdf_page_count == 3
    assert factory.get("123456").rules.weighting_scheme == "float_market_cap_weighted"
    assert factory.get("999999").status == "not_found"
    assert factory.downloaded_count == 1
    assert factory.pdf_read_count == 1


def test_methodology_rules_builds_object_rule_set() -> None:
    rules = MethodologyRules(
        index_name="FnGuide Sample Sector Index",
        updated="April 2026",
        methodology_family="sector_theme",
        review_frequency="semiannual",
        review_months=[6, 12],
        rebalance_timing="D+2 after futures/options expiry",
        selection_count=20,
        weighting_scheme="score_weighted",
        weight_cap="25%",
        has_free_float_adjustment=True,
        has_market_cap_screen=True,
        has_liquidity_screen=True,
        has_keyword_filter=True,
        has_fics_filter=True,
    )

    rule_set = rules.to_rule_set()

    assert rule_set.rebalance.frequency == "semiannual"
    assert rule_set.rebalance.months == (6, 12)
    assert rule_set.rebalance.is_periodic is True
    assert rule_set.weighting.scheme == "score_weighted"
    assert rule_set.weighting.requires_score_data is True
    assert rule_set.weighting.uses_free_float is True
    assert rule_set.weighting.cap_percent == 25.0
    assert rule_set.screening.selection_count == 20
    assert rule_set.screening.requires_external_keyword_data is True


def test_factory_exposes_rule_sets_and_grouping(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    manifest_path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "code": "111111",
                        "name": "SAMPLE ETF",
                        "status": "downloaded",
                        "source_url": "",
                        "page_url": "",
                        "file_path": str(pdf_path),
                        "sha256": "abc",
                        "bytes": 9,
                        "query": "",
                        "error": "",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def reader(path: Path) -> PdfMethodologyDocument:
        return PdfMethodologyDocument(
            path=path,
            page_count=1,
            text="""
            FnGuide Sample Index Methodology Book
            Updated April 2026
            2.3 개별 종목의 지수 편입 비중 산정 방법
            유동시가총액 가중방식으로 비중을 결정한다.
            2.4 종목 선정일 및 개편 일정
            정기변경은 매년 2회 6월 및 12월에 수행한다.
            """,
        )

    factory = IndexMethodologyFactory.from_manifest(manifest_path, pdf_reader=reader)

    assert factory.rule_set("111111").rebalance.months == (6, 12)
    assert factory.rule_set("111111").weighting.scheme == "float_market_cap_weighted"
    assert [item.code for item in factory.by_weighting_scheme("float_market_cap_weighted")] == ["111111"]
    assert [item.code for item in factory.by_review_frequency("semiannual")] == ["111111"]


def test_write_index_methodologies_uses_simple_file_names(tmp_path: Path) -> None:
    factory = IndexMethodologyFactory.from_records(
        [
            {
                "code": "111111",
                "name": "SAMPLE ETF",
                "status": "not_found",
                "source_url": "",
                "page_url": "",
                "file_path": "",
                "sha256": "",
                "bytes": 0,
                "query": "",
                "error": "",
            }
        ],
        pdf_reader=lambda path: PdfMethodologyDocument(path=path, page_count=0, text=""),
    )

    csv_path, json_path = write_index_methodologies(factory, tmp_path)

    assert csv_path.name == "rules.csv"
    assert json_path.name == "rules.json"
