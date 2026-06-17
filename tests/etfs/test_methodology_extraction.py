import json
from pathlib import Path

from etfs.fnguide.methodology_extraction import (
    PdfTextDocument,
    PdfTextPage,
    build_methodology_extractions,
    extract_constituent_count_fields,
    extract_top2_plus_fields,
    extract_weighting_cap_fields,
    write_methodology_extractions,
)
from etfs.fnguide.methodology_specs import (
    apply_spec_overrides,
    build_draft_specs,
    write_draft_specs,
    write_methodology_specs,
)


TOP2_PLUS_TEXT = """
FnGuide AI반도체TOP2플러스 지수
본 지수는 FICS Industry Group 기준 반도체 섹터의 시가총액 상위 2종목을 TOP2로 선정합니다.
유니버스에서 합산스코어(매출액 모멘텀+주가 모멘텀) 상위 4종목을 우선 선정하고,
시가총액 상위 종목 순으로 4종목을 선정하여 최종 10종목으로 지수를 구성합니다.
TOP2 종목의 비중은 각각 25%로 고정하며, 나머지 8종목은 유동시가총액가중
방식으로 15% 실링을 적용합니다.
편입 비중 산정 방법 FnGuide AI반도체TOP2플러스 지수는 유동시가총액 가중방식을
사용하여 비중을 결정합니다. 정기 변경 시 특정 종목의 지수 내 비중이 15%을 넘을
경우 최대 15%로 비중을 제한합니다. 이때, TOP2 종목의 경우 지수 내 비중을 25%로
고정합니다.
"""


def _top2_rule_item(**overrides):
    item = {
        "code": "0167A0",
        "name": "SOL AI반도체TOP2플러스",
        "status": "downloaded",
        "source_url": "https://example.com/top2.pdf",
        "page_url": "https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.KSS",
        "file_path": "etfs/output/methodologies/0167A0.pdf",
        "sha256": "abc123",
        "pdf_page_count": 20,
        "rules": {
            "index_name": "FnGuide AI Semiconductor TOP2 Plus Index",
            "updated": "January 2026",
            "methodology_family": "keyword_theme",
            "review_frequency": "quarterly",
            "review_months": [1, 4, 7, 10],
            "rebalance_timing": "2영업일째(D+2)에 정기변경을 수행합니다",
            "selection_count": 2,
            "weighting_scheme": "float_market_cap_weighted",
            "weight_cap": "15%",
            "has_free_float_adjustment": True,
            "has_market_cap_screen": True,
            "has_liquidity_screen": True,
            "has_keyword_filter": True,
            "has_fics_filter": True,
            "evidence": {
                "universe": TOP2_PLUS_TEXT,
                "weighting": TOP2_PLUS_TEXT,
                "schedule": "매년 1, 4, 7, 10월 옵션 만기일 이후 2영업일째(D+2)에 정기변경을 수행합니다.",
            },
        },
    }
    item.update(overrides)
    return item


def test_extract_top2_plus_fields_requires_evidence_rich_bucket_values() -> None:
    fields = extract_top2_plus_fields(TOP2_PLUS_TEXT)

    assert fields["selection.total_constituents"].value == 10
    assert fields["selection.buckets.top2.count"].value == 2
    assert fields["selection.buckets.top2.weight"].value == 0.25
    assert fields["selection.buckets.momentum.count"].value == 4
    assert fields["selection.buckets.market_cap_fill.count"].value == 4
    assert fields["weighting.residual.cap"].value == 0.15
    assert all(field.evidence for field in fields.values())


def test_extract_constituent_count_fields_accepts_only_exact_counts() -> None:
    exact_text = (
        "FnGuide 2차전지 산업 지수는 관련 키워드 분석을 통하여 종목별 스코어를 부여하고, "
        "스코어 상위 25종목을 선정하여 스코어와 유동시가총액을 동시에 반영하는 가중방식으로 구성한 지수입니다."
    )
    max_text = "관련도가 높은 최대 15종목을 선정하여 구성한 지수입니다."

    exact = extract_constituent_count_fields(exact_text)
    maximum = extract_constituent_count_fields(max_text)

    assert exact["selection.total_constituents"].value == 25
    assert exact["selection.total_constituents"].evidence[0].text.startswith("FnGuide 2차전지")
    assert "selection.total_constituents" not in maximum
    assert maximum["selection.max_constituents"].value == 15


def test_extract_constituent_count_fields_preserves_min_max_range_without_exact_total() -> None:
    text = (
        "재무건전성 및 유동성 기준을 동시에 만족하는 종목 수가 15개 보다 많을 경우, "
        "유동시가총액을 기준으로 상위 15개 종목을 최종 구성종목으로 선정합니다. "
        "재무건전성 및 유동성 기준을 동시에 만족하는 종목 수가 10개 보다 적을 경우, "
        "추가 편입을 통해 10종목으로 지수를 구성합니다."
    )

    fields = extract_constituent_count_fields(text)

    assert "selection.total_constituents" not in fields
    assert fields["selection.min_constituents"].value == 10
    assert fields["selection.max_constituents"].value == 15


def test_extract_constituent_count_fields_reads_selected_final_count_phrase() -> None:
    fields = extract_constituent_count_fields(
        "최종 구성 종목으로 선정된 24종목 중 1차 종목 선정에 포함된 종목일 경우 단일 종목의 편입 비중을 각각 적용합니다."
    )

    assert fields["selection.total_constituents"].value == 24


def test_extract_constituent_count_fields_reads_high_relevance_selected_count() -> None:
    fields = extract_constituent_count_fields(
        "시가총액과 키워드 유사도 스코어링을 고려한 최종 스코어가 우수한 AI반도체 관련도가 높은 10종목을 선정하여 구성한 지수입니다."
    )

    assert fields["selection.total_constituents"].value == 10


def test_extract_constituent_count_fields_reads_portfolio_composed_count() -> None:
    fields = extract_constituent_count_fields(
        "유동시가총액과 키워드 유사도 스코어링을 고려하여 방산 산업과 관련도가 높은 5종목을 포트폴리오로 구성합니다."
    )

    assert fields["selection.total_constituents"].value == 5


def test_extract_constituent_count_fields_reads_top_rank_selected_count() -> None:
    fields = extract_constituent_count_fields(
        "종목선정일 기준 20영업일 유동시가총액 평균을 내림차순으로 정렬한 후, 상위 10종목을 선정합니다."
    )

    assert fields["selection.total_constituents"].value == 10


def test_extract_constituent_count_fields_reads_explicit_total_count_phrase() -> None:
    fields = extract_constituent_count_fields(
        "종목선정일 기준 조선 기자재 TOP3 유니버스와 이를 제외한 유니버스 내에서 시가총액 상위 7종목을 편입하여 총 10종목을 최종 구성종목으로 선정합니다."
    )

    assert fields["selection.total_constituents"].value == 10


def test_extract_constituent_count_fields_reads_spaced_total_count_phrase() -> None:
    fields = extract_constituent_count_fields(
        "이를 제외한 나머지 종목들을 시가총액 상위 순으로 편입하여 총 10 종 목으로 구성합니다."
    )

    assert fields["selection.total_constituents"].value == 10


def test_extract_constituent_count_fields_reads_cumulative_weight_variable_count() -> None:
    fields = extract_constituent_count_fields(
        "상기 조건을 만족하는 종목을 유니버스로 하여 유동주식비율을 반영한 시가총액 가중 방식으로 계산된 "
        "비중이 누적 편입비중의 95% 이상이 되도록 편입한다. 단, 지수 구성종목 수가 10개 미만인 경우 "
        "유동주식비율을 반영한 시가총액이 큰 순서로 편입하여 10종목으로 지수를 구성한다."
    )

    assert fields["selection.variable_count.method"].value == "cumulative_weight_threshold"
    assert fields["selection.variable_count.threshold"].value == 0.95
    assert fields["selection.min_constituents"].value == 10
    assert "selection.total_constituents" not in fields


def test_extract_constituent_count_fields_reads_first_cumulative_weight_exceedance() -> None:
    fields = extract_constituent_count_fields(
        "비중이 높은 순으로 최종 구성종목 편입을 진행한다. 이 때, 누적 편입비중이 최초로 95%를 초과할 때의 "
        "종목이 마지막 편입 종목이 되며, 나머지 하위 종목들은 미편입된다."
    )

    assert fields["selection.variable_count.method"].value == "cumulative_weight_threshold"
    assert fields["selection.variable_count.threshold"].value == 0.95


def test_extract_weighting_cap_fields_reads_plain_single_security_cap() -> None:
    fields = extract_weighting_cap_fields(
        "정기 변경 시 특정 종목의 지수내 비중이 15%을 넘을 경우 최대 15%로 비중을 제한합니다."
    )

    assert fields["weighting.security_cap"].value == 0.15
    assert fields["weighting.security_cap"].evidence[0].section == "weighting"


def test_extract_weighting_cap_fields_reads_compact_single_security_cap() -> None:
    fields = extract_weighting_cap_fields(
        "정기변경시 특정종목의 지수 내 비중이 25%를 초과할 경우 25%로 비중을 제한한다."
    )

    assert fields["weighting.security_cap"].value == 0.25


def test_extract_weighting_cap_fields_reads_one_security_cap_phrase() -> None:
    fields = extract_weighting_cap_fields(
        "포트폴리오에 선정된 종목들은 유동시총 가중방식으로 편입 비중이 결정되며, 한 종목의 비중이 8%가 넘는 경우 종목의 비중을 8%이하가 되도록 실링을 적용합니다."
    )

    assert fields["weighting.security_cap"].value == 0.08


def test_extract_weighting_cap_fields_ignores_aggregate_industry_caps() -> None:
    fields = extract_weighting_cap_fields(
        "FICS 소분류 기준 반도체 및 관련장비에 속하는 종목 비중의 합이 15%를 넘을 경우 최대 15%로 비중을 제한합니다."
    )

    assert fields == {}


def test_extract_weighting_cap_fields_ignores_tiered_security_caps() -> None:
    fields = extract_weighting_cap_fields(
        "유동시가총액 최초 계산 시, 지수 내 비중이 15%를 초과하는 종목의 경우, 15%로 비중을 제한하며, 그 외 종목에 대해서는 10%로 비중을 제한합니다."
    )

    assert fields == {}


def test_extract_weighting_cap_fields_ignores_tiered_caps_with_section_heading() -> None:
    fields = extract_weighting_cap_fields(
        "2.3 개별 종목의 지수 편입 비중 산정 방법 FnGuide AI반도체 TOP10 지수는 유동시가총액 최초 계산 시, "
        "지수 내 비중이 15%를 초과하는 종목의 경우, 15%로 비중을 제한하며, 그 외 종목에 대해서는 10%로 비중을 제한합니다."
    )

    assert fields == {}


def test_build_methodology_extractions_marks_top2_plus_open_questions() -> None:
    extraction = build_methodology_extractions([_top2_rule_item()], pdf_reader=_pdf_reader)[0]

    assert extraction.index_code == "FI00.WLT.KSS"
    assert extraction.extraction_status == "draft_extracted"
    assert extraction.sections["selection"]["page"] == 4
    assert extraction.fields["selection.total_constituents"].evidence[0].source == "methodology_pdf"
    assert extraction.fields["selection.total_constituents"].evidence[0].page == 4
    assert extraction.fields["selection.total_constituents"].value == 10
    assert "rules.selection_count means top2 bucket count, not total constituents" in extraction.open_questions


def test_build_methodology_extractions_skips_items_without_index_page() -> None:
    missing = {
        "code": "102960",
        "name": "KODEX 기계장비",
        "status": "not_found",
        "page_url": "",
        "file_path": "",
        "rules": {},
    }

    extractions = build_methodology_extractions([missing, _top2_rule_item()], pdf_reader=_pdf_reader)

    assert [extraction.index_code for extraction in extractions] == ["FI00.WLT.KSS"]


def test_build_methodology_extractions_reports_generic_selection_count_mismatch_without_top2_evidence() -> None:
    item = _top2_rule_item(
        name="SOL Infrastructure",
        page_url="https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.DSA",
        rules={
            **_top2_rule_item()["rules"],
            "index_name": "FnGuide AI Semiconductor & Infrastructure Index",
            "selection_count": 4,
            "evidence": {
                "universe": "최종 구성 종목으로 선정된 24종목 중 1차 종목 선정에 포함된 종목일 경우 비중을 다르게 적용합니다.",
                "weighting": "동일 가중 방식으로 비중을 배분합니다.",
                "schedule": "매년 3, 6, 9, 12월 정기변경을 수행합니다.",
            },
        },
    )

    extraction = build_methodology_extractions([item], pdf_reader=_selected_final_count_pdf_reader)[0]

    assert "rules.selection_count differs from PDF total constituents" in extraction.open_questions
    assert "rules.selection_count means top2 bucket count, not total constituents" not in extraction.open_questions


def test_build_methodology_extractions_reads_adjacent_pdf_page_for_selection_evidence() -> None:
    item = _top2_rule_item(
        name="SOL Adjacent",
        page_url="https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.ADJ",
        rules={
            **_top2_rule_item()["rules"],
            "index_name": "FnGuide Adjacent Page Index",
            "selection_count": None,
            "weight_cap": "",
            "evidence": {"universe": "", "weighting": "", "schedule": ""},
        },
    )

    extraction = build_methodology_extractions([item], pdf_reader=_adjacent_count_pdf_reader)[0]

    assert extraction.sections["selection"]["page"] == 3
    assert extraction.fields["selection.total_constituents"].value == 10
    assert extraction.fields["selection.total_constituents"].evidence[0].page == 3


def test_build_methodology_extractions_reads_second_adjacent_pdf_page_for_selection_evidence() -> None:
    item = _top2_rule_item(
        name="SOL Second Adjacent",
        page_url="https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.ADJ2",
        rules={
            **_top2_rule_item()["rules"],
            "index_name": "FnGuide Second Adjacent Page Index",
            "selection_count": None,
            "weight_cap": "",
            "evidence": {"universe": "", "weighting": "", "schedule": ""},
        },
    )

    extraction = build_methodology_extractions([item], pdf_reader=_second_adjacent_count_pdf_reader)[0]

    assert extraction.sections["selection"]["page"] == 3
    assert extraction.fields["selection.total_constituents"].value == 10


def test_build_draft_specs_turns_verified_evidence_into_bucket_spec() -> None:
    extraction = build_methodology_extractions([_top2_rule_item()], pdf_reader=_pdf_reader)[0]

    spec = build_draft_specs([extraction])[0]

    assert spec.index_code == "FI00.WLT.KSS"
    assert spec.status == "draft_extracted"
    assert spec.selection["total_constituents"] == 10
    assert [bucket["name"] for bucket in spec.selection["buckets"]] == ["top2", "momentum", "market_cap_fill"]
    assert spec.selection["buckets"][0]["weight"] == {"type": "fixed", "value": 0.25}
    assert spec.weighting["residual"]["cap"] == 0.15


def test_build_draft_specs_preserves_all_etf_products_for_duplicate_index() -> None:
    items = [
        _top2_rule_item(code="0174J0", name="DAISHIN343 오피스리츠플러스"),
        _top2_rule_item(code="395160", name="KODEX AI반도체TOP2플러스"),
        _top2_rule_item(code="0167A0", name="SOL AI반도체TOP2플러스"),
    ]

    spec = build_draft_specs(build_methodology_extractions(items, pdf_reader=_pdf_reader))[0]

    assert spec.products == [
        {"etf_code": "0174J0", "etf_name": "DAISHIN343 오피스리츠플러스"},
        {"etf_code": "395160", "etf_name": "KODEX AI반도체TOP2플러스"},
        {"etf_code": "0167A0", "etf_name": "SOL AI반도체TOP2플러스"},
    ]


def test_build_draft_specs_does_not_invent_top2_buckets_for_plain_constituent_count() -> None:
    item = _top2_rule_item(
        name="SOL High Dividend",
        page_url="https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.NDV",
        rules={
            **_top2_rule_item()["rules"],
            "index_name": "FnGuide K High Dividend Index",
            "selection_count": None,
            "weight_cap": "",
            "evidence": {
                "universe": "FnGuide K High Dividend Index는 배당수익률 상위 40종목을 선정하여 구성한 지수입니다.",
                "weighting": "유동시가총액가중방식으로 비중을 결정합니다.",
                "schedule": "매년 1, 4, 7, 10월 정기변경을 수행합니다.",
            },
        },
    )

    extraction = build_methodology_extractions([item], pdf_reader=_plain_count_pdf_reader)[0]
    spec = build_draft_specs([extraction])[0]

    assert spec.selection == {"total_constituents": 40, "buckets": []}
    assert spec.weighting["security_cap"] == 0.15
    assert spec.weighting["residual"] == {}


def test_build_draft_specs_preserves_constituent_range_without_exact_total() -> None:
    item = _top2_rule_item(
        name="SOL Shipbuilding",
        page_url="https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.NSH",
        rules={
            **_top2_rule_item()["rules"],
            "index_name": "FnGuide Shipbuilding & Shipping Industry Index",
            "selection_count": None,
            "weight_cap": "15%",
            "evidence": {
                "universe": (
                    "재무건전성 및 유동성 기준을 동시에 만족하는 종목 수가 15개 보다 많을 경우, "
                    "유동시가총액을 기준으로 상위 15개 종목을 최종 구성종목으로 선정합니다. "
                    "재무건전성 및 유동성 기준을 동시에 만족하는 종목 수가 10개 보다 적을 경우, "
                    "추가 편입을 통해 10종목으로 지수를 구성합니다."
                ),
                "weighting": "정기 변경 시 특정 종목의 지수내 비중이 15%을 넘을 경우 최대 15%로 비중을 제한합니다.",
                "schedule": "매년 1, 4, 7, 10월 정기변경을 수행합니다.",
            },
        },
    )

    extraction = build_methodology_extractions([item], pdf_reader=_range_count_pdf_reader)[0]
    spec = build_draft_specs([extraction])[0]

    assert spec.selection == {
        "total_constituents": None,
        "min_constituents": 10,
        "max_constituents": 15,
        "buckets": [],
    }
    assert "selection.total_constituents not extracted with evidence" in spec.open_questions


def test_build_draft_specs_preserves_variable_count_without_total_open_question() -> None:
    item = _top2_rule_item(
        name="SOL Sector Coverage",
        page_url="https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.IAM",
        rules={
            **_top2_rule_item()["rules"],
            "index_name": "FnGuide Sector Coverage Index Series",
            "selection_count": None,
            "weight_cap": "25%",
            "evidence": {
                "universe": (
                    "상기 조건을 만족하는 종목을 유니버스로 하여 유동주식비율을 반영한 시가총액 가중 방식으로 "
                    "계산된 비중이 누적 편입비중의 95% 이상이 되도록 편입한다. 단, 지수 구성종목 수가 "
                    "10개 미만인 경우 유동주식비율을 반영한 시가총액이 큰 순서로 편입하여 10종목으로 지수를 구성한다."
                ),
                "weighting": "정기변경 시 특정종목의 지수 내 비중이 25%를 초과할 경우 25%로 비중을 제한한다.",
                "schedule": "매년 6, 12월 정기변경을 수행합니다.",
            },
        },
    )

    extraction = build_methodology_extractions([item], pdf_reader=_variable_count_pdf_reader)[0]
    spec = build_draft_specs([extraction])[0]

    assert spec.selection["total_constituents"] is None
    assert spec.selection["variable_count"] == {"method": "cumulative_weight_threshold", "threshold": 0.95}
    assert spec.selection["min_constituents"] == 10
    assert "selection.total_constituents not extracted with evidence" not in spec.open_questions


def test_write_extraction_and_draft_spec_files(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"items": [_top2_rule_item()]}, ensure_ascii=False), encoding="utf-8")

    extraction_json, extraction_md = write_methodology_extractions(rules_path, tmp_path, pdf_reader=_pdf_reader)
    draft_json = write_draft_specs(extraction_json, tmp_path)

    assert extraction_json.name == "methodology_extractions.json"
    assert extraction_md.name == "methodology_extractions.md"
    assert draft_json.name == "draft_specs.json"
    assert "selection.total_constituents" in extraction_md.read_text(encoding="utf-8")
    payload = json.loads(draft_json.read_text(encoding="utf-8"))
    assert payload["indices"][0]["selection"]["total_constituents"] == 10


def test_apply_spec_overrides_promotes_canonical_spec_status() -> None:
    extraction = build_methodology_extractions([_top2_rule_item()], pdf_reader=_pdf_reader)[0]
    draft = build_draft_specs([extraction])[0]

    specs = apply_spec_overrides(
        [draft],
        [
            {
                "index_code": "FI00.WLT.KSS",
                "status": "methodology_verified",
                "review": {"reviewed_by": "agent", "evidence": "PDF page 4/6 checked"},
                "open_questions": [],
                "overrides": {
                    "selection": {"total_constituents": 10},
                    "validation": {"required_modes": ["etf_holdings_constituents"]},
                },
            }
        ],
    )

    assert specs[0].status == "methodology_verified"
    assert specs[0].selection["total_constituents"] == 10
    assert specs[0].validation == {"required_modes": ["etf_holdings_constituents"]}
    assert specs[0].review == {"reviewed_by": "agent", "evidence": "PDF page 4/6 checked"}
    assert specs[0].open_questions == []


def test_write_methodology_specs_applies_override_file(tmp_path: Path) -> None:
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"items": [_top2_rule_item()]}, ensure_ascii=False), encoding="utf-8")
    extraction_json, _ = write_methodology_extractions(rules_path, tmp_path, pdf_reader=_pdf_reader)
    draft_json = write_draft_specs(extraction_json, tmp_path)
    overrides_path = tmp_path / "spec_overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "indices": [
                    {
                        "index_code": "FI00.WLT.KSS",
                        "status": "methodology_verified",
                        "overrides": {"validation": {"required_modes": ["etf_holdings_constituents"]}},
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    specs_json = write_methodology_specs(draft_json, tmp_path, overrides_path=overrides_path)

    payload = json.loads(specs_json.read_text(encoding="utf-8"))
    assert specs_json.name == "methodology_specs.json"
    assert payload["indices"][0]["status"] == "methodology_verified"
    assert payload["indices"][0]["validation"]["required_modes"] == ["etf_holdings_constituents"]


def _pdf_reader(path: Path) -> PdfTextDocument:
    return PdfTextDocument(
        path=path,
        pages=(
            PdfTextPage(page_number=1, text="FnGuide AI Semiconductor TOP2 Plus Index Methodology Book"),
            PdfTextPage(page_number=4, text=TOP2_PLUS_TEXT),
            PdfTextPage(
                page_number=6,
                text="매년 1, 4, 7, 10월 옵션 만기일 이후 2영업일째(D+2)에 정기변경을 수행합니다.",
            ),
        ),
    )


def _plain_count_pdf_reader(path: Path) -> PdfTextDocument:
    return PdfTextDocument(
        path=path,
        pages=(
            PdfTextPage(page_number=1, text="FnGuide K High Dividend Index Methodology Book"),
            PdfTextPage(
                page_number=3,
                text=(
                    "FnGuide K High Dividend Index는 배당수익률 상위 40종목을 선정하여 구성한 지수입니다. "
                    "정기 변경 시 특정 종목의 지수내 비중이 15%을 넘을 경우 최대 15%로 비중을 제한합니다."
                ),
            ),
        ),
    )


def _range_count_pdf_reader(path: Path) -> PdfTextDocument:
    return PdfTextDocument(
        path=path,
        pages=(
            PdfTextPage(page_number=1, text="FnGuide Shipbuilding & Shipping Industry Index Methodology Book"),
            PdfTextPage(
                page_number=4,
                text=(
                    "재무건전성 및 유동성 기준을 동시에 만족하는 종목 수가 15개 보다 많을 경우, "
                    "유동시가총액을 기준으로 상위 15개 종목을 최종 구성종목으로 선정합니다. "
                    "재무건전성 및 유동성 기준을 동시에 만족하는 종목 수가 10개 보다 적을 경우, "
                    "추가 편입을 통해 10종목으로 지수를 구성합니다. "
                    "정기 변경 시 특정 종목의 지수내 비중이 15%을 넘을 경우 최대 15%로 비중을 제한합니다."
                ),
            ),
        ),
    )


def _selected_final_count_pdf_reader(path: Path) -> PdfTextDocument:
    return PdfTextDocument(
        path=path,
        pages=(
            PdfTextPage(page_number=1, text="FnGuide AI Semiconductor & Infrastructure Index Methodology Book"),
            PdfTextPage(
                page_number=4,
                text="최종 구성 종목으로 선정된 24종목 중 1차 종목 선정에 포함된 종목일 경우 비중을 다르게 적용합니다.",
            ),
        ),
    )


def _variable_count_pdf_reader(path: Path) -> PdfTextDocument:
    return PdfTextDocument(
        path=path,
        pages=(
            PdfTextPage(page_number=1, text="FnGuide Sector Coverage Index Series Methodology Book"),
            PdfTextPage(
                page_number=11,
                text=(
                    "상기 조건을 만족하는 종목을 유니버스로 하여 유동주식비율을 반영한 시가총액 가중 방식으로 "
                    "계산된 비중이 누적 편입비중의 95% 이상이 되도록 편입한다. 단, 지수 구성종목 수가 "
                    "10개 미만인 경우 유동주식비율을 반영한 시가총액이 큰 순서로 편입하여 10종목으로 지수를 구성한다. "
                    "정기변경 시 특정종목의 지수 내 비중이 25%를 초과할 경우 25%로 비중을 제한한다."
                ),
            ),
        ),
    )


def _adjacent_count_pdf_reader(path: Path) -> PdfTextDocument:
    return PdfTextDocument(
        path=path,
        pages=(
            PdfTextPage(page_number=1, text="FnGuide Adjacent Page Index Methodology Book"),
            PdfTextPage(page_number=3, text="2.2 종목구성 방법 TOP2 유니버스 선정 설명만 있는 페이지입니다."),
            PdfTextPage(page_number=4, text="최종 스코어가 우수한 관련도가 높은 10종목을 선정하여 구성한 지수입니다."),
        ),
    )


def _second_adjacent_count_pdf_reader(path: Path) -> PdfTextDocument:
    return PdfTextDocument(
        path=path,
        pages=(
            PdfTextPage(page_number=1, text="FnGuide Second Adjacent Page Index Methodology Book"),
            PdfTextPage(page_number=3, text="2.2 종목구성 방법 유니버스 선정 설명만 있는 페이지입니다."),
            PdfTextPage(page_number=4, text="2.2.2 구성 종목 선정 배당 필터와 섹터 조건 설명만 있는 페이지입니다."),
            PdfTextPage(page_number=5, text="상위 10종목을 선정합니다."),
        ),
    )
