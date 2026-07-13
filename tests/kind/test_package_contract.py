import re
from dataclasses import FrozenInstanceError

import pytest

from kind import Confidence, Disclosure, MatchResult, ParsedPage
from kind.selectors import (
    COMPANY_LINK_ONCLICK,
    DISCLOSURE_LINK_ONCLICK,
    EXPECTED_CELL_COUNT,
    FORECAST_TITLE_PATTERN,
    FORM_DEFAULTS,
    ISSUER_PATTERN,
    KIND_MAIN_URL,
    KIND_SUB_URL,
    PAGE_PATTERN,
    PARSER_SCHEMA_VERSION,
    PROVISIONAL_TITLE_PATTERN,
    RECEIPT_PATTERN,
    ROW_SELECTOR,
    TABLE_SELECTOR,
    TIME_PATTERN,
)


def _disclosure() -> Disclosure:
    return Disclosure(
        announcement_date="2024-04-25",
        time="08:05",
        company="SK하이닉스",
        title="영업 (잠정) 실적 (공정공시)",
        submitter="SK하이닉스",
        issuer_id="00066",
        receipt_id="20240425000004",
        page=1,
        position=1,
    )


def test_package_root_exports_model_contract() -> None:
    assert [confidence.value for confidence in Confidence] == [
        "EXACT_MATCH",
        "NORMALIZED_MATCH",
        "MULTIPLE_MATCH",
        "NO_MATCH",
    ]
    assert Disclosure.__module__ == "kind.models"
    assert ParsedPage.__module__ == "kind.models"
    assert MatchResult.__module__ == "kind.models"


def test_selector_contract_is_exact() -> None:
    assert KIND_MAIN_URL == (
        "https://kind.krx.co.kr/disclosure/todaydisclosure.do"
        "?method=searchTodayDisclosureMain"
    )
    assert KIND_SUB_URL == "https://kind.krx.co.kr/disclosure/todaydisclosure.do"
    assert PARSER_SCHEMA_VERSION == 1
    assert FORM_DEFAULTS == {
        "method": "searchTodayDisclosureSub",
        "currentPageSize": "100",
        "orderMode": "0",
        "orderStat": "D",
        "marketType": "",
        "forward": "todaydisclosure_sub",
        "searchMode": "",
        "searchCodeType": "",
        "chose": "S",
        "todayFlag": "N",
        "repIsuSrtCd": "",
        "kosdaqSegment": "",
        "searchCorpName": "",
        "copyUrl": "",
    }
    assert TABLE_SELECTOR == "table.list"
    assert ROW_SELECTOR == "tbody tr"
    assert COMPANY_LINK_ONCLICK == "companysummary_open"
    assert DISCLOSURE_LINK_ONCLICK == "openDisclsViewer"
    assert EXPECTED_CELL_COUNT == 5


def test_time_pattern_accepts_only_zero_padded_24_hour_times() -> None:
    assert re.fullmatch(TIME_PATTERN, "08:05")
    assert re.fullmatch(TIME_PATTERN, "23:59")
    assert not re.fullmatch(TIME_PATTERN, "8:05")
    assert not re.fullmatch(TIME_PATTERN, "24:00")


def test_page_pattern_captures_only_ascii_numeric_pages() -> None:
    match = re.fullmatch(PAGE_PATTERN, "fnPageGo('12')")

    assert match is not None
    assert match.group(1) == "12"
    assert not re.fullmatch(PAGE_PATTERN, "fnPageGo('١٢')")
    assert not re.fullmatch(PAGE_PATTERN, "fnPageGo('12a')")


def test_issuer_pattern_accepts_alphanumeric_ids_and_rejects_malformed_calls() -> None:
    numeric = re.fullmatch(ISSUER_PATTERN, "companysummary_open('00066')")
    alphanumeric = re.fullmatch(ISSUER_PATTERN, "companysummary_open('A12B3')")

    assert numeric is not None
    assert numeric.group(1) == "00066"
    assert alphanumeric is not None
    assert alphanumeric.group(1) == "A12B3"
    assert not re.fullmatch(ISSUER_PATTERN, "companysummary_open('A-12')")
    assert not re.fullmatch(ISSUER_PATTERN, "companysummary_open('A12'")


def test_receipt_pattern_requires_ascii_digits_and_the_next_argument_delimiter() -> None:
    match = re.search(RECEIPT_PATTERN, "openDisclsViewer('20240425000004','')")

    assert match is not None
    assert match.group(1) == "20240425000004"
    assert not re.search(RECEIPT_PATTERN, "openDisclsViewer('2024A425000004','')")
    assert not re.search(RECEIPT_PATTERN, "openDisclsViewer('20240425000004')")


def test_title_patterns_distinguish_supported_disclosure_types() -> None:
    assert re.fullmatch(
        PROVISIONAL_TITLE_PATTERN,
        "영업 (잠정) 실적 (공정공시)",
    )
    assert not re.fullmatch(
        PROVISIONAL_TITLE_PATTERN,
        "영업 확정 실적 (공정공시)",
    )
    assert re.fullmatch(FORECAST_TITLE_PATTERN, "영업실적 등에 대한 전망")
    assert not re.fullmatch(FORECAST_TITLE_PATTERN, "영업실적 등에 대한 공시")


def test_disclosure_is_frozen_and_slotted() -> None:
    disclosure = _disclosure()

    assert disclosure.company == "SK하이닉스"
    assert not hasattr(disclosure, "__dict__")
    with pytest.raises(FrozenInstanceError):
        disclosure.time = "08:06"  # type: ignore[misc]


def test_parsed_page_is_frozen_and_slotted() -> None:
    page = ParsedPage(disclosures=(_disclosure(),), total_pages=5)

    assert page.disclosures[0].receipt_id == "20240425000004"
    assert not hasattr(page, "__dict__")
    with pytest.raises(FrozenInstanceError):
        page.total_pages = 6  # type: ignore[misc]


def test_match_result_is_frozen_and_slotted() -> None:
    result = MatchResult(
        confidence=Confidence.EXACT_MATCH,
        disclosure=_disclosure(),
    )

    assert result.candidates == ()
    assert result.rejection_reason is None
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.confidence = Confidence.NO_MATCH  # type: ignore[misc]


@pytest.mark.parametrize(
    ("confidence", "has_disclosure"),
    [
        (Confidence.EXACT_MATCH, True),
        (Confidence.NORMALIZED_MATCH, True),
        (Confidence.MULTIPLE_MATCH, False),
        (Confidence.NO_MATCH, False),
    ],
)
def test_match_result_accepts_consistent_states(
    confidence: Confidence,
    has_disclosure: bool,
) -> None:
    disclosure = _disclosure() if has_disclosure else None

    result = MatchResult(confidence=confidence, disclosure=disclosure)

    assert result.disclosure is disclosure


@pytest.mark.parametrize(
    "confidence",
    [Confidence.EXACT_MATCH, Confidence.NORMALIZED_MATCH],
)
def test_match_result_requires_disclosure_for_single_match_confidence(
    confidence: Confidence,
) -> None:
    with pytest.raises(ValueError, match="requires a disclosure"):
        MatchResult(confidence=confidence, disclosure=None)


@pytest.mark.parametrize(
    "confidence",
    [Confidence.MULTIPLE_MATCH, Confidence.NO_MATCH],
)
def test_match_result_forbids_disclosure_for_non_single_match_confidence(
    confidence: Confidence,
) -> None:
    with pytest.raises(ValueError, match="must not include a disclosure"):
        MatchResult(confidence=confidence, disclosure=_disclosure())
