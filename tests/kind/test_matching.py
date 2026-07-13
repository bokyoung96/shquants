from __future__ import annotations

from dataclasses import replace

import pytest

from kind.matching import match_disclosure, normalize_company_name
from kind.models import Confidence, Disclosure


PROVISIONAL_TITLE = "영업(잠정)실적(공정공시)"
CONNECTED_TITLE = f"연결재무제표기준 {PROVISIONAL_TITLE}"


def disclosure(
    *,
    company: str = "삼성전자",
    title: str = PROVISIONAL_TITLE,
    time: str = "08:31",
    receipt: str = "20240430000001",
    issuer: str | None = "00593",
    page: int = 1,
    position: int = 1,
) -> Disclosure:
    return Disclosure(
        announcement_date="2024-04-30",
        time=time,
        company=company,
        title=title,
        submitter=company,
        issuer_id=issuer,
        receipt_id=receipt,
        page=page,
        position=position,
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (" (주) 삼성-전자 ", "삼성전자"),
        ("㈜삼성 전자", "삼성전자"),
        ("주식회사 (삼성), 전자!", "삼성전자"),
        (" ＳＡＭＳＵＮＧ Co., Ltd. ", "samsungcoltd"),
        ("Acme—Holdings", "acmeholdings"),
        ("ＡＢＣ", "abc"),
        ("--- ( ) !!!", ""),
    ],
)
def test_normalize_company_name_removes_only_formatting_and_legal_forms(
    source: str,
    expected: str,
) -> None:
    assert normalize_company_name(source) == expected


def test_normalize_company_name_does_not_apply_fuzzy_aliases() -> None:
    assert normalize_company_name("현대차") != normalize_company_name("현대자동차")


def test_trimmed_exact_company_candidate_is_accepted() -> None:
    candidate = disclosure(company="  삼성전자  ", title=CONNECTED_TITLE)

    result = match_disclosure("A005930", " 삼성전자 ", [candidate])

    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is candidate
    assert result.candidates == (candidate,)
    assert result.rejection_reason is None


@pytest.mark.parametrize("candidate_company", ["(주) 삼성-전자", "㈜삼성 전자"])
def test_normalized_legal_form_company_candidate_is_accepted(
    candidate_company: str,
) -> None:
    candidate = disclosure(company=candidate_company)

    result = match_disclosure("A005930", "삼성전자", [candidate])

    assert result.confidence is Confidence.NORMALIZED_MATCH
    assert result.disclosure is candidate
    assert result.candidates == (candidate,)


def test_exact_stage_wins_before_normalized_stage() -> None:
    normalized = disclosure(
        company="(주) 삼성전자", receipt="20240430000001"
    )
    exact = disclosure(company="삼성전자", receipt="20240430000002")

    result = match_disclosure("A005930", "삼성전자", [normalized, exact])

    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is exact
    assert result.candidates == (exact,)


def test_empty_expected_normalized_name_fails_closed() -> None:
    result = match_disclosure(
        "A005930", " --- ", [disclosure(company="!!!")]
    )

    assert result.confidence is Confidence.NO_MATCH
    assert result.disclosure is None
    assert result.candidates == ()
    assert result.rejection_reason is not None


@pytest.mark.parametrize(
    ("ticker", "issuer"),
    [("A005930", "99999"), ("A0126Z0", "01260")],
)
def test_conflicting_issuer_rejects_an_otherwise_exact_name(
    ticker: str,
    issuer: str,
) -> None:
    result = match_disclosure(ticker, "삼성전자", [disclosure(issuer=issuer)])

    assert result.confidence is Confidence.NO_MATCH
    assert result.disclosure is None
    assert result.candidates == ()
    assert "issuer" in (result.rejection_reason or "")


def test_uppercase_alphanumeric_ticker_uses_first_five_code_characters() -> None:
    candidate = disclosure(issuer="0126Z")

    result = match_disclosure("A0126Z0", "삼성전자", [candidate])

    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is candidate


def test_absent_candidate_issuer_does_not_conflict() -> None:
    candidate = disclosure(issuer=None)

    result = match_disclosure("A005930", "삼성전자", [candidate])

    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is candidate


def test_historical_same_name_with_different_issuer_is_rejected() -> None:
    historical = disclosure(issuer="99999", receipt="20240430000001")
    current = disclosure(issuer="00593", receipt="20240430000002")

    result = match_disclosure(
        "A005930", "삼성전자", [historical, current]
    )

    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is current
    assert result.candidates == (current,)


def test_original_direct_supersedes_correction_and_subsidiary() -> None:
    correction = disclosure(
        title=f"  [정정]{PROVISIONAL_TITLE}",
        time="10:00",
        receipt="20240430000001",
    )
    subsidiary = disclosure(
        title=f"{PROVISIONAL_TITLE} (자회사의 주요경영사항)",
        time="11:00",
        receipt="20240430000002",
    )
    direct = disclosure(
        title=PROVISIONAL_TITLE,
        time="08:31",
        receipt="20240430000003",
    )

    result = match_disclosure(
        "A005930", "삼성전자", [correction, subsidiary, direct]
    )

    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is direct
    assert result.candidates == (direct,)


def test_correction_direct_supersedes_original_subsidiary() -> None:
    subsidiary = disclosure(
        title=f"{PROVISIONAL_TITLE} (자회사의 주요경영사항)",
        receipt="20240430000001",
    )
    correction = disclosure(
        title=f"[정정] {PROVISIONAL_TITLE}",
        receipt="20240430000002",
    )

    result = match_disclosure(
        "A005930", "삼성전자", [subsidiary, correction]
    )

    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is correction
    assert result.candidates == (correction,)


def test_single_correction_is_accepted() -> None:
    correction = disclosure(title=f"[정정] {PROVISIONAL_TITLE}")

    result = match_disclosure("A005930", "삼성전자", [correction])

    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is correction
    assert result.candidates == (correction,)


def test_two_direct_original_receipts_remain_ambiguous_in_dom_order() -> None:
    separate = disclosure(
        title=PROVISIONAL_TITLE,
        time="09:00",
        receipt="20240430000002",
        page=2,
        position=4,
    )
    consolidated = disclosure(
        title=CONNECTED_TITLE,
        time="08:00",
        receipt="20240430000001",
        page=1,
        position=2,
    )

    result = match_disclosure(
        "A005930", "삼성전자", [separate, consolidated]
    )

    assert result.confidence is Confidence.MULTIPLE_MATCH
    assert result.disclosure is None
    assert result.candidates == (separate, consolidated)
    assert "multiple" in (result.rejection_reason or "")


def test_identical_receipt_repeated_across_pages_is_deduplicated() -> None:
    first = disclosure(page=1, position=3)
    repeated = replace(first, page=2, position=1)

    result = match_disclosure("A005930", "삼성전자", [first, repeated])

    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is first
    assert result.candidates == (first,)


@pytest.mark.parametrize(
    "changed",
    [
        {"company": "삼성 전자"},
        {"title": f"[정정] {PROVISIONAL_TITLE}"},
        {"time": "09:30"},
        {"issuer_id": "99999"},
    ],
)
def test_conflicting_evidence_for_one_receipt_fails_closed(
    changed: dict[str, str],
) -> None:
    first = disclosure(page=1, position=3)
    conflicting = replace(first, page=2, position=1, **changed)

    result = match_disclosure(
        "A005930", "삼성전자", [first, conflicting]
    )

    assert result.confidence in {
        Confidence.MULTIPLE_MATCH,
        Confidence.NO_MATCH,
    }
    assert result.disclosure is None
    assert "conflict" in (result.rejection_reason or "")


def test_title_must_contain_provisional_pattern() -> None:
    result = match_disclosure(
        "A005930",
        "삼성전자",
        [disclosure(title="영업실적 공시")],
    )

    assert result.confidence is Confidence.NO_MATCH
    assert result.disclosure is None
    assert result.candidates == ()
    assert "title" in (result.rejection_reason or "")


def test_forecast_text_excludes_otherwise_provisional_title() -> None:
    title = f"{PROVISIONAL_TITLE} / 영업실적 등에 대한 전망"

    result = match_disclosure(
        "A005930", "삼성전자", [disclosure(title=title)]
    )

    assert result.confidence is Confidence.NO_MATCH
    assert result.disclosure is None
    assert result.candidates == ()
    assert "title" in (result.rejection_reason or "")


def test_matching_does_not_mutate_input_sequence() -> None:
    correction = disclosure(
        title=f"[정정] {PROVISIONAL_TITLE}",
        receipt="20240430000001",
    )
    direct = disclosure(receipt="20240430000002")
    inputs = [correction, direct]
    before = list(inputs)

    match_disclosure("A005930", "삼성전자", inputs)

    assert inputs == before
    assert inputs[0] is correction
    assert inputs[1] is direct
