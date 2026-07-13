from dataclasses import FrozenInstanceError

import pytest

from kind.models import Confidence, Disclosure, MatchResult, ParsedPage
from kind.selectors import FORM_DEFAULTS, KIND_MAIN_URL, KIND_SUB_URL


def test_kind_package_contract() -> None:
    assert [confidence.value for confidence in Confidence] == [
        "EXACT_MATCH",
        "NORMALIZED_MATCH",
        "MULTIPLE_MATCH",
        "NO_MATCH",
    ]
    assert KIND_MAIN_URL.endswith("method=searchTodayDisclosureMain")
    assert KIND_SUB_URL.endswith("/disclosure/todaydisclosure.do")
    assert FORM_DEFAULTS["method"] == "searchTodayDisclosureSub"

    disclosure = Disclosure(
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
    page = ParsedPage(disclosures=(disclosure,), total_pages=5)
    result = MatchResult(confidence=Confidence.EXACT_MATCH, disclosure=disclosure)

    assert page.disclosures[0].time == "08:05"
    assert page.total_pages == 5
    assert result.confidence is Confidence.EXACT_MATCH
    assert result.disclosure is disclosure
    assert result.candidates == ()
    assert result.rejection_reason is None
    with pytest.raises(FrozenInstanceError):
        disclosure.time = "08:06"  # type: ignore[misc]
