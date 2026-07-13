from __future__ import annotations

from collections.abc import Iterable
import re
import unicodedata

from kind.models import Confidence, Disclosure, MatchResult
from kind.selectors import FORECAST_TITLE_PATTERN, PROVISIONAL_TITLE_PATTERN


LEGAL_FORMS = ("(주)", "주식회사")
SUBSIDIARY_MARKER = "자회사의 주요경영사항"
CORRECTION_PREFIX = "[정정]"
VALID_TICKER_PATTERN = re.compile(r"A[0-9A-Z]{6}")


def normalize_company_name(value: str) -> str:
    """Normalize formatting without inventing company-name aliases."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    for legal_form in LEGAL_FORMS:
        normalized = normalized.replace(legal_form, "")
    return "".join(character for character in normalized if character.isalnum())


def _issuer_id_for_ticker(ticker: str) -> str | None:
    if VALID_TICKER_PATTERN.fullmatch(ticker) is None:
        return None
    return ticker[1:6]


def _is_eligible(candidate: Disclosure) -> bool:
    return (
        re.search(PROVISIONAL_TITLE_PATTERN, candidate.title) is not None
        and re.search(FORECAST_TITLE_PATTERN, candidate.title) is None
    )


def _receipt_evidence(candidate: Disclosure) -> tuple[str, str, str, str | None]:
    return (
        candidate.company,
        candidate.title,
        candidate.time,
        candidate.issuer_id,
    )


def _deduplicate_receipts(
    candidates: Iterable[Disclosure],
) -> tuple[list[Disclosure], list[Disclosure]]:
    unique: list[Disclosure] = []
    conflicts: list[Disclosure] = []
    first_by_receipt: dict[str, Disclosure] = {}

    for candidate in candidates:
        first = first_by_receipt.get(candidate.receipt_id)
        if first is None:
            first_by_receipt[candidate.receipt_id] = candidate
            unique.append(candidate)
        elif _receipt_evidence(first) != _receipt_evidence(candidate):
            if first not in conflicts:
                conflicts.append(first)
            conflicts.append(candidate)

    return unique, conflicts


def _is_issuer_viable(
    candidate: Disclosure,
    expected_issuer: str | None,
) -> bool:
    return (
        expected_issuer is None
        or candidate.issuer_id is None
        or candidate.issuer_id == expected_issuer
    )


def _is_correction(candidate: Disclosure) -> bool:
    return candidate.title.strip().startswith(CORRECTION_PREFIX)


def _is_subsidiary(candidate: Disclosure) -> bool:
    return SUBSIDIARY_MARKER in candidate.title


def _apply_title_priority(candidates: list[Disclosure]) -> list[Disclosure]:
    direct_originals = [
        candidate
        for candidate in candidates
        if not _is_correction(candidate) and not _is_subsidiary(candidate)
    ]
    if direct_originals:
        return direct_originals

    direct = [
        candidate for candidate in candidates if not _is_subsidiary(candidate)
    ]
    if direct:
        return direct

    subsidiary_originals = [
        candidate for candidate in candidates if not _is_correction(candidate)
    ]
    return subsidiary_originals or candidates


def match_disclosure(
    ticker: str,
    company: str,
    disclosures: Iterable[Disclosure],
) -> MatchResult:
    """Return one evidenced match or fail closed with auditable candidates."""

    eligible = [candidate for candidate in disclosures if _is_eligible(candidate)]
    if not eligible:
        return MatchResult(
            confidence=Confidence.NO_MATCH,
            disclosure=None,
            rejection_reason="no eligible provisional title candidate",
        )

    expected_issuer = _issuer_id_for_ticker(ticker)

    exact_candidates = [
        candidate
        for candidate in eligible
        if candidate.company.strip() == company.strip()
        and _is_issuer_viable(candidate, expected_issuer)
    ]
    candidates, conflicts = _deduplicate_receipts(exact_candidates)
    confidence = Confidence.EXACT_MATCH

    if not candidates and not conflicts:
        normalized_company = normalize_company_name(company)
        if not normalized_company:
            return MatchResult(
                confidence=Confidence.NO_MATCH,
                disclosure=None,
                rejection_reason="expected company name normalizes to empty",
            )
        normalized_candidates = [
            candidate
            for candidate in eligible
            if normalize_company_name(candidate.company) == normalized_company
            and _is_issuer_viable(candidate, expected_issuer)
        ]
        candidates, conflicts = _deduplicate_receipts(normalized_candidates)
        confidence = Confidence.NORMALIZED_MATCH

    if conflicts:
        return MatchResult(
            confidence=Confidence.MULTIPLE_MATCH,
            disclosure=None,
            candidates=tuple(conflicts),
            rejection_reason="conflicting evidence for duplicate receipt identifier",
        )

    if not candidates:
        issuer_conflicting_company_candidates = [
            candidate
            for candidate in eligible
            if candidate.company.strip() == company.strip()
            or normalize_company_name(candidate.company)
            == normalize_company_name(company)
        ]
        reason = "no eligible company name candidate"
        if any(
            not _is_issuer_viable(candidate, expected_issuer)
            for candidate in issuer_conflicting_company_candidates
        ):
            reason = "company candidates conflict with expected issuer"
        return MatchResult(
            confidence=Confidence.NO_MATCH,
            disclosure=None,
            rejection_reason=reason,
        )

    candidates = _apply_title_priority(candidates)
    if len(candidates) > 1:
        return MatchResult(
            confidence=Confidence.MULTIPLE_MATCH,
            disclosure=None,
            candidates=tuple(candidates),
            rejection_reason="multiple eligible receipt identifiers",
        )

    candidate = candidates[0]
    return MatchResult(
        confidence=confidence,
        disclosure=candidate,
        candidates=(candidate,),
    )
