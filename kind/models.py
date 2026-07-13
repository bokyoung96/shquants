from dataclasses import dataclass, field
from enum import Enum


class Confidence(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    NORMALIZED_MATCH = "NORMALIZED_MATCH"
    MULTIPLE_MATCH = "MULTIPLE_MATCH"
    NO_MATCH = "NO_MATCH"


@dataclass(frozen=True, slots=True)
class Disclosure:
    announcement_date: str
    time: str
    company: str
    title: str
    submitter: str
    issuer_id: str | None
    receipt_id: str
    page: int
    position: int


@dataclass(frozen=True, slots=True)
class ParsedPage:
    disclosures: tuple[Disclosure, ...]
    total_pages: int


@dataclass(frozen=True, slots=True)
class MatchResult:
    confidence: Confidence
    disclosure: Disclosure | None
    candidates: tuple[Disclosure, ...] = field(default_factory=tuple)
    rejection_reason: str | None = None
