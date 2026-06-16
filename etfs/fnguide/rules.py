from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MethodologyRules:
    index_name: str = ""
    updated: str = ""
    methodology_family: str = "unknown"
    review_frequency: str = "unknown"
    review_months: list[int] = field(default_factory=list)
    rebalance_timing: str = ""
    selection_count: int | None = None
    weighting_scheme: str = "unknown"
    weight_cap: str = ""
    has_free_float_adjustment: bool = False
    has_market_cap_screen: bool = False
    has_liquidity_screen: bool = False
    has_keyword_filter: bool = False
    has_fics_filter: bool = False
    evidence: dict[str, str] = field(default_factory=dict)

    def to_rule_set(self) -> "IndexMethodologyRuleSet":
        return IndexMethodologyRuleSet.from_rules(self)


@dataclass(frozen=True, slots=True)
class RebalanceRule:
    frequency: str
    months: tuple[int, ...] = ()
    timing: str = ""

    @property
    def is_periodic(self) -> bool:
        return self.frequency in {"monthly", "quarterly", "semiannual", "annual"} and bool(self.months)


@dataclass(frozen=True, slots=True)
class WeightingRule:
    scheme: str
    cap: str = ""
    uses_free_float: bool = False

    @property
    def cap_percent(self) -> float | None:
        if not self.cap:
            return None
        match = re.search(r"\d+(?:\.\d+)?", self.cap)
        return float(match.group(0)) if match else None

    @property
    def requires_score_data(self) -> bool:
        return self.scheme == "score_weighted"

    @property
    def requires_fundamental_data(self) -> bool:
        return self.scheme == "fundamental_float_adjusted"

    @property
    def is_equal_weighted(self) -> bool:
        return self.scheme == "equal_weighted"


@dataclass(frozen=True, slots=True)
class ScreeningRule:
    selection_count: int | None = None
    market_cap: bool = False
    liquidity: bool = False
    keyword: bool = False
    fics: bool = False

    @property
    def requires_external_keyword_data(self) -> bool:
        return self.keyword


@dataclass(frozen=True, slots=True)
class IndexMethodologyRuleSet:
    index_name: str
    updated: str
    family: str
    rebalance: RebalanceRule
    weighting: WeightingRule
    screening: ScreeningRule
    evidence: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_rules(cls, rules: MethodologyRules) -> "IndexMethodologyRuleSet":
        return cls(
            index_name=rules.index_name,
            updated=rules.updated,
            family=rules.methodology_family,
            rebalance=RebalanceRule(
                frequency=rules.review_frequency,
                months=tuple(rules.review_months),
                timing=rules.rebalance_timing,
            ),
            weighting=WeightingRule(
                scheme=rules.weighting_scheme,
                cap=rules.weight_cap,
                uses_free_float=rules.has_free_float_adjustment,
            ),
            screening=ScreeningRule(
                selection_count=rules.selection_count,
                market_cap=rules.has_market_cap_screen,
                liquidity=rules.has_liquidity_screen,
                keyword=rules.has_keyword_filter,
                fics=rules.has_fics_filter,
            ),
            evidence=dict(rules.evidence),
        )
