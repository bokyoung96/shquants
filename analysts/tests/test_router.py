from analysts.domain import ParseQuality, ParsedDocument
from analysts.router import TaskRouter


def test_routes_semiconductor_reports_to_sector_semiconductors() -> None:
    parsed = ParsedDocument(
        title="Packaging Capacity Update",
        content="NVIDIA, TSMC, and HBM packaging demand continue to tighten AI supply.",
        sections=["Executive Summary:\nNVIDIA and TSMC are expanding packaging lines."],
        entities=["NVIDIA", "TSMC"],
        tickers=["NVDA", "TSM"],
        routes=[],
        parse_quality=ParseQuality.HIGH,
    )

    decisions = TaskRouter().route(parsed)

    assert [(decision.lane, decision.topic) for decision in decisions] == [("sector", "semiconductors")]
    assert decisions[0].rationale == "matched_keywords: ai, hbm, nvidia, packaging, tsmc"


def test_routes_rates_reports_to_macro_rates() -> None:
    parsed = ParsedDocument(
        title="Rates Daily",
        content="The Federal Reserve remains focused on duration, yields, and the Treasury curve.",
        sections=["Macro:\nFed speakers kept rate-cut expectations pushed into 2H."],
        entities=["Federal Reserve", "Treasury"],
        tickers=[],
        routes=[],
        parse_quality=ParseQuality.HIGH,
    )

    decisions = TaskRouter().route(parsed)

    assert [(decision.lane, decision.topic) for decision in decisions] == [("macro", "rates")]
    assert decisions[0].rationale == "matched_keywords: curve, duration, fed, treasury, yields"


def test_routes_unknown_reports_to_macro_general_fallback() -> None:
    parsed = ParsedDocument(
        title="General Market Note",
        content="Management reiterated guidance and discussed staffing plans.",
        sections=["Notes:\nNo sector or macro taxonomy keywords were present."],
        entities=["Management"],
        tickers=[],
        routes=[],
        parse_quality=ParseQuality.DEGRADED,
        degraded_reason="short_text",
    )

    decisions = TaskRouter().route(parsed)

    assert [(decision.lane, decision.topic) for decision in decisions] == [("macro", "general")]
    assert decisions[0].rationale == "fallback:no_taxonomy_match"
