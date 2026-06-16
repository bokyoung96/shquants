from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import urlparse

import fitz

from etfs import paths


PdfReader = Callable[[Path], "PdfTextDocument"]


@dataclass(frozen=True, slots=True)
class PdfTextPage:
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class PdfTextDocument:
    path: Path
    pages: tuple[PdfTextPage, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)


@dataclass(frozen=True, slots=True)
class FieldEvidence:
    source: str
    section: str
    text: str
    page: int | None = None


@dataclass(frozen=True, slots=True)
class ExtractedField:
    value: object
    confidence: str
    evidence: list[FieldEvidence] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MethodologyExtraction:
    etf_code: str
    etf_name: str
    index_code: str
    index_name: str
    provider: str
    extraction_status: str
    source: dict[str, object]
    sections: dict[str, dict[str, object]]
    fields: dict[str, ExtractedField]
    open_questions: list[str]


def load_rule_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"unsupported rules payload: {path}")
    return list(payload["items"])


def read_pdf_text_pages(path: Path) -> PdfTextDocument:
    with fitz.open(path) as document:
        pages = tuple(
            PdfTextPage(page_number=index + 1, text=page.get_text("text"))
            for index, page in enumerate(document)
        )
    return PdfTextDocument(path=path, pages=pages)


def build_methodology_extractions(
    items: Iterable[Mapping[str, object]],
    *,
    pdf_reader: PdfReader | None = None,
) -> list[MethodologyExtraction]:
    reader = pdf_reader or read_pdf_text_pages
    return [_build_extraction(item, pdf_reader=reader) for item in items if _index_code_from_page_url(str(item.get("page_url", "")))]


def extract_constituent_count_fields(
    text: str,
    *,
    evidence_source: str = "rules_evidence",
    evidence_page: int | None = None,
) -> dict[str, ExtractedField]:
    fields: dict[str, ExtractedField] = {}
    normalized = _normalize_space(text)

    final = "\ucd5c\uc885"
    item = "\uc885\ubaa9"
    item_pattern = "\uc885" + r"\s*" + "\ubaa9"
    object_particle = "\uc744"
    top = "\uc0c1\uc704"
    selected = "\uc120\uc815\ud558\uc5ec"
    selected_done = "\uc120\uc815\ub41c"
    composed = "\uad6c\uc131"
    high = "\ub192\uc740"
    portfolio = "\ud3ec\ud2b8\ud3f4\ub9ac\uc624\ub85c"
    as_particle = "\uc73c\ub85c"
    rank_suffix = "\uc704\ub97c"
    target = "\ub300\uc0c1\uc73c\ub85c"
    about = "\ub0b4\uc678"
    maximum = "\ucd5c\ub300"
    more_than = "\ubcf4\ub2e4 \ub9ce\uc744 \uacbd\uc6b0"
    less_than = "\ubcf4\ub2e4 \uc801\uc744 \uacbd\uc6b0"
    below_case = "\ubbf8\ub9cc\uc778 \uacbd\uc6b0"
    cumulative_weight = "\ub204\uc801 \ud3b8\uc785\ube44\uc911"
    total_word = "\ucd1d"
    exact_patterns = (
        rf"{final}\s*(\d+)\s*{item}",
        rf"{total_word}\s*(\d+)\s*{item_pattern}[^.]{{0,80}}?{composed}",
        rf"{total_word}\s*(\d+)\s*{item_pattern}[^.]{{0,80}}?{selected}",
        rf"{final}\s*{composed}\s*{item}{as_particle}\s*{selected_done}\s*(\d+)\s*{item}",
        rf"{top}\s*(\d+)\s*{item}{object_particle}?\s*{selected}[^.]{{0,160}}?{composed}",
        rf"{top}\s*(\d+)\s*{item}{object_particle}?\s*\uc120\uc815\ud569\ub2c8\ub2e4",
        rf"{high}\s*(\d+)\s*{item}{object_particle}?\s*{selected}[^.]{{0,80}}?{composed}",
        rf"{high}\s*(\d+)\s*{item}{object_particle}?\s*{portfolio}\s*{composed}",
        rf"{top}\s*(\d+)\s*{rank_suffix}\s*{target}[^.]{{0,80}}?{composed}",
    )
    for pattern in exact_patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        context = normalized[max(0, match.start() - 20) : match.end() + 20]
        if about in context or maximum in context:
            continue
        fields["selection.total_constituents"] = _field(
            int(match.group(1)),
            "selection",
            _sentence_containing(normalized, match.group(0)),
            source=evidence_source,
            page=evidence_page,
        )
        return fields

    cumulative_match = re.search(rf"{cumulative_weight}[^.]{{0,80}}?(\d+(?:\.\d+)?)\s*%", normalized)
    if cumulative_match:
        fields["selection.variable_count.method"] = _field(
            "cumulative_weight_threshold",
            "selection",
            _sentence_containing(normalized, cumulative_match.group(0)),
            source=evidence_source,
            page=evidence_page,
        )
        fields["selection.variable_count.threshold"] = _field(
            float(cumulative_match.group(1)) / 100.0,
            "selection",
            _sentence_containing(normalized, cumulative_match.group(0)),
            source=evidence_source,
            page=evidence_page,
        )

    max_threshold = re.search(rf"(\d+)\s*\uac1c\s*{more_than}", normalized)
    if max_threshold:
        fields["selection.max_constituents"] = _field(
            int(max_threshold.group(1)),
            "selection",
            _sentence_containing(normalized, max_threshold.group(0)),
            source=evidence_source,
            page=evidence_page,
        )

    min_threshold = re.search(rf"(\d+)\s*\uac1c\s*(?:{less_than}|{below_case})", normalized)
    if min_threshold:
        fields["selection.min_constituents"] = _field(
            int(min_threshold.group(1)),
            "selection",
            _sentence_containing(normalized, min_threshold.group(0)),
            source=evidence_source,
            page=evidence_page,
        )

    max_match = re.search(rf"{maximum}\s*(\d+)\s*{item}", normalized)
    if max_match and "selection.max_constituents" not in fields:
        fields["selection.max_constituents"] = _field(
            int(max_match.group(1)),
            "selection",
            _sentence_containing(normalized, max_match.group(0)),
            source=evidence_source,
            page=evidence_page,
        )
    return fields


def extract_weighting_cap_fields(
    text: str,
    *,
    evidence_source: str = "rules_evidence",
    evidence_page: int | None = None,
) -> dict[str, ExtractedField]:
    fields: dict[str, ExtractedField] = {}
    normalized = _normalize_space(text)
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", normalized):
        sentence = _sentence_containing(normalized, match.group(0))
        if len(set(re.findall(r"(\d+(?:\.\d+)?)\s*%", sentence))) > 1:
            continue
        if _is_plain_single_security_cap_sentence(sentence):
            fields["weighting.security_cap"] = _field(
                float(match.group(1)) / 100.0,
                "weighting",
                sentence,
                source=evidence_source,
                page=evidence_page,
            )
            return fields
    return fields


def extract_top2_plus_fields(
    text: str,
    *,
    evidence_source: str = "rules_evidence",
    evidence_page: int | None = None,
) -> dict[str, ExtractedField]:
    fields: dict[str, ExtractedField] = {}
    normalized = _normalize_space(text)

    total = _extract_int_before(normalized, ("종목으로 지수를 구성", "종목으로 구성", "최종"))
    if total is not None:
        fields["selection.total_constituents"] = _field(
            total,
            "selection",
            _sentence_containing(normalized, "최종"),
            source=evidence_source,
            page=evidence_page,
        )

    if "TOP2" in normalized.upper():
        fields["selection.buckets.top2.count"] = _field(
            2,
            "selection",
            _sentence_containing(normalized, "TOP2"),
            source=evidence_source,
            page=evidence_page,
        )

    top2_weight = _extract_top2_fixed_weight(normalized)
    if top2_weight is not None:
        fields["selection.buckets.top2.weight"] = _field(
            top2_weight,
            "weighting",
            _sentence_containing(normalized, "TOP2"),
            source=evidence_source,
            page=evidence_page,
        )

    momentum_count = _extract_count_after_phrase(normalized, "합산스코어")
    if momentum_count is not None:
        fields["selection.buckets.momentum.count"] = _field(
            momentum_count,
            "selection",
            _sentence_containing(normalized, "합산스코어"),
            source=evidence_source,
            page=evidence_page,
        )

    market_cap_count = _extract_count_after_phrase(normalized, "시가총액 상위 종목 순으로")
    if market_cap_count is None:
        market_cap_count = _extract_count_after_marker(normalized, "우선 선정하고", "시가총액 상위")
    if market_cap_count is not None:
        fields["selection.buckets.market_cap_fill.count"] = _field(
            market_cap_count,
            "selection",
            _sentence_containing(normalized, "시가총액 상위"),
            source=evidence_source,
            page=evidence_page,
        )

    residual_count = _extract_count_after_phrase(normalized, "나머지")
    if residual_count is not None:
        fields["weighting.residual.count"] = _field(
            residual_count,
            "weighting",
            _sentence_containing(normalized, "나머지"),
            source=evidence_source,
            page=evidence_page,
        )

    cap = _extract_percent_near(normalized, "실링") or _extract_percent_near(normalized, "최대")
    if cap is not None:
        fields["weighting.residual.cap"] = _field(
            cap,
            "weighting",
            _sentence_containing(normalized, "15%"),
            source=evidence_source,
            page=evidence_page,
        )

    return fields


def write_methodology_extractions(
    rules_path: Path,
    output_dir: Path,
    *,
    pdf_reader: PdfReader | None = None,
) -> tuple[Path, Path]:
    extractions = build_methodology_extractions(load_rule_items(rules_path), pdf_reader=pdf_reader)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "methodology_extractions.json"
    md_path = output_dir / "methodology_extractions.md"
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(extractions),
        "items": [_to_jsonable(extraction) for extraction in extractions],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(extractions), encoding="utf-8")
    return json_path, md_path


def _build_extraction(item: Mapping[str, object], *, pdf_reader: PdfReader) -> MethodologyExtraction:
    rules = _mapping(item.get("rules"))
    sections = _extract_pdf_sections(item, rules, pdf_reader=pdf_reader)
    combined = "\n".join(str(section.get("text", "")) for section in sections.values() if section.get("text"))
    fields = _base_rule_fields(rules)
    primary_section = _primary_section(sections)
    fields.update(
        extract_constituent_count_fields(
            combined,
            evidence_source=str(primary_section.get("source", "rules_evidence")),
            evidence_page=_int_or_none(primary_section.get("page")),
        )
    )
    fields.update(
        extract_weighting_cap_fields(
            combined,
            evidence_source=str(primary_section.get("source", "rules_evidence")),
            evidence_page=_int_or_none(primary_section.get("page")),
        )
    )
    if "TOP2 Plus" in str(rules.get("index_name", "")) or "TOP2플러스" in str(item.get("name", "")):
        fields.update(
            extract_top2_plus_fields(
                combined,
                evidence_source=str(primary_section.get("source", "rules_evidence")),
                evidence_page=_int_or_none(primary_section.get("page")),
            )
        )
    questions = _open_questions(item, fields)
    return MethodologyExtraction(
        etf_code=str(item.get("code", "")),
        etf_name=str(item.get("name", "")),
        index_code=_index_code_from_page_url(str(item.get("page_url", ""))),
        index_name=str(rules.get("index_name", "")),
        provider="fnguide",
        extraction_status="draft_extracted",
        source={
            "methodology_pdf_path": str(item.get("file_path", "")),
            "methodology_pdf_sha256": str(item.get("sha256", "")),
            "source_url": str(item.get("source_url", "")),
            "page_url": str(item.get("page_url", "")),
            "pdf_page_count": int(item.get("pdf_page_count") or 0),
            "methodology_updated": str(rules.get("updated", "")),
        },
        sections=sections,
        fields=fields,
        open_questions=questions,
    )


def _base_rule_fields(rules: Mapping[str, object]) -> dict[str, ExtractedField]:
    fields: dict[str, ExtractedField] = {}
    fields["rebalance.frequency"] = _field(str(rules.get("review_frequency", "")), "rules", "rules.review_frequency")
    fields["rebalance.implementation_months"] = _field(
        list(rules.get("review_months") or []),
        "rules",
        "rules.review_months",
    )
    fields["rebalance.implementation_timing"] = _field(
        str(rules.get("rebalance_timing", "")),
        "rules",
        "rules.rebalance_timing",
    )
    fields["weighting.base"] = _field(str(rules.get("weighting_scheme", "")), "rules", "rules.weighting_scheme")
    fields["weighting.rule_cap"] = _field(str(rules.get("weight_cap", "")), "rules", "rules.weight_cap")
    return fields


def _open_questions(item: Mapping[str, object], fields: Mapping[str, ExtractedField]) -> list[str]:
    rules = _mapping(item.get("rules"))
    questions: list[str] = []
    selection_count = rules.get("selection_count")
    total = fields.get("selection.total_constituents")
    if selection_count and total and selection_count != total.value:
        if _has_top2_evidence(item, rules, fields):
            questions.append("rules.selection_count means top2 bucket count, not total constituents")
        else:
            questions.append("rules.selection_count differs from PDF total constituents")
    if not total and "selection.variable_count.method" not in fields:
        questions.append("selection.total_constituents not extracted with evidence")
    if "weighting.residual.cap" not in fields and "weighting.security_cap" not in fields and rules.get("weight_cap"):
        questions.append("weight cap exists in rules but residual/fixed bucket scope is unresolved")
    return questions


def _has_top2_evidence(
    item: Mapping[str, object],
    rules: Mapping[str, object],
    fields: Mapping[str, ExtractedField],
) -> bool:
    return (
        "selection.buckets.top2.count" in fields
        or "TOP2" in str(rules.get("index_name", "")).upper()
        or "TOP2" in str(item.get("name", "")).upper()
    )


def _extract_pdf_sections(
    item: Mapping[str, object],
    rules: Mapping[str, object],
    *,
    pdf_reader: PdfReader,
) -> dict[str, dict[str, object]]:
    file_path = Path(str(item.get("file_path", "")))
    if str(item.get("status", "")) == "downloaded" and str(file_path):
        try:
            document = pdf_reader(file_path)
        except Exception as exc:  # noqa: BLE001
            sections = _sections_from_rules(rules)
            sections["pdf_read_error"] = {"source": "methodology_pdf", "page": None, "text": str(exc)}
            return sections
        sections = _sections_from_pdf_document(document)
        if any(section.get("text") for section in sections.values()):
            return sections
    return _sections_from_rules(rules)


def _sections_from_pdf_document(document: PdfTextDocument) -> dict[str, dict[str, object]]:
    return {
        "selection": _find_pdf_section(
            document,
            ("최종", "합산스코어", "종목구성", "유니버스", "TOP2"),
        ),
        "weighting": _find_pdf_section(
            document,
            ("편입 비중", "비중", "실링", "25%", "15%"),
        ),
        "rebalance": _find_pdf_section(
            document,
            ("정기변경", "D+2", "옵션 만기", "만기일"),
        ),
    }


def _find_pdf_section(document: PdfTextDocument, needles: tuple[str, ...]) -> dict[str, object]:
    scored: list[tuple[int, PdfTextPage]] = []
    for page in document.pages:
        text = _normalize_space(page.text)
        score = sum(1 for needle in needles if needle in text)
        if score:
            scored.append((score, page))
    if not scored:
        return {"source": "methodology_pdf", "page": None, "text": ""}
    _, page = max(scored, key=lambda item: (item[0], -item[1].page_number))
    return {
        "source": "methodology_pdf",
        "page": page.page_number,
        "text": _snippet(_page_window_text(document, page.page_number), limit=4000),
    }


def _page_window_text(document: PdfTextDocument, anchor_page_number: int) -> str:
    return "\n".join(
        page.text
        for page in document.pages
        if anchor_page_number <= page.page_number <= anchor_page_number + 2
    )


def _sections_from_rules(rules: Mapping[str, object]) -> dict[str, dict[str, object]]:
    evidence = _mapping(rules.get("evidence"))
    return {
        "selection": {"source": "rules_evidence", "page": None, "text": str(evidence.get("universe", ""))},
        "weighting": {"source": "rules_evidence", "page": None, "text": str(evidence.get("weighting", ""))},
        "rebalance": {"source": "rules_evidence", "page": None, "text": str(evidence.get("schedule", ""))},
    }


def _primary_section(sections: Mapping[str, Mapping[str, object]]) -> Mapping[str, object]:
    for name in ("selection", "weighting", "rebalance"):
        section = sections.get(name, {})
        if section.get("text"):
            return section
    return {}


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _field(
    value: object,
    section: str,
    evidence_text: str,
    *,
    source: str = "rules_evidence",
    page: int | None = None,
) -> ExtractedField:
    return ExtractedField(
        value=value,
        confidence="high" if evidence_text and not evidence_text.startswith("rules.") else "medium",
        evidence=[FieldEvidence(source=source, section=section, text=evidence_text, page=page)] if evidence_text else [],
    )


def _extract_int_before(text: str, markers: tuple[str, ...]) -> int | None:
    final_match = re.search(r"최종\s*(\d+)\s*종목", text)
    if final_match:
        return int(final_match.group(1))
    for marker in markers:
        position = text.find(marker)
        if position < 0:
            continue
        prefix = text[max(0, position - 40) : position]
        matches = re.findall(r"(\d+)\s*종목", prefix)
        if matches:
            return int(matches[-1])
    return None


def _extract_count_after_phrase(text: str, phrase: str) -> int | None:
    position = text.find(phrase)
    if position < 0:
        return None
    window = text[position : min(len(text), position + 80)]
    match = re.search(r"(\d+)\s*종목", window)
    return int(match.group(1)) if match else None


def _extract_count_after_marker(text: str, marker: str, phrase: str) -> int | None:
    marker_position = text.find(marker)
    if marker_position < 0:
        return None
    phrase_position = text.find(phrase, marker_position)
    if phrase_position < 0:
        return None
    window = text[phrase_position : min(len(text), phrase_position + 100)]
    match = re.search(r"(\d+)\s*종목", window)
    return int(match.group(1)) if match else None


def _extract_percent_near(text: str, phrase: str) -> float | None:
    for match in re.finditer(re.escape(phrase), text):
        position = match.start()
        window = text[max(0, position - 80) : min(len(text), position + 120)]
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", window)
        if matches:
            return float(matches[-1]) / 100.0
    return None


def _extract_top2_fixed_weight(text: str) -> float | None:
    for match in re.finditer(r"TOP2[^.%]{0,80}?(\d+(?:\.\d+)?)\s*%", text, flags=re.I):
        return float(match.group(1)) / 100.0
    return None


def _is_plain_single_security_cap_sentence(sentence: str) -> bool:
    specific_security = "\ud2b9\uc815 \uc885\ubaa9"
    compact_specific_security = "\ud2b9\uc815\uc885\ubaa9"
    individual_security = "\uac1c\ubcc4 \uc885\ubaa9"
    compact_individual_security = "\uac1c\ubcc4\uc885\ubaa9"
    one_security = "\ud55c \uc885\ubaa9"
    weight = "\ube44\uc911"
    cap_words = (
        "\uc81c\ud55c",
        "\ucd5c\ub300",
        "\ub118\uc744 \uacbd\uc6b0",
        "\ub118\uc744\uacbd\uc6b0",
        "\ub118\ub294 \uacbd\uc6b0",
        "\ub118\ub294\uacbd\uc6b0",
        "\ucd08\uacfc\ud560 \uacbd\uc6b0",
        "\ucd08\uacfc\ud560\uacbd\uc6b0",
        "\ucd08\uacfc\ud560 \uc218 \uc5c6",
    )
    if "TOP2" in sentence.upper():
        return False
    if (
        specific_security not in sentence
        and compact_specific_security not in sentence
        and individual_security not in sentence
        and compact_individual_security not in sentence
        and one_security not in sentence
    ):
        return False
    return weight in sentence and any(word in sentence for word in cap_words)


def _sentence_containing(text: str, needle: str) -> str:
    position = text.find(needle)
    if position < 0:
        return _snippet(text)
    start = max(text.rfind(".", 0, position), text.rfind("\n", 0, position))
    end_candidates = [candidate for candidate in (text.find(".", position), text.find("\n", position)) if candidate >= 0]
    end = min(end_candidates) if end_candidates else min(len(text), position + 260)
    return _snippet(text[start + 1 : end + 1])


def _index_code_from_page_url(url: str) -> str:
    path = urlparse(url).path
    return path.rstrip("/").split("/")[-1] if path else ""


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _snippet(text: str, *, limit: int = 400) -> str:
    value = _normalize_space(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _to_jsonable(extraction: MethodologyExtraction) -> dict[str, object]:
    return asdict(extraction)


def _markdown_summary(extractions: list[MethodologyExtraction]) -> str:
    lines = [
        "# FnGuide Methodology Extractions",
        "",
        "| index_code | etf_code | index_name | fields | open_questions |",
        "| --- | --- | --- | --- | --- |",
    ]
    for extraction in extractions:
        field_names = ", ".join(sorted(extraction.fields))
        questions = "<br>".join(extraction.open_questions)
        lines.append(
            f"| {extraction.index_code} | {extraction.etf_code} | {extraction.index_name} | {field_names} | {questions} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build evidence-rich FnGuide methodology extractions.")
    parser.add_argument("--rules", default=paths.FNGUIDE_RULES_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_EXTRACTION_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_path, md_path = write_methodology_extractions(Path(args.rules), Path(args.output_dir))
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
