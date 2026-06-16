from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import fitz

from etfs import paths
from etfs.fnguide.rules import IndexMethodologyRuleSet, MethodologyRules


PdfReader = Callable[[Path], "PdfMethodologyDocument"]


@dataclass(frozen=True, slots=True)
class PdfMethodologyDocument:
    path: Path
    page_count: int
    text: str


@dataclass(frozen=True, slots=True)
class IndexMethodology:
    code: str
    name: str
    status: str
    source_url: str
    page_url: str
    file_path: str
    sha256: str
    bytes: int
    query: str
    error: str
    pdf_page_count: int = 0
    pdf_text_chars: int = 0
    rules: MethodologyRules = field(default_factory=MethodologyRules)

    @property
    def rule_set(self) -> IndexMethodologyRuleSet:
        return self.rules.to_rule_set()


class IndexMethodologyFactory:
    def __init__(self, methodologies: Iterable[IndexMethodology]) -> None:
        self._items = tuple(methodologies)
        self._by_code = {item.code: item for item in self._items}

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        *,
        pdf_reader: PdfReader | None = None,
    ) -> "IndexMethodologyFactory":
        records = load_manifest_records(Path(manifest_path))
        return cls.from_records(records, pdf_reader=pdf_reader)

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, object]],
        *,
        pdf_reader: PdfReader | None = None,
    ) -> "IndexMethodologyFactory":
        return cls(build_index_methodologies(records, pdf_reader=pdf_reader))

    def get(self, code: str) -> IndexMethodology:
        return self._by_code[code]

    def all(self) -> tuple[IndexMethodology, ...]:
        return self._items

    def by_status(self, status: str) -> tuple[IndexMethodology, ...]:
        return tuple(item for item in self._items if item.status == status)

    def by_methodology_family(self, family: str) -> tuple[IndexMethodology, ...]:
        return tuple(item for item in self._items if item.rules.methodology_family == family)

    def by_weighting_scheme(self, scheme: str) -> tuple[IndexMethodology, ...]:
        return tuple(item for item in self._items if item.rule_set.weighting.scheme == scheme)

    def by_review_frequency(self, frequency: str) -> tuple[IndexMethodology, ...]:
        return tuple(item for item in self._items if item.rule_set.rebalance.frequency == frequency)

    def rule_set(self, code: str) -> IndexMethodologyRuleSet:
        return self.get(code).rule_set

    @property
    def count(self) -> int:
        return len(self._items)

    @property
    def downloaded_count(self) -> int:
        return sum(item.status == "downloaded" for item in self._items)

    @property
    def pdf_read_count(self) -> int:
        return sum(item.pdf_page_count > 0 for item in self._items)


def load_manifest_records(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return list(payload["items"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"unsupported methodology manifest shape: {path}")


def read_pdf_methodology(path: Path) -> PdfMethodologyDocument:
    with fitz.open(path) as document:
        text = "\n".join(page.get_text("text") for page in document)
        return PdfMethodologyDocument(path=path, page_count=document.page_count, text=text)


def build_index_methodologies(
    records: Iterable[Mapping[str, object]],
    *,
    pdf_reader: PdfReader | None = None,
) -> list[IndexMethodology]:
    reader = pdf_reader or read_pdf_methodology
    methodologies: list[IndexMethodology] = []
    for record in records:
        base = _base_fields(record)
        if base["status"] != "downloaded" or not base["file_path"]:
            methodologies.append(IndexMethodology(**base))
            continue
        try:
            document = reader(Path(base["file_path"]))
            rules = extract_methodology_rules(document.text, etf_name=base["name"])
            methodologies.append(
                IndexMethodology(
                    **base,
                    pdf_page_count=document.page_count,
                    pdf_text_chars=len(document.text),
                    rules=rules,
                )
            )
        except Exception as exc:  # noqa: BLE001
            methodologies.append(
                IndexMethodology(
                    **{**base, "status": "pdf_read_failed", "error": str(exc)},
                )
            )
    return methodologies


def extract_methodology_rules(text: str, *, etf_name: str = "") -> MethodologyRules:
    normalized = _normalize_text(text)
    schedule_text = _section_text(
        normalized,
        starts=("종목 선정일 및 개편 일정", "개편 일정", "종목 선정 기준일"),
        stops=("수시변경", "주가지수 계산", "Appendix", "3."),
    )
    if not schedule_text:
        schedule_text = _section_text(
            normalized,
            starts=("정기변경 프로세스", "지수 산출 프로세스"),
            stops=("종목구성 방법", "수시변경", "주가지수 계산", "Appendix", "3."),
        )
    weighting_text = _section_text(
        normalized,
        starts=("편입 비중 산정 방법", "비중 산정 방법", "편입비중의 결정"),
        stops=("종목 선정일", "개편 일정", "수시변경", "주가지수 계산", "Appendix"),
    )
    universe_text = _section_text(
        normalized,
        starts=("종목구성 방법", "Universe", "유니버스"),
        stops=("편입 비중", "종목 선정일", "개편 일정", "수시변경", "주가지수 계산"),
    )
    schedule_context = schedule_text or _near_keywords(normalized, ("정기변경", "개편", "리밸런싱"), radius=900)
    weighting_context = weighting_text or _near_keywords(normalized, ("편입 비중", "가중", "동일가중"), radius=700)
    universe_context = universe_text or _near_keywords(normalized, ("종목구성", "거래대금", "시가총액", "키워드"), radius=900)

    review_months = _extract_review_months(schedule_context)
    if not review_months:
        review_months = _extract_review_months(normalized)
    evidence: dict[str, str] = {}
    if schedule_context:
        evidence["schedule"] = _snippet(schedule_context)
    if weighting_context:
        evidence["weighting"] = _snippet(weighting_context)
    if universe_context:
        evidence["universe"] = _snippet(universe_context)

    weighting_scheme, weight_cap = _extract_weighting(weighting_context or normalized)
    return MethodologyRules(
        index_name=_extract_index_name(normalized),
        updated=_extract_updated(normalized),
        methodology_family=_infer_family(normalized, etf_name=etf_name),
        review_frequency=_review_frequency(review_months, schedule_context),
        review_months=review_months,
        rebalance_timing=_extract_rebalance_timing(schedule_context),
        selection_count=_extract_selection_count(normalized),
        weighting_scheme=weighting_scheme,
        weight_cap=weight_cap,
        has_free_float_adjustment=_contains_any(
            normalized,
            ("유동시가총액", "유동주식", "유동비율", "Free Float", "free float"),
        ),
        has_market_cap_screen=_contains_any(universe_context or normalized, ("시가총액", "Market Cap", "market cap")),
        has_liquidity_screen=_contains_any(universe_context or normalized, ("거래대금", "유동성", "Liquidity", "liquidity")),
        has_keyword_filter=_contains_any(universe_context or normalized, ("키워드", "Keyword", "keyword")),
        has_fics_filter=_contains_any(universe_context or normalized, ("FICS", "Industry Classification")),
        evidence=evidence,
    )


def write_index_methodologies(factory: IndexMethodologyFactory, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rules.json"
    csv_path = output_dir / "rules.csv"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": factory.count,
        "downloaded_count": factory.downloaded_count,
        "pdf_read_count": factory.pdf_read_count,
        "items": [_to_jsonable(item) for item in factory.all()],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "code",
        "name",
        "status",
        "index_name",
        "updated",
        "methodology_family",
        "review_frequency",
        "review_months",
        "rebalance_timing",
        "selection_count",
        "weighting_scheme",
        "weight_cap",
        "has_free_float_adjustment",
        "has_market_cap_screen",
        "has_liquidity_screen",
        "has_keyword_filter",
        "has_fics_filter",
        "pdf_page_count",
        "pdf_text_chars",
        "source_url",
        "file_path",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in factory.all():
            writer.writerow(_flat_row(item))
    return csv_path, json_path


def _base_fields(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "code": str(record.get("code", "")),
        "name": str(record.get("name", "")),
        "status": str(record.get("status", "")),
        "source_url": str(record.get("source_url", "")),
        "page_url": str(record.get("page_url", "")),
        "file_path": str(record.get("file_path", "")),
        "sha256": str(record.get("sha256", "")),
        "bytes": int(record.get("bytes") or 0),
        "query": str(record.get("query", "")),
        "error": str(record.get("error", "")),
    }


def _extract_index_name(text: str) -> str:
    matches = list(re.finditer(
        r"((?:FnGuide|Maekyung|MKF|KRX|KOSPI|KOSDAQ)[\s\S]{0,180}?Index(?:\s+Series|\s+[A-Z]{1,3}|\s*\([^)]*\))?)\s+Methodology Book",
        text,
        flags=re.I,
    ))
    if matches:
        match = min(matches, key=lambda item: len(item.group(1)))
        name = _normalize_text(match.group(1))
    else:
        fallback = re.search(r"Methodology Book\s+((?:FnGuide|Maekyung|MKF)[^\n]{0,120}?(?:지수|Index(?:\s+Series)?))", text)
        if fallback:
            name = _normalize_text(fallback.group(1))
        else:
            title = re.search(r"((?:FnGuide|Maekyung|MKF)[\s\S]{0,180}?)\s+Methodology Book", text)
            if not title:
                return ""
            name = _normalize_text(title.group(1))
    if "FnGuide Inc." in name and "FnGuide " in name:
        name = name[name.rfind("FnGuide ") :]
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def _extract_updated(text: str) -> str:
    match = re.search(r"Updated\s+([A-Za-z]+\s+\d{4})", text)
    return match.group(1) if match else ""


def _extract_review_months(text: str) -> list[int]:
    if not text:
        return []
    schedule_patterns = (
        r"(?:매년|연)\s*(?:\d{1,2}\s*,\s*)*\d{1,2}\s*월[^.。\n]{0,120}(?:정기\s*(?:변경|개편)|리밸런싱)",
        r"(?:정기\s*(?:변경|개편)|리밸런싱)[^.。\n]{0,120}(?:매년|연)\s*(?:\d{1,2}\s*,\s*)*\d{1,2}\s*월",
    )
    focused = []
    for pattern in schedule_patterns:
        focused.extend(match.group(0) for match in re.finditer(pattern, text))
    target = " ".join(focused) if focused else text
    months = _month_values(target)
    return sorted(set(months))


def _month_values(text: str) -> list[int]:
    months = [int(value) for value in re.findall(r"(?<!\d)(1[0-2]|[1-9])\s*월", text)]
    for match in re.finditer(r"((?:1[0-2]|[1-9])(?:\s*,\s*(?:1[0-2]|[1-9]))+)\s*월", text):
        months.extend(int(value) for value in re.findall(r"1[0-2]|[1-9]", match.group(1)))
    return months


def _review_frequency(months: Sequence[int], text: str) -> str:
    if len(months) == 12:
        return "monthly"
    if len(months) == 4:
        return "quarterly"
    if len(months) == 2:
        return "semiannual"
    if len(months) == 1:
        return "annual"
    if "매월" in text:
        return "monthly"
    if "분기" in text or "연 4회" in text or "매년 4회" in text:
        return "quarterly"
    if "반기" in text or "연 2회" in text or "매년 2회" in text:
        return "semiannual"
    if "연 1회" in text or "매년 1회" in text:
        return "annual"
    return "unknown"


def _extract_rebalance_timing(text: str) -> str:
    if not text:
        return ""
    patterns = (
        r"(선물옵션 만기일\s*[^\.\n]{0,40})",
        r"(\d+\s*번째\s*\w*요일\s*[^\.\n]{0,40})",
        r"(\d+\s*영업일\s*[^\.\n]{0,40})",
        r"(마지막\s*영업일\s*[^\.\n]{0,40})",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return _snippet(match.group(1), limit=120)
    return _snippet(text, limit=120)


def _extract_weighting(text: str) -> tuple[str, str]:
    cap = ""
    cap_match = re.search(r"(?:비중상한|상한|cap|Cap|Ceiling|실링)[^\d%]{0,80}(\d+(?:\.\d+)?\s*%)", text)
    if cap_match:
        cap = cap_match.group(1).replace(" ", "")
    else:
        limit_match = re.search(r"비중[^\d%]{0,30}(\d+(?:\.\d+)?\s*%)[^.\n]{0,30}초과", text)
        if limit_match:
            cap = limit_match.group(1).replace(" ", "")
    if not cap and _contains_any(text, ("Ceiling", "실링")):
        fallback_cap = re.search(r"(\d+(?:\.\d+)?\s*%)", text)
        if fallback_cap:
            cap = fallback_cap.group(1).replace(" ", "")

    if _contains_any(text, ("동일가중", "동일 가중", "동일 비중", "Equal Weight", "equal weight")):
        return "equal_weighted", cap
    if _contains_any(text, ("스코어", "score")) and _contains_any(text, ("지수포함가중치", "유동주식비율", "유동비율")):
        return "score_weighted", cap
    if _contains_any(text, ("펀더멘탈 비중", "fundamental")) and _contains_any(text, ("유동주식비율", "유동비율")):
        return "fundamental_float_adjusted", cap
    if _contains_any(text, ("유동시총", "유동시가총액", "유동 시가총액", "float-adjusted", "free float market cap")):
        return "float_market_cap_weighted", cap
    if _contains_any(text, ("유동주식비율을 반영한 시가총액", "유동주식비율과 상장주식수")):
        return "float_market_cap_weighted", cap
    if _contains_any(text, ("지수포함비율", "지수 포함 비율", "IIF")):
        return "iif_adjusted", cap
    if _contains_any(text, ("MVt", "기준가 기준 시가총액", "시가총액")) and _contains_any(text, ("Wi", "비중을 적용")):
        return "market_cap_weighted", cap
    if _contains_any(text, ("시가총액 가중", "시가총액가중", "market capitalization weighted")):
        return "market_cap_weighted", cap
    if _contains_any(text, ("가중방식", "가중 방식", "weighted")):
        return "custom_weighted", cap
    return "unknown", cap


def _extract_selection_count(text: str) -> int | None:
    matches = [int(value) for value in re.findall(r"(?:TOP|Top|상위)\s*(\d+)", text)]
    return min(matches) if matches else None


def _infer_family(text: str, *, etf_name: str) -> str:
    haystack = f"{etf_name}\n{text}"
    if _contains_any(haystack, ("고배당", "High Dividend", "Dividend")):
        return "dividend"
    if _contains_any(haystack, ("동일가중", "Equal Weight")):
        return "equal_weight"
    if _contains_any(haystack, ("키워드", "Keyword")):
        return "keyword_theme"
    if _contains_any(haystack, ("FICS", "Industry Classification", "섹터", "Sector")):
        return "sector_theme"
    if _contains_any(haystack, ("TOP", "Top", "상위")):
        return "top_n_theme"
    return "theme"


def _section_text(text: str, *, starts: Sequence[str], stops: Sequence[str]) -> str:
    start_positions = [
        match.start()
        for start in starts
        for match in re.finditer(re.escape(start), text)
    ]
    if not start_positions:
        return ""
    start = next((position for position in sorted(start_positions) if position > 2500), max(start_positions))
    stop_candidates = [text.find(stop, start + 1) for stop in stops if text.find(stop, start + 1) >= 0]
    stop = min(stop_candidates) if stop_candidates else min(len(text), start + 2500)
    return text[start:stop]


def _near_keywords(text: str, keywords: Sequence[str], *, radius: int) -> str:
    chunks = []
    for keyword in keywords:
        position = text.find(keyword)
        if position >= 0:
            chunks.append(text[max(0, position - radius) : min(len(text), position + radius)])
    return "\n".join(chunks)


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def _normalize_text(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _snippet(text: str, *, limit: int = 260) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _to_jsonable(item: IndexMethodology) -> dict[str, object]:
    payload = asdict(item)
    payload["rule_set"] = asdict(item.rule_set)
    return payload


def _flat_row(item: IndexMethodology) -> dict[str, object]:
    rules = item.rules
    return {
        "code": item.code,
        "name": item.name,
        "status": item.status,
        "index_name": rules.index_name,
        "updated": rules.updated,
        "methodology_family": rules.methodology_family,
        "review_frequency": rules.review_frequency,
        "review_months": ",".join(str(month) for month in rules.review_months),
        "rebalance_timing": rules.rebalance_timing,
        "selection_count": rules.selection_count or "",
        "weighting_scheme": rules.weighting_scheme,
        "weight_cap": rules.weight_cap,
        "has_free_float_adjustment": rules.has_free_float_adjustment,
        "has_market_cap_screen": rules.has_market_cap_screen,
        "has_liquidity_screen": rules.has_liquidity_screen,
        "has_keyword_filter": rules.has_keyword_filter,
        "has_fics_filter": rules.has_fics_filter,
        "pdf_page_count": item.pdf_page_count,
        "pdf_text_chars": item.pdf_text_chars,
        "source_url": item.source_url,
        "file_path": item.file_path,
        "error": item.error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build index methodology objects from downloaded FnGuide PDFs.")
    parser.add_argument("--manifest", default=paths.FNGUIDE_PDFS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.FNGUIDE_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    factory = IndexMethodologyFactory.from_manifest(args.manifest)
    csv_path, json_path = write_index_methodologies(factory, Path(args.output_dir))
    print(
        f"processed {factory.count} methodologies; read {factory.pdf_read_count} PDFs; "
        f"wrote {csv_path} and {json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
