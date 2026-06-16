from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths


SPGLOBAL_METHODOLOGY_LIBRARY_URL = "https://www.spglobal.com/spdji/en/governance/methodologies/"
SP_US_INDICES_METHODOLOGY_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-indices.pdf"
SP_EQUAL_WEIGHT_METHODOLOGY_URL = "https://www.spice-indices.com/idpfiles/spice-assets/resources/public/documents/methodology-sp-500-equal-weight-index.pdf"
SP_LOW_VOL_METHODOLOGY_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-500-low-volatility-index.pdf"
SP_STYLE_METHODOLOGY_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-us-style.pdf"
SP_GLOBAL_INFRA_METHODOLOGY_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-thematic-indices.pdf"
SP_SELECT_INDUSTRY_METHODOLOGY_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-select-industry-indices.pdf"
SP_KENSHO_METHODOLOGY_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-kensho-new-economies.pdf"
SP_DIVIDEND_KINGS_METHODOLOGY_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-dividend-kings-index.pdf"
SP_GSCI_METHODOLOGY_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-gsci.pdf"
SP_INDEX_MATH_METHODOLOGY_URL = "https://www.spglobal.com/spdji/en/documents/methodologies/methodology-index-math.pdf"


@dataclass(frozen=True, slots=True)
class SpGlobalMethodologyProbe:
    code: str
    name: str
    product_family: str
    source_confidence: str
    index_name_candidates: list[str]
    methodology_probe: str
    methodology_url_candidates: list[str]
    next_action: str


def load_source_items(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"unsupported sources payload: {path}")
    return list(payload["items"])


def build_spglobal_records(items: Iterable[Mapping[str, object]]) -> list[SpGlobalMethodologyProbe]:
    records: list[SpGlobalMethodologyProbe] = []
    for item in items:
        if str(item.get("source_candidate", "")) != "sp_global":
            continue
        name = str(item.get("name", ""))
        candidates = infer_spglobal_index_candidates(name)
        records.append(
            SpGlobalMethodologyProbe(
                code=str(item.get("code", "")),
                name=name,
                product_family=str(item.get("product_family", "")),
                source_confidence=str(item.get("confidence", "")),
                index_name_candidates=candidates,
                methodology_probe="spglobal_methodology_library",
                methodology_url_candidates=infer_methodology_urls(candidates),
                next_action="probe_spglobal_methodology_library" if candidates else "manual_spglobal_index_mapping",
            )
        )
    return records


def infer_spglobal_index_candidates(name: str) -> list[str]:
    if "럭셔리" in name:
        return ["S&P Global Luxury"]
    if "바이오" in name:
        return ["S&P Biotechnology Select Industry"]
    if "스마트모빌리티" in name:
        return ["S&P Kensho Smart Transportation"]
    if "배당킹" in name:
        return ["S&P Dividend Kings"]
    if "원유생산기업" in name:
        return ["S&P Oil & Gas Exploration & Production Select Industry"]
    if "탄소배출권" in name:
        return ["S&P GSCI Carbon Emission Allowances"]
    if "글로벌인프라" in name:
        return ["S&P Global Infrastructure"]
    if "로우볼" in name or "저변동" in name:
        return ["S&P Korea Low Volatility"]
    if "동일가중" in name:
        return ["S&P 500 Equal Weight"]
    if "배당귀족" in name:
        return ["S&P 500 Dividend Aristocrats"]
    if "성장주" in name:
        return ["S&P 500 Growth"]
    if _contains_any(name, ("금융", "경기소비재", "산업재", "에너지", "유틸리티", "커뮤니케이션", "테크놀로지", "필수소비재", "헬스케어")):
        return ["S&P 500 Sector"]
    if "S&P500" in name or "S&P 500" in name:
        return ["S&P 500"]
    return []


def infer_methodology_urls(index_name_candidates: list[str]) -> list[str]:
    urls: list[str] = []
    for candidate in index_name_candidates:
        if candidate in {"S&P 500", "S&P 500 Sector", "S&P 500 Dividend Aristocrats"}:
            urls.append(SP_US_INDICES_METHODOLOGY_URL)
        elif candidate == "S&P 500 Equal Weight":
            urls.extend([SP_US_INDICES_METHODOLOGY_URL, SP_EQUAL_WEIGHT_METHODOLOGY_URL])
        elif candidate == "S&P Korea Low Volatility":
            urls.append(SP_LOW_VOL_METHODOLOGY_URL)
        elif candidate == "S&P 500 Growth":
            urls.extend([SP_US_INDICES_METHODOLOGY_URL, SP_STYLE_METHODOLOGY_URL])
        elif candidate == "S&P Global Infrastructure":
            urls.append(SP_GLOBAL_INFRA_METHODOLOGY_URL)
        elif candidate in {"S&P Global Luxury"}:
            urls.append(SP_GLOBAL_INFRA_METHODOLOGY_URL)
        elif candidate in {"S&P Biotechnology Select Industry", "S&P Oil & Gas Exploration & Production Select Industry"}:
            urls.append(SP_SELECT_INDUSTRY_METHODOLOGY_URL)
        elif candidate == "S&P Kensho Smart Transportation":
            urls.append(SP_KENSHO_METHODOLOGY_URL)
        elif candidate == "S&P Dividend Kings":
            urls.append(SP_DIVIDEND_KINGS_METHODOLOGY_URL)
        elif candidate == "S&P GSCI Carbon Emission Allowances":
            urls.append(SP_GSCI_METHODOLOGY_URL)
    if urls:
        urls.append(SP_INDEX_MATH_METHODOLOGY_URL)
    return _dedupe(urls)


def write_spglobal_manifest(sources_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    records = build_spglobal_records(load_source_items(sources_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "spglobal.csv"
    json_path = output_dir / "spglobal.json"
    md_path = output_dir / "spglobal.md"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_csv_row(records[0]).keys()) if records else list(SpGlobalMethodologyProbe.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))

    payload = {
        "provider": "sp_global",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(records),
        "mapped_count": sum(bool(record.index_name_candidates) for record in records),
        "index_candidate_counts": _index_candidate_counts(records),
        "methodology_library_url": SPGLOBAL_METHODOLOGY_LIBRARY_URL,
        "items": [asdict(record) for record in records],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(records), encoding="utf-8")
    return csv_path, json_path, md_path


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _csv_row(record: SpGlobalMethodologyProbe) -> dict[str, object]:
    row = asdict(record)
    row["index_name_candidates"] = "|".join(record.index_name_candidates)
    row["methodology_url_candidates"] = "|".join(record.methodology_url_candidates)
    return row


def _index_candidate_counts(records: list[SpGlobalMethodologyProbe]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        for candidate in record.index_name_candidates:
            counter[candidate] += 1
    return dict(counter)


def _markdown_summary(records: list[SpGlobalMethodologyProbe]) -> str:
    candidate_counts = _index_candidate_counts(records)
    unmapped = [record for record in records if not record.index_name_candidates]
    lines = [
        "# S&P Global methodology probe manifest",
        "",
        f"- ETFs covered: {len(records)}",
        f"- Index candidates mapped: {len(records) - len(unmapped)}",
        f"- Manual S&P mappings: {len(unmapped)}",
        f"- Methodology library: {SPGLOBAL_METHODOLOGY_LIBRARY_URL}",
        "",
        "## Index Candidates",
        "",
    ]
    for name, count in sorted(candidate_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {name}: {count}")

    lines.extend(["", "## Manual S&P Mapping", ""])
    if unmapped:
        lines.extend(["| code | name | family |", "| --- | --- | --- |"])
        for record in unmapped:
            lines.append(f"| {record.code} | {record.name} | {record.product_family} |")
    else:
        lines.append("- No S&P records require manual index mapping.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build S&P Global methodology probe manifest from source candidates.")
    parser.add_argument("--sources", default=paths.SOURCES_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.SPGLOBAL_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path, json_path, md_path = write_spglobal_manifest(Path(args.sources), Path(args.output_dir))
    print(f"wrote {csv_path}, {json_path}, and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
