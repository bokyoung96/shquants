from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from etfs import paths
from etfs.common.holdings import ValidationSnapshot, load_validation_fixtures


@dataclass(frozen=True, slots=True)
class CapPolicy:
    index_code: str
    index_name: str
    methodology_version: str
    regular_security_cap: float
    special_trigger_weight: float
    special_target_cap: float
    special_check_excluded_months: tuple[int, ...]
    special_effective_lag_business_days: int


@dataclass(frozen=True, slots=True)
class CapCandidate:
    event_type: str
    as_of: str
    security_code: str
    name: str
    quantity: float
    market_value: float
    weight: float
    cap: float
    excess_weight: float
    effective_date: str = ""


def cap_policy_from_mapping(value: dict[str, object]) -> CapPolicy:
    regular_cap = _mapping(value.get("regular_cap"))
    special_cap = _mapping(value.get("special_cap"))
    return CapPolicy(
        index_code=str(value.get("index_code", "")),
        index_name=str(value.get("index_name", "")),
        methodology_version=str(value.get("methodology_version", "")),
        regular_security_cap=float(regular_cap.get("max_security_weight", 0.0) or 0.0),
        special_trigger_weight=float(special_cap.get("trigger_if_any_security_above", 0.0) or 0.0),
        special_target_cap=float(special_cap.get("target_max_weight", 0.0) or 0.0),
        special_check_excluded_months=tuple(int(month) for month in special_cap.get("check_months_excluding", [])),
        special_effective_lag_business_days=int(special_cap.get("effective_lag_business_days", 0) or 0),
    )


def cap_policy_from_methodology_spec(spec: Mapping[str, object]) -> CapPolicy:
    cap_values = _security_level_caps(spec)
    regular_security_cap = max(cap_values, default=0.0)
    special_cap = _mapping(spec.get("special_cap"))
    return CapPolicy(
        index_code=str(spec.get("index_code", "")),
        index_name=str(spec.get("index_name", "")),
        methodology_version=str(spec.get("methodology_version", "")),
        regular_security_cap=regular_security_cap,
        special_trigger_weight=float(special_cap.get("trigger_if_any_security_above", 0.0) or 0.0),
        special_target_cap=float(special_cap.get("target_max_weight", 0.0) or 0.0),
        special_check_excluded_months=tuple(int(month) for month in special_cap.get("check_months_excluding", [])),
        special_effective_lag_business_days=int(special_cap.get("effective_lag_business_days", 0) or 0),
    )


def select_cap_candidates(snapshot: ValidationSnapshot, policy: CapPolicy) -> list[CapCandidate]:
    if policy.regular_security_cap <= 0:
        return []
    candidates: list[CapCandidate] = []
    for holding in snapshot.equity_holdings:
        if holding.weight > policy.regular_security_cap:
            candidates.append(
                CapCandidate(
                    event_type="regular_cap_excess",
                    as_of=snapshot.as_of,
                    security_code=holding.ticker,
                    name=holding.name,
                    quantity=holding.quantity,
                    market_value=holding.amount,
                    weight=holding.weight,
                    cap=policy.regular_security_cap,
                    excess_weight=round(holding.weight - policy.regular_security_cap, 12),
                )
            )
    return sorted(candidates, key=lambda item: item.excess_weight, reverse=True)


def resolve_special_cap_events(
    snapshots: Iterable[ValidationSnapshot],
    policy: CapPolicy,
    *,
    trading_dates: Iterable[str],
) -> list[CapCandidate]:
    date_list = sorted(set(trading_dates))
    events: list[CapCandidate] = []
    for snapshot in _latest_snapshot_by_month(snapshots):
        snapshot_date = date.fromisoformat(snapshot.as_of)
        if snapshot_date.month in policy.special_check_excluded_months:
            continue
        triggered_holdings = [
            holding
            for holding in snapshot.equity_holdings
            if holding.weight > policy.special_trigger_weight
        ]
        if not triggered_holdings:
            continue
        effective_date = _business_day_offset(
            snapshot.as_of,
            date_list,
            policy.special_effective_lag_business_days,
        )
        for holding in triggered_holdings:
            events.append(
                CapCandidate(
                    event_type="special_cap_trigger",
                    as_of=snapshot.as_of,
                    security_code=holding.ticker,
                    name=holding.name,
                    quantity=holding.quantity,
                    market_value=holding.amount,
                    weight=holding.weight,
                    cap=policy.special_target_cap,
                    excess_weight=round(holding.weight - policy.special_target_cap, 12),
                    effective_date=effective_date,
                )
            )
    return sorted(events, key=lambda item: (item.as_of, item.security_code))


def build_cap_candidate_report(
    fixtures: Iterable[object],
    specs: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    policies = {
        policy.index_code: policy
        for policy in (cap_policy_from_methodology_spec(spec) for spec in specs)
        if policy.index_code and policy.regular_security_cap > 0
    }
    items: list[dict[str, object]] = []
    for fixture in fixtures:
        index_code = str(getattr(fixture, "index_code", ""))
        policy = policies.get(index_code)
        snapshots = list(getattr(fixture, "snapshots", []))
        if policy is None or not snapshots:
            continue
        latest = max(snapshots, key=lambda snapshot: snapshot.as_of)
        for candidate in select_cap_candidates(latest, policy):
            items.append(
                {
                    "etf_code": str(getattr(fixture, "etf_code", "")),
                    "etf_name": str(getattr(fixture, "etf_name", "")),
                    "index_code": policy.index_code,
                    "index_name": policy.index_name,
                    "as_of": candidate.as_of,
                    "event_type": candidate.event_type,
                    "security_code": candidate.security_code,
                    "security_name": candidate.name,
                    "quantity": candidate.quantity,
                    "market_value": candidate.market_value,
                    "weight": candidate.weight,
                    "cap": candidate.cap,
                    "excess_weight": candidate.excess_weight,
                    "effective_date": candidate.effective_date,
                }
            )
    items.sort(key=lambda item: (str(item["as_of"]), str(item["etf_code"]), -float(item["excess_weight"]), str(item["security_code"])))
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(items),
        "items": items,
    }


def write_cap_candidate_report(fixtures_path: Path, specs_path: Path, output_dir: Path) -> tuple[Path, Path]:
    specs_payload = json.loads(specs_path.read_text(encoding="utf-8"))
    fixtures = load_validation_fixtures(fixtures_path)
    specs = [item for item in specs_payload.get("indices", []) if isinstance(item, Mapping)]
    report = build_cap_candidate_report(fixtures, specs)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / paths.CAP_CANDIDATES_JSON.name
    markdown_path = output_dir / paths.CAP_CANDIDATES_MD.name
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_cap_candidate_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select cap breach candidates from ETF holdings validation fixtures.")
    parser.add_argument("--fixtures", default=paths.VALIDATION_FIXTURES_JSON.as_posix())
    parser.add_argument("--specs", default=paths.FNGUIDE_METHODOLOGY_SPECS_JSON.as_posix())
    parser.add_argument("--output-dir", default=paths.VALIDATION_OUTPUT_DIR.as_posix())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    json_path, markdown_path = write_cap_candidate_report(
        Path(args.fixtures),
        Path(args.specs),
        Path(args.output_dir),
    )
    print(f"wrote {json_path} and {markdown_path}")
    return 0


def _latest_snapshot_by_month(snapshots: Iterable[ValidationSnapshot]) -> list[ValidationSnapshot]:
    by_month: dict[str, ValidationSnapshot] = {}
    for snapshot in snapshots:
        month_key = snapshot.as_of[:7]
        previous = by_month.get(month_key)
        if previous is None or snapshot.as_of > previous.as_of:
            by_month[month_key] = snapshot
    return [by_month[key] for key in sorted(by_month)]


def _business_day_offset(as_of: str, trading_dates: list[str], offset: int) -> str:
    try:
        index = trading_dates.index(as_of)
    except ValueError as exc:
        raise ValueError(f"trading calendar does not contain {as_of}") from exc
    target_index = index + offset
    if target_index >= len(trading_dates):
        raise ValueError(f"trading calendar does not contain T+{offset} for {as_of}")
    return trading_dates[target_index]


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _security_level_caps(spec: Mapping[str, object]) -> list[float]:
    values: list[float] = []
    weighting = _mapping(spec.get("weighting"))
    _append_positive_float(values, weighting.get("security_cap"))
    _append_positive_float(values, _mapping(weighting.get("residual")).get("cap"))

    selection = _mapping(spec.get("selection"))
    buckets = selection.get("buckets", [])
    if isinstance(buckets, list):
        for bucket in buckets:
            if not isinstance(bucket, Mapping):
                continue
            weight = _mapping(bucket.get("weight"))
            if weight.get("type") == "fixed":
                _append_positive_float(values, weight.get("value"))
    return values


def _append_positive_float(values: list[float], value: object) -> None:
    if value is None:
        return
    number = float(value)
    if number > 0:
        values.append(number)


def _cap_candidate_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Cap Candidates",
        "",
        f"- Count: {report.get('count', 0)}",
        "",
        "| ETF | Index | As of | Security | Weight | Cap | Excess | Quantity | Market value | Event |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    items = report.get("items", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                "| {etf} | {index} | {as_of} | {security} | {weight} | {cap} | {excess} | {quantity} | {market_value} | {event} |".format(
                    etf=_markdown_cell(f"{item.get('etf_code', '')} {item.get('etf_name', '')}".strip()),
                    index=_markdown_cell(f"{item.get('index_code', '')} {item.get('index_name', '')}".strip()),
                    as_of=_markdown_cell(item.get("as_of", "")),
                    security=_markdown_cell(f"{item.get('security_code', '')} {item.get('security_name', '')}".strip()),
                    weight=_markdown_cell(item.get("weight", "")),
                    cap=_markdown_cell(item.get("cap", "")),
                    excess=_markdown_cell(item.get("excess_weight", "")),
                    quantity=_markdown_cell(item.get("quantity", "")),
                    market_value=_markdown_cell(item.get("market_value", "")),
                    event=_markdown_cell(item.get("event_type", "")),
                )
            )
    return "\n".join(lines) + "\n"


def _markdown_cell(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
