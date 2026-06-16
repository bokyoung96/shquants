from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Mapping


KSS_BUCKET_COUNTS = {"top2": 2, "momentum": 4, "market_cap_fill": 4}


def select_kss_buckets(rows: Iterable[Mapping[str, object]]) -> dict[str, list[dict[str, object]]]:
    candidates = [_kss_candidate(row) for row in rows if _truthy(row.get("is_eligible")) and _truthy(row.get("is_semiconductor_theme"))]
    if len(candidates) < 10:
        raise ValueError(f"KSS requires 10 eligible theme constituents, got {len(candidates)}")

    top2 = _take_ranked(candidates, count=2, metric="float_market_cap")
    remaining_after_top2 = _exclude(candidates, top2)
    momentum = _take_ranked(remaining_after_top2, count=4, metric="composite_momentum_score")
    remaining_after_momentum = _exclude(remaining_after_top2, momentum)
    market_cap_fill = _take_ranked(remaining_after_momentum, count=4, metric="float_market_cap")

    return {
        "top2": [_bucket_row(item, "top2", "float_market_cap") for item in top2],
        "momentum": [_bucket_row(item, "momentum", "composite_momentum_score") for item in momentum],
        "market_cap_fill": [_bucket_row(item, "market_cap_fill", "float_market_cap") for item in market_cap_fill],
    }


def _kss_candidate(row: Mapping[str, object]) -> dict[str, object]:
    code = str(row.get("security_code", "")).strip()
    if not code:
        raise ValueError("KSS candidate security_code is required")
    float_market_cap = _parse_metric(code, "float_market_cap", row.get("float_market_cap"))
    composite_momentum_score = _parse_metric(code, "composite_momentum_score", row.get("composite_momentum_score"))
    return {
        "as_of": str(row.get("as_of", "")),
        "security_code": code,
        "name": str(row.get("name", "")),
        "float_market_cap": float_market_cap,
        "composite_momentum_score": composite_momentum_score,
    }


def _take_ranked(rows: list[dict[str, object]], *, count: int, metric: str) -> list[dict[str, object]]:
    if len(rows) < count:
        raise ValueError(f"KSS {metric} bucket requires {count} constituents, got {len(rows)}")
    return sorted(
        rows,
        key=lambda row: (-float(row[metric]), -float(row["float_market_cap"]), str(row["security_code"])),
    )[:count]


def _exclude(rows: list[dict[str, object]], selected: list[dict[str, object]]) -> list[dict[str, object]]:
    selected_codes = {str(row["security_code"]) for row in selected}
    return [row for row in rows if str(row["security_code"]) not in selected_codes]


def _bucket_row(row: Mapping[str, object], bucket: str, rank_metric: str) -> dict[str, object]:
    return {
        "bucket": bucket,
        "rank_metric": rank_metric,
        "security_code": str(row["security_code"]),
        "name": str(row.get("name", "")),
        "float_market_cap": float(row["float_market_cap"]),
        "composite_momentum_score": float(row["composite_momentum_score"]),
    }


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        return math.isfinite(float(value)) and float(value) != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"", "false", "0", "no", "n"}:
            return False
        return False
    return False


def _is_missing_metric(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _parse_metric(code: str, metric: str, value: object) -> float:
    if _is_missing_metric(value):
        raise ValueError(f"{code} missing {metric}")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{code} invalid {metric}") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{code} invalid {metric}")
    if parsed <= 0:
        raise ValueError(f"{code} {metric} must be positive")
    return parsed
