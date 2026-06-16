import pytest

from etfs.fnguide.selection import select_kss_buckets


def _row(code: str, float_cap: float, momentum: float, *, eligible: bool = True, theme: bool = True) -> dict[str, object]:
    return {
        "as_of": "2026-05-29",
        "security_code": code,
        "name": code,
        "is_eligible": eligible,
        "is_semiconductor_theme": theme,
        "float_market_cap": float_cap,
        "composite_momentum_score": momentum,
    }


def test_select_kss_buckets_builds_top2_momentum_and_fill() -> None:
    rows = [
        _row("A000001", 1000, 1),
        _row("A000002", 900, 2),
        _row("A000003", 800, 90),
        _row("A000004", 700, 80),
        _row("A000005", 600, 70),
        _row("A000006", 500, 60),
        _row("A000007", 400, 10),
        _row("A000008", 300, 20),
        _row("A000009", 200, 30),
        _row("A000010", 100, 40),
        _row("A000011", 50, 5),
    ]

    buckets = select_kss_buckets(rows)

    assert [item["security_code"] for item in buckets["top2"]] == ["A000001", "A000002"]
    assert [item["security_code"] for item in buckets["momentum"]] == ["A000003", "A000004", "A000005", "A000006"]
    assert [item["security_code"] for item in buckets["market_cap_fill"]] == ["A000007", "A000008", "A000009", "A000010"]


def test_select_kss_buckets_excludes_ineligible_and_non_theme_names() -> None:
    rows = [
        _row("A000001", 1000, 1),
        _row("A000002", 900, 2),
        _row("A000003", 850, 99, eligible=False),
        _row("A000004", 840, 98, theme=False),
        _row("A000005", 800, 90),
        _row("A000006", 700, 80),
        _row("A000007", 600, 70),
        _row("A000008", 500, 60),
        _row("A000009", 400, 50),
        _row("A000010", 300, 40),
        _row("A000011", 200, 30),
        _row("A000012", 100, 20),
    ]

    buckets = select_kss_buckets(rows)
    selected_codes = {item["security_code"] for members in buckets.values() for item in members}

    assert "A000003" not in selected_codes
    assert "A000004" not in selected_codes
    assert len(selected_codes) == 10


def test_select_kss_buckets_uses_deterministic_tie_breakers() -> None:
    rows = [
        _row("A000002", 1000, 1),
        _row("A000001", 1000, 1),
        _row("A000003", 800, 50),
        _row("A000004", 700, 50),
        _row("A000005", 600, 50),
        _row("A000006", 500, 50),
        _row("A000007", 400, 40),
        _row("A000008", 300, 30),
        _row("A000009", 200, 20),
        _row("A000010", 100, 10),
    ]

    buckets = select_kss_buckets(rows)

    assert [item["security_code"] for item in buckets["top2"]] == ["A000001", "A000002"]
    assert [item["security_code"] for item in buckets["momentum"]] == ["A000003", "A000004", "A000005", "A000006"]


def test_select_kss_buckets_rejects_missing_metric() -> None:
    rows = [_row(f"A{i:06d}", 1000 - i, 100 - i) for i in range(10)]
    rows[3]["composite_momentum_score"] = None

    with pytest.raises(ValueError, match="A000003 missing composite_momentum_score"):
        select_kss_buckets(rows)


def test_select_kss_buckets_accepts_truthy_non_boolean_flags() -> None:
    rows = [
        _row("A000001", 1000, 1, eligible=2, theme=3),
        _row("A000002", 900, 2),
        _row("A000003", 800, 90),
        _row("A000004", 700, 80),
        _row("A000005", 600, 70),
        _row("A000006", 500, 60),
        _row("A000007", 400, 10),
        _row("A000008", 300, 20),
        _row("A000009", 200, 30),
        _row("A000010", 100, 40),
    ]

    buckets = select_kss_buckets(rows)

    assert [item["security_code"] for item in buckets["top2"]] == ["A000001", "A000002"]


def test_select_kss_buckets_treats_whitespace_metric_as_missing() -> None:
    rows = [_row(f"A{i:06d}", 1000 - i, 100 - i) for i in range(10)]
    rows[3]["composite_momentum_score"] = "   "

    with pytest.raises(ValueError, match="A000003 missing composite_momentum_score"):
        select_kss_buckets(rows)


def test_select_kss_buckets_rejects_insufficient_candidates() -> None:
    rows = [_row(f"A{i:06d}", 1000 - i, 100 - i) for i in range(9)]

    with pytest.raises(ValueError, match="KSS requires 10 eligible theme constituents"):
        select_kss_buckets(rows)
