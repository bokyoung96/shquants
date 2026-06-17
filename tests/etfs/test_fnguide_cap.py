from etfs.common.cap import CapPolicy, cap_policy_from_mapping, resolve_special_cap_events, select_cap_candidates
from etfs.common.holdings import ValidationHolding, ValidationSnapshot


def test_select_cap_candidates_flags_regular_25_percent_excess() -> None:
    policy = CapPolicy(
        index_code="FI00.WLT.NMS",
        index_name="Example TOP2+ Index",
        methodology_version="v2.5",
        regular_security_cap=0.25,
        special_trigger_weight=0.30,
        special_target_cap=0.27,
        special_check_excluded_months=(5, 11),
        special_effective_lag_business_days=3,
    )
    snapshot = ValidationSnapshot(
        as_of="2026-06-16",
        equity_holdings=[
            ValidationHolding("009150", "A009150", "삼성전기", 161, 329728000, 0.2575),
            ValidationHolding("000660", "A000660", "SK하이닉스", 133, 316806000, 0.2474),
        ],
        cash={"weight": 0.0013},
    )

    candidates = select_cap_candidates(snapshot, policy)

    assert len(candidates) == 1
    assert candidates[0].event_type == "regular_cap_excess"
    assert candidates[0].security_code == "009150"
    assert candidates[0].name == "삼성전기"
    assert candidates[0].weight == 0.2575
    assert candidates[0].cap == 0.25
    assert candidates[0].excess_weight == 0.0075


def test_resolve_special_cap_events_uses_month_end_trigger_and_t_plus_3() -> None:
    policy = cap_policy_from_mapping(
        {
            "index_code": "FI00.WLT.NMS",
            "index_name": "Example TOP2+ Index",
            "methodology_version": "v2.5",
            "regular_cap": {"max_security_weight": 0.25},
            "special_cap": {
                "check_months_excluding": [5, 11],
                "trigger_if_any_security_above": 0.30,
                "target_max_weight": 0.27,
                "effective_lag_business_days": 3,
            },
        }
    )
    snapshot = ValidationSnapshot(
        as_of="2026-06-30",
        equity_holdings=[
            ValidationHolding("000660", "A000660", "SK하이닉스", 133, 400000000, 0.312),
            ValidationHolding("009150", "A009150", "삼성전기", 161, 300000000, 0.234),
        ],
        cash={"weight": 0.001},
    )

    events = resolve_special_cap_events(
        [snapshot],
        policy,
        trading_dates=["2026-06-30", "2026-07-01", "2026-07-02", "2026-07-03"],
    )

    assert len(events) == 1
    assert events[0].event_type == "special_cap_trigger"
    assert events[0].as_of == "2026-06-30"
    assert events[0].effective_date == "2026-07-03"
    assert events[0].security_code == "000660"
    assert events[0].cap == 0.27
    assert events[0].excess_weight == 0.042


def test_resolve_special_cap_events_does_not_need_future_calendar_without_trigger() -> None:
    policy = cap_policy_from_mapping(
        {
            "index_code": "GENERIC",
            "index_name": "Generic capped index",
            "special_cap": {
                "check_months_excluding": [],
                "trigger_if_any_security_above": 0.30,
                "target_max_weight": 0.27,
                "effective_lag_business_days": 3,
            },
        }
    )
    snapshot = ValidationSnapshot(
        as_of="2026-06-16",
        equity_holdings=[
            ValidationHolding("000001", "A000001", "Below Trigger", 1, 1, 0.299),
        ],
        cash={},
    )

    assert resolve_special_cap_events([snapshot], policy, trading_dates=["2026-06-16"]) == []
