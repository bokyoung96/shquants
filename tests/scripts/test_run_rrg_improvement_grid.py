from __future__ import annotations


def test_rrg_improvement_grid_has_5000_unique_prefixed_variants() -> None:
    from scripts.run_rrg_improvement_grid import variant_grid

    variants = variant_grid()
    ids = [variant.id for variant in variants]

    assert len(variants) == 5000
    assert len(set(ids)) == 5000
    assert all(identifier.startswith("rrgi_") for identifier in ids)


def test_rrg_improvement_grid_uses_fixed_candidate_dimensions() -> None:
    from scripts.run_rrg_improvement_grid import variant_grid

    variants = variant_grid()

    assert {variant.score_mode for variant in variants} == {"qavg", "op12", "blend", "accel", "eps_op"}
    assert {variant.event_mode for variant in variants} == {
        "none",
        "accel",
        "cross_up",
        "sector_turn",
        "price_turn",
        "new_high",
        "reclaim",
        "vol_break",
        "drawdown_repair",
        "op_lead",
    }
    assert {variant.flow_gate for variant in variants} == {
        "none",
        "smart",
        "foreign",
        "inst",
        "retail_contra",
        "smart_accel",
        "foreign_accel",
        "inst_accel",
        "flow_breadth",
        "anti_retail_breadth",
    }
    assert len({variant.construction_mode for variant in variants}) == 10


def test_rrg_hurdle_requires_beating_existing_rrg_leaders() -> None:
    from scripts.run_rrg_improvement_grid import rrg_hurdle

    hurdle = rrg_hurdle()

    assert hurdle["min_cagr"] >= 0.956171
    assert hurdle["min_sharpe"] >= 2.302752
    assert hurdle["min_mdd"] >= -0.166760


def test_select_outperformers_rejects_weaker_candidates() -> None:
    from scripts.run_rrg_improvement_grid import select_outperformers

    rows = [
        {"strategy_id": "bad", "cagr": 1.0, "mdd": -0.30, "sharpe": 2.50},
        {"strategy_id": "also_bad", "cagr": 0.90, "mdd": -0.10, "sharpe": 2.50},
        {"strategy_id": "good", "cagr": 1.0, "mdd": -0.10, "sharpe": 2.40},
    ]

    selected = select_outperformers(rows)

    assert [row["strategy_id"] for row in selected] == ["good"]
