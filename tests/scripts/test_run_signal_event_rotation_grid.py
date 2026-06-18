from __future__ import annotations


def test_signal_event_grid_has_500_unique_prefixed_variants() -> None:
    from scripts.run_signal_event_rotation_grid import variant_grid

    variants = variant_grid()
    ids = [variant.id for variant in variants]

    assert len(variants) == 500
    assert len(set(ids)) == 500
    assert all(identifier.startswith("sev_") for identifier in ids)


def test_signal_event_grid_uses_fixed_candidate_dimensions() -> None:
    from scripts.run_signal_event_rotation_grid import variant_grid

    variants = variant_grid()

    assert {variant.score_mode for variant in variants} == {"qavg", "op12", "blend", "accel", "eps_op"}
    assert {variant.event_mode for variant in variants} == {"cross_up", "accel", "sector_turn", "new_high", "reclaim"}
    assert {variant.flow_gate for variant in variants} == {"none", "smart", "foreign", "inst", "retail_contra"}
    assert {variant.construction_mode for variant in variants} == {"k1", "k2", "k3", "breadth"}
    assert {variant.risk_mode for variant in variants} == {"lo", "ls02", "ls03", "ls05", "ls07"}
