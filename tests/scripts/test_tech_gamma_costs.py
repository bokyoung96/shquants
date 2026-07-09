from __future__ import annotations

from scripts.tech_gamma_costs import (
    COMMISSION_BPS_PER_SIDE,
    ROUND_TRIP_COST_BPS,
    SELL_TAX_BPS,
    SLIPPAGE_BPS,
    net_return_after_costs,
)


def test_round_trip_cost_includes_commission_tax_and_slippage() -> None:
    assert COMMISSION_BPS_PER_SIDE == 2.0
    assert SELL_TAX_BPS == 20.0
    assert SLIPPAGE_BPS == 11.0
    assert ROUND_TRIP_COST_BPS == 35.0
    assert net_return_after_costs(0.10) == 0.0965
