from __future__ import annotations


COMMISSION_BPS_PER_SIDE = 2.0
SELL_TAX_BPS = 20.0
ROUND_TRIP_COST_BPS = COMMISSION_BPS_PER_SIDE * 2.0 + SELL_TAX_BPS


def net_return_after_costs(gross_return: float) -> float:
    return gross_return - ROUND_TRIP_COST_BPS / 10_000.0
