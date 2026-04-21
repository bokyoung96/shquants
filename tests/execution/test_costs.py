import pytest

from backtesting.execution.costs import CostModel, TradeCost


def test_cost_model_applies_fee_tax_and_slippage() -> None:
    model = CostModel(fee=0.001, sell_tax=0.002, slippage=0.001)
    cost = model.calc(price=100.0, qty=10.0, side="sell")

    assert isinstance(cost, TradeCost)
    assert round(cost.total, 4) == 4.0


def test_cost_model_does_not_apply_sell_tax_on_buy() -> None:
    model = CostModel(fee=0.001, sell_tax=0.002, slippage=0.001)

    cost = model.calc(price=100.0, qty=10.0, side="buy")

    assert cost.tax == 0.0
    assert round(cost.total, 4) == 2.0


def test_cost_model_rejects_invalid_side() -> None:
    model = CostModel(fee=0.001, sell_tax=0.002, slippage=0.001)

    with pytest.raises(ValueError, match="unsupported side: hold"):
        model.calc(price=100.0, qty=10.0, side="hold")
