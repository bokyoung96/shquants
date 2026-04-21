from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TradeCost:
    fee: float
    tax: float
    slippage: float

    @property
    def total(self) -> float:
        return self.fee + self.tax + self.slippage


@dataclass(frozen=True, slots=True)
class CostModel:
    fee: float = 0.0
    sell_tax: float = 0.0
    slippage: float = 0.0

    def calc(self, price: float, qty: float, side: str) -> TradeCost:
        if side not in {"buy", "sell"}:
            raise ValueError(f"unsupported side: {side}")

        gross = abs(price * qty)
        fee = gross * self.fee
        tax = gross * self.sell_tax if side == "sell" else 0.0
        slip = gross * self.slippage
        return TradeCost(fee=fee, tax=tax, slippage=slip)
