from __future__ import annotations

from dataclasses import dataclass

from backtesting.signals.cashflow_inflection_core_band import CashflowInflectionCoreBandSignalProducer

from .event_core_band import EventCoreBandStrategy


@dataclass(slots=True)
class CashflowInflectionCoreBandTopN(EventCoreBandStrategy):
    top_n: int = 10
    core_fraction: float = 0.85
    active_fractions: tuple[float, ...] = (0.05, 0.05, 0.05)
    fundamental_lookback: int = 63
    support_momentum_lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = CashflowInflectionCoreBandSignalProducer(
            fundamental_lookback=self.fundamental_lookback,
            support_momentum_lookback=self.support_momentum_lookback,
        )
        EventCoreBandStrategy.__post_init__(self)
