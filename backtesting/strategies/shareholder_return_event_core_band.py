from __future__ import annotations

from dataclasses import dataclass

from backtesting.signals.shareholder_return_event_core_band import ShareholderReturnEventCoreBandSignalProducer

from .event_core_band import EventCoreBandStrategy


@dataclass(slots=True)
class ShareholderReturnEventCoreBandTopN(EventCoreBandStrategy):
    top_n: int = 10
    core_fraction: float = 0.85
    active_fractions: tuple[float, ...] = (0.05, 0.05, 0.05)
    buyback_threshold: float = -0.01
    flow_lookback: int = 20

    def __post_init__(self) -> None:
        self.signal_producer = ShareholderReturnEventCoreBandSignalProducer(
            buyback_threshold=self.buyback_threshold,
            flow_lookback=self.flow_lookback,
        )
        EventCoreBandStrategy.__post_init__(self)
