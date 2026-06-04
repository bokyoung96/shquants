from __future__ import annotations

from typing import Protocol

import pandas as pd

try:
    from ...core.mapping import Linker
except ImportError:  # pragma: no cover - direct script compatibility
    from core.mapping import Linker


class Builder(Protocol):
    def current(self, names: pd.DataFrame, factset: pd.DataFrame) -> pd.DataFrame:
        ...

    def history(self, names: pd.DataFrame, factset: pd.DataFrame) -> pd.DataFrame:
        ...

    def at(self, date: str, history: pd.DataFrame) -> pd.DataFrame:
        ...

    def latest_rows(self, history: pd.DataFrame) -> pd.DataFrame:
        ...


class UniverseBuilder:
    name = "builder"

    columns = [
        "permno",
        "permco",
        "market",
        "exchange",
        "primaryexch",
        "crsp_ticker",
        "trade_ticker",
        "factset_ticker",
        "ticker_exchange",
        "company",
        "hdrcusip",
        "hdrcusip9",
        "cusip",
        "cusip9",
        "shareclass",
        "sharetype",
        "securitytype",
        "securitysubtype",
        "usincflg",
        "issuertype",
        "siccd",
        "conditionaltype",
        "tradingstatusflg",
        "fsym_regional_id",
        "fsym_security_id",
        "factset_entity_id",
        "namedt",
        "nameendt",
        "securitybegdt",
        "securityenddt",
        "link_bdate",
        "link_edate",
        "start_date",
        "end_date",
    ]

    def __init__(self) -> None:
        self.linker = Linker(self.columns)

    def current(self, names: pd.DataFrame, factset: pd.DataFrame) -> pd.DataFrame:
        return self.linker.current(
            names,
            factset,
            key="permno",
            left_start="namedt",
            left_end="nameendt",
            right_start="link_bdate",
            right_end="link_edate",
        )

    def history(self, names: pd.DataFrame, factset: pd.DataFrame) -> pd.DataFrame:
        return self.linker.history(
            names,
            factset,
            key="permno",
            left_start="namedt",
            left_end="nameendt",
            right_start="link_bdate",
            right_end="link_edate",
            desc="join history",
        )

    def at(self, date: str, history: pd.DataFrame) -> pd.DataFrame:
        return self.linker.at(history, date)

    def latest_rows(self, history: pd.DataFrame) -> pd.DataFrame:
        return self.linker.latest(history)
