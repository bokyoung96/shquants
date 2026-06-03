from __future__ import annotations

from datetime import date
from typing import Iterable, Protocol

from .source import SINCE, Plan, Source, Table


class Named(Protocol):
    name: str


class Registry:
    def __init__(self, sources: Iterable[Source]) -> None:
        self.sources = tuple(sources)
        self._rank = {str(source.rank): source for source in self.sources}
        self._name = {source.name: source for source in self.sources}

    @classmethod
    def default(cls) -> "Registry":
        end = date.today().year
        return cls(
            [
                Source(
                    1,
                    "crsp",
                    "Current CRSP CIZ US equity prices, returns, and identifiers.",
                    (
                        Table("stkdlysecuritydata", "Daily CIZ stock prices and returns.", "dlycaldt", SINCE, end),
                        Table("stkmthsecuritydata", "Monthly CIZ stock prices and returns.", "mthcaldt", SINCE, end),
                        Table("stksecurityinfohist", "CIZ security identifier history."),
                        Table("msedelist", "Monthly delisting events.", "dlstdt", SINCE, end),
                        Table("dsedelist", "Daily delisting events.", "dlstdt", SINCE, end),
                        Table("dsi", "Daily market indexes.", "date", SINCE, end),
                    ),
                ),
                Source(
                    2,
                    "comp",
                    "Compustat fundamentals and company/security metadata.",
                    (
                        Table("funda", "Annual fundamentals.", "datadate", SINCE, end),
                        Table("fundq", "Quarterly fundamentals.", "datadate", SINCE, end),
                        Table("company", "Company metadata."),
                        Table("security", "Security metadata."),
                        Table("g_names", "Global name history."),
                    ),
                ),
                Source(
                    4,
                    "ibes",
                    "Analyst estimates, actuals, summaries, and identifier maps.",
                    (
                        Table("det_epsus", "US EPS detail estimates.", "fpedats", SINCE, end),
                        Table("statsumu_epsus", "US EPS unadjusted summary estimates.", "statpers", SINCE, end),
                        Table("actu_epsus", "US EPS unadjusted actuals.", "pends", SINCE, end),
                        Table("id", "IBES identifier history."),
                        Table("det_xepsus", "US ex-item EPS detail estimates.", "fpedats", SINCE, end),
                        Table("statsumu_xepsus", "US ex-item EPS summary estimates.", "statpers", SINCE, end),
                        Table("actu_xepsus", "US ex-item EPS actuals.", "pends", SINCE, end),
                    ),
                ),
                Source(
                    12,
                    "crsp_a_indexes",
                    "CRSP index, S&P 500, and index portfolio series.",
                    (
                        Table("dsix", "Daily index series.", "caldt", SINCE, end),
                        Table("msix", "Monthly index series.", "caldt", SINCE, end),
                        Table("dsp500", "Daily S&P 500 returns.", "caldt", SINCE, end),
                        Table("msp500", "Monthly S&P 500 returns.", "caldt", SINCE, end),
                        Table("dsp500list", "Daily S&P 500 membership/list data."),
                        Table("msp500list", "Monthly S&P 500 membership/list data."),
                        Table("inddlyseriesdata_ind", "Daily index series metadata/data.", "dlycaldt", SINCE, end),
                        Table("indmthseriesdata_ind", "Monthly index series metadata/data.", "mthcaldt", SINCE, end),
                    ),
                ),
            ]
        )

    def plan(self, choices: Iterable[str], *, tables: Iterable[str] | None = None) -> Plan:
        sources = []
        for choice in choices:
            source = self.get(choice)
            source = source.withtables(tables)
            if source.tables:
                sources.append(source)
        return Plan(tuple(sources))

    def get(self, choice: str) -> Source:
        key = choice.strip()
        if key in self._rank:
            return self._rank[key]
        if key in self._name:
            return self._name[key]
        raise ValueError(f"unknown data library selection: {choice}")


class ObjectRegistry:
    def __init__(self, items: Iterable[Named]) -> None:
        self._items = {item.name: item for item in items}

    def get(self, name: str) -> Named:
        try:
            return self._items[name]
        except KeyError as exc:
            raise ValueError(f"unknown registry item: {name}") from exc


class StrategyRegistry(ObjectRegistry):
    pass


class BrokerRegistry(ObjectRegistry):
    pass


DataCatalog = Registry
