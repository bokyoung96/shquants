from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from ...downloads.batch import BatchCsvWriter, OutputFile
except ImportError:  # pragma: no cover - direct script compatibility
    from downloads.batch import BatchCsvWriter, OutputFile

from .registry import USRegistry
from .sources import (
    DATE_COLUMNS,
    EXCHANGE,
    FACTSET_TABLE,
    INT_COLUMNS,
    NAME_TABLE,
    LinkSource,
    StockSource,
    clean,
)
from .strategies import Builder


@dataclass
class Coverage:
    stock_date: pd.Timestamp
    factset_date: pd.Timestamp

    @property
    def common_date(self) -> str:
        return str(min(self.stock_date, self.factset_date).date())


class US:
    def __init__(
        self,
        client=None,
        *,
        stocks: StockSource | None = None,
        factsets: LinkSource | None = None,
        builder: Builder | None = None,
    ) -> None:
        registry = USRegistry.default(client) if stocks is None or factsets is None or builder is None else None
        self.stocks = stocks or registry.get("stocks")
        self.factsets = factsets or registry.get("factsets")
        self.builder = builder or registry.get("builder")

    @classmethod
    def from_registry(cls, registry: USRegistry) -> "US":
        return cls(
            stocks=registry.get("stocks"),
            factsets=registry.get("factsets"),
            builder=registry.get("builder"),
        )

    def coverage(self) -> Coverage:
        return Coverage(self.stocks.latest_date(), self.factsets.latest_date())

    def latest(self) -> str:
        return self.coverage().common_date

    def names(self, *, date: str = "latest", limit: int | None = None) -> pd.DataFrame:
        if date == "latest":
            date = self.latest()
        return self.stocks.current(date=date, limit=limit)

    def names_history(self, *, limit: int | None = None) -> pd.DataFrame:
        return self.stocks.history(limit=limit)

    def factset(self, *, date: str = "latest", limit: int | None = None) -> pd.DataFrame:
        if date == "latest":
            date = self.latest()
        return self.factsets.current(date=date, limit=limit)

    def factset_history(self, *, limit: int | None = None) -> pd.DataFrame:
        return self.factsets.history(limit=limit)

    def build(self, names: pd.DataFrame, factset: pd.DataFrame) -> pd.DataFrame:
        return self.builder.current(names, factset)

    def current(self, *, date: str = "latest", limit: int | None = None) -> pd.DataFrame:
        if date == "latest":
            date = self.latest()
        return self.builder.current(self.names(date=date, limit=limit), self.factset(date=date, limit=limit))

    def history(self, *, limit: int | None = None) -> pd.DataFrame:
        return self.builder.history(self.names_history(limit=limit), self.factset_history(limit=limit))

    def history_from(self, names: pd.DataFrame, factset: pd.DataFrame) -> pd.DataFrame:
        return self.builder.history(names, factset)

    def at(self, date: str, *, history: pd.DataFrame | None = None) -> pd.DataFrame:
        if history is None:
            history = self.history()
        return self.builder.at(date, history)

    def latest_rows(self, history: pd.DataFrame) -> pd.DataFrame:
        return self.builder.latest_rows(history)

    def save_current(self, *, date: str = "latest", output: str | Path = "wrds/output/datas/us/current", limit: int | None = None) -> None:
        if date == "latest":
            date = self.latest()
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        steps = tqdm(total=4, desc="current", unit="step")
        names = self.names(date=date, limit=limit)
        steps.update()
        factset = self.factset(date=date, limit=limit)
        steps.update()
        universe = self.build(names, factset)
        steps.update()
        BatchCsvWriter().write(
            output,
            (
                OutputFile("names", "names.csv", names),
                OutputFile("factset_links", "factset_links.csv", factset),
                OutputFile("universe", "universe.csv", universe),
            ),
        )
        steps.update()
        steps.close()
        print(f"date={date}")
        print(f"names={len(names)} {output / 'names.csv'}")
        print(f"factset_links={len(factset)} {output / 'factset_links.csv'}")
        print(f"universe={len(universe)} {output / 'universe.csv'}")

    def save_history(self, *, output: str | Path = "wrds/output/datas/us/history", limit: int | None = None) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        steps = tqdm(total=5, desc="history", unit="step")
        names = self.names_history(limit=limit)
        steps.update()
        factset = self.factset_history(limit=limit)
        steps.update()
        history = self.history_from(names, factset)
        steps.update()
        latest = self.latest_rows(history)
        steps.update()
        BatchCsvWriter().write(
            output,
            (
                OutputFile("names", "names.csv", names),
                OutputFile("factset_links", "factset_links.csv", factset),
                OutputFile("history", "history.csv", history),
                OutputFile("latest", "latest.csv", latest),
            ),
        )
        steps.update()
        steps.close()
        print(f"names={len(names)} {output / 'names.csv'}")
        print(f"factset_links={len(factset)} {output / 'factset_links.csv'}")
        print(f"history={len(history)} {output / 'history.csv'}")
        print(f"latest={len(latest)} {output / 'latest.csv'}")

    def save_at(self, *, date: str, output: str | Path = "wrds/output/datas/us/at.csv", history_path: str | Path | None = None) -> None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        history = self.history() if history_path is None else clean(pd.read_csv(history_path))
        frame = self.at(date, history=history)
        BatchCsvWriter().write(output.parent, (OutputFile("rows", output.name, frame),))
        print(f"date={date}")
        print(f"rows={len(frame)} {output}")

    def save(self, *, date: str = "latest", output: str | Path = "wrds/output/datas/us", limit: int | None = None) -> None:
        self.save_current(date=date, output=output, limit=limit)

    @staticmethod
    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        return clean(frame)


__all__ = (
    "Builder",
    "Coverage",
    "DATE_COLUMNS",
    "EXCHANGE",
    "FACTSET_TABLE",
    "INT_COLUMNS",
    "LinkSource",
    "NAME_TABLE",
    "StockSource",
    "US",
    "USRegistry",
    "clean",
)
