from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from ...downloads.batch import BatchCsvWriter, OutputFile
except ImportError:  # pragma: no cover - direct script compatibility
    from downloads.batch import BatchCsvWriter, OutputFile

from .registry import OptionRegistry
from .sources import (
    DATE_COLUMNS,
    INT_COLUMNS,
    LINK_TABLE,
    NAME_TABLE,
    PRICE_PREFIX,
    SECURITY_TABLE,
    STOCK_PREFIX,
    STD_PREFIX,
    SURFACE_PREFIX,
    Links,
    Meta,
    Prices,
    clean,
)


class Options:
    def __init__(
        self,
        client=None,
        *,
        links: Links | None = None,
        meta: Meta | None = None,
        prices: Prices | None = None,
    ) -> None:
        registry = OptionRegistry.default(client) if links is None or meta is None or prices is None else None
        self.links = links or registry.get("links")
        self.meta = meta or registry.get("meta")
        self.prices = prices or registry.get("prices")

    @classmethod
    def from_registry(cls, registry: OptionRegistry) -> "Options":
        return cls(
            links=registry.get("links"),
            meta=registry.get("meta"),
            prices=registry.get("prices"),
        )

    def save_raw(self, *, date: str, output: str | Path, limit: int | None = 1000) -> None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        year = pd.Timestamp(date).year
        steps = tqdm(total=6, desc="option raw", unit="table")
        opcrsphist = self.links.at(date=date, limit=limit)
        steps.update()
        securd = self.meta.securities(limit=limit)
        steps.update()
        secnmd = self.meta.names(limit=limit)
        steps.update()
        secprd = self.prices.stocks(date=date, limit=limit)
        steps.update()
        opprcd = self.prices.quotes(date=date, limit=limit)
        steps.update()
        stdopd = self.prices.standard(date=date, limit=limit)
        steps.update()
        BatchCsvWriter().write(
            output,
            (
                OutputFile("opcrsphist", "opcrsphist.csv", opcrsphist),
                OutputFile("securd", "securd.csv", securd),
                OutputFile("secnmd", "secnmd.csv", secnmd),
                OutputFile(f"secprd{year}", f"secprd{year}.csv", secprd),
                OutputFile(f"opprcd{year}", f"opprcd{year}.csv", opprcd),
                OutputFile(f"stdopd{year}", f"stdopd{year}.csv", stdopd),
            ),
        )
        steps.close()
        print(f"date={date}")
        print(f"opcrsphist={len(opcrsphist)} {output / 'opcrsphist.csv'}")
        print(f"securd={len(securd)} {output / 'securd.csv'}")
        print(f"secnmd={len(secnmd)} {output / 'secnmd.csv'}")
        print(f"secprd{year}={len(secprd)} {output / f'secprd{year}.csv'}")
        print(f"opprcd{year}={len(opprcd)} {output / f'opprcd{year}.csv'}")
        print(f"stdopd{year}={len(stdopd)} {output / f'stdopd{year}.csv'}")


__all__ = (
    "DATE_COLUMNS",
    "INT_COLUMNS",
    "LINK_TABLE",
    "Links",
    "Meta",
    "NAME_TABLE",
    "OptionRegistry",
    "Options",
    "PRICE_PREFIX",
    "Prices",
    "SECURITY_TABLE",
    "STOCK_PREFIX",
    "STD_PREFIX",
    "SURFACE_PREFIX",
    "clean",
)
