from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd
from tqdm import tqdm

from mapping import Cleaner, Columns


LINK_TABLE = "wrdsapps.opcrsphist"
SECURITY_TABLE = "optionm.securd"
NAME_TABLE = "optionm.secnmd"
PRICE_PREFIX = "optionm.opprcd"
STOCK_PREFIX = "optionm.secprd"
STD_PREFIX = "optionm.stdopd"
SURFACE_PREFIX = "optionm.vsurfd"

DATE_COLUMNS = ("sdate", "edate", "effect_date", "date", "exdate", "last_date")
INT_COLUMNS = ("permno", "secid", "optionid", "volume", "open_interest")


def _limit(sql: str, value: int | None) -> str:
    if value is None:
        return sql
    return f"{sql} limit {int(value)}"


class Links(Protocol):
    def history(self, *, limit: int | None = None) -> pd.DataFrame:
        ...

    def at(self, date: str, *, limit: int | None = None) -> pd.DataFrame:
        ...


class Meta(Protocol):
    def securities(self, *, limit: int | None = None) -> pd.DataFrame:
        ...

    def names(self, *, limit: int | None = None) -> pd.DataFrame:
        ...


class Prices(Protocol):
    def quotes(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        ...

    def stocks(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        ...

    def standard(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        ...

    def surface(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        ...


class Named(Protocol):
    name: str


class OptionRegistry:
    def __init__(self, items: Iterable[Named]) -> None:
        self._items = {item.name: item for item in items}

    @classmethod
    def default(cls, client) -> "OptionRegistry":
        return cls((OptionLinks(client), OptionMeta(client), OptionPrices(client)))

    def get(self, name: str):
        try:
            return self._items[name]
        except KeyError as exc:
            raise ValueError(f"unknown option component: {name}") from exc


class OptionLinks:
    name = "links"

    def __init__(self, client) -> None:
        self.client = client

    def history(self, *, limit: int | None = None) -> pd.DataFrame:
        sql = (
            "select secid, sdate, edate, permno, score "
            f"from {LINK_TABLE} "
            "where permno is not null and secid is not null "
            "order by permno, sdate, score"
        )
        return clean(self.client.query(_limit(sql, limit)))

    def at(self, date: str, *, limit: int | None = None) -> pd.DataFrame:
        sql = (
            "select secid, sdate, edate, permno, score "
            f"from {LINK_TABLE} "
            f"where sdate <= '{date}' and edate >= '{date}' "
            "and permno is not null and secid is not null "
            "order by permno, score, secid"
        )
        return clean(self.client.query(_limit(sql, limit)))


class OptionMeta:
    name = "meta"

    def __init__(self, client) -> None:
        self.client = client

    def securities(self, *, limit: int | None = None) -> pd.DataFrame:
        sql = (
            "select secid, cusip, ticker, sic, index_flag, exchange_d, class, issue_type, industry_group "
            f"from {SECURITY_TABLE} "
            "order by secid"
        )
        return clean(self.client.query(_limit(sql, limit)))

    def names(self, *, limit: int | None = None) -> pd.DataFrame:
        sql = (
            "select secid, effect_date, cusip, ticker, class, issuer, issue, sic "
            f"from {NAME_TABLE} "
            "order by secid, effect_date"
        )
        return clean(self.client.query(_limit(sql, limit)))


class OptionPrices:
    name = "prices"

    def __init__(self, client) -> None:
        self.client = client

    def quotes(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        table = f"{PRICE_PREFIX}{pd.Timestamp(date).year}"
        sql = (
            "select secid, date, symbol, exdate, cp_flag, strike_price, "
            "best_bid, best_offer, volume, open_interest, impl_volatility, "
            "delta, gamma, vega, theta, optionid, forward_price, root, suffix "
            f"from {table} "
            f"where date = '{date}' "
            "order by secid, exdate, cp_flag, strike_price"
        )
        return clean(self.client.query(_limit(sql, limit)))

    def stocks(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        table = f"{STOCK_PREFIX}{pd.Timestamp(date).year}"
        sql = (
            "select secid, date, low, high, close, volume, return, cfadj, open, cfret, shrout "
            f"from {table} "
            f"where date = '{date}' "
            "order by secid"
        )
        return clean(self.client.query(_limit(sql, limit)))

    def standard(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        table = f"{STD_PREFIX}{pd.Timestamp(date).year}"
        sql = (
            "select secid, date, days, forward_price, strike_price, premium, "
            "impl_volatility, delta, gamma, theta, vega, cp_flag "
            f"from {table} "
            f"where date = '{date}' "
            "order by secid, days, cp_flag, strike_price"
        )
        return clean(self.client.query(_limit(sql, limit)))

    def surface(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        table = f"{SURFACE_PREFIX}{pd.Timestamp(date).year}"
        sql = (
            "select secid, date, days, delta, impl_volatility, impl_strike, impl_premium, dispersion, cp_flag "
            f"from {table} "
            f"where date = '{date}' "
            "order by secid, days, cp_flag, delta"
        )
        return clean(self.client.query(_limit(sql, limit)))


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
        opcrsphist.to_csv(output / "opcrsphist.csv", index=False)
        securd.to_csv(output / "securd.csv", index=False)
        secnmd.to_csv(output / "secnmd.csv", index=False)
        secprd.to_csv(output / f"secprd{year}.csv", index=False)
        opprcd.to_csv(output / f"opprcd{year}.csv", index=False)
        stdopd.to_csv(output / f"stdopd{year}.csv", index=False)
        steps.close()
        print(f"date={date}")
        print(f"opcrsphist={len(opcrsphist)} {output / 'opcrsphist.csv'}")
        print(f"securd={len(securd)} {output / 'securd.csv'}")
        print(f"secnmd={len(secnmd)} {output / 'secnmd.csv'}")
        print(f"secprd{year}={len(secprd)} {output / f'secprd{year}.csv'}")
        print(f"opprcd{year}={len(opprcd)} {output / f'opprcd{year}.csv'}")
        print(f"stdopd{year}={len(stdopd)} {output / f'stdopd{year}.csv'}")


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return Cleaner(Columns(dates=DATE_COLUMNS, ints=INT_COLUMNS)).frame(frame)
