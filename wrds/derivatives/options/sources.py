from __future__ import annotations

from typing import Protocol

import pandas as pd

try:
    from ...mapping import Cleaner, Columns
except ImportError:  # pragma: no cover - direct script compatibility
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


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return Cleaner(Columns(dates=DATE_COLUMNS, ints=INT_COLUMNS)).frame(frame)
