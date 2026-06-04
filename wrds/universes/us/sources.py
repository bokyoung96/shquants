from __future__ import annotations

from typing import Protocol

import pandas as pd

try:
    from ...mapping import Cleaner, Columns
except ImportError:  # pragma: no cover - direct script compatibility
    from mapping import Cleaner, Columns


NAME_TABLE = "crsp.stocknames_v2"
FACTSET_TABLE = "wrdsapps.fscrsplink"
EXCHANGE = {"N": "NYSE", "A": "AMEX", "Q": "NASDAQ"}
DATE_COLUMNS = (
    "namedt",
    "nameendt",
    "securitybegdt",
    "securityenddt",
    "link_bdate",
    "link_edate",
    "start_date",
    "end_date",
)
INT_COLUMNS = ("permno", "permco", "siccd")


def _limit(sql: str, value: int | None) -> str:
    if value is None:
        return sql
    return f"{sql} limit {int(value)}"


def _date_value(frame: pd.DataFrame, column: str) -> pd.Timestamp | None:
    if frame.empty or column not in frame or pd.isna(frame.iloc[0][column]):
        return None
    return pd.Timestamp(frame.iloc[0][column])


class StockSource(Protocol):
    def latest_date(self) -> pd.Timestamp:
        ...

    def current(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        ...

    def history(self, *, limit: int | None = None) -> pd.DataFrame:
        ...


class LinkSource(Protocol):
    def latest_date(self) -> pd.Timestamp:
        ...

    def current(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        ...

    def history(self, *, limit: int | None = None) -> pd.DataFrame:
        ...


class StockNames:
    name = "stocks"

    def __init__(self, client) -> None:
        self.client = client

    def latest_date(self) -> pd.Timestamp:
        frame = self.client.query(
            "select max(nameenddt) as date "
            f"from {NAME_TABLE} "
            "where primaryexch in ('N','A','Q') "
            "and sharetype = 'NS' "
            "and securitytype = 'EQTY' "
            "and securitysubtype = 'COM' "
            "and usincflg = 'Y'"
        )
        date = _date_value(frame, "date")
        if date is None:
            raise ValueError("no stock name date is available")
        return date

    def current(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        sql = (
            "select "
            "permno, permco, ticker as crsp_ticker, ticker as trade_ticker, "
            "issuernm as company, hdrcusip, hdrcusip9, cusip, cusip9, "
            "shareclass, sharetype, securitytype, securitysubtype, usincflg, issuertype, siccd, "
            "primaryexch, "
            "case primaryexch when 'N' then 'NYSE' when 'A' then 'AMEX' when 'Q' then 'NASDAQ' else null end as exchange, "
            "conditionaltype, tradingstatusflg, namedt, nameenddt as nameendt, securitybegdt, securityenddt "
            f"from {NAME_TABLE} "
            f"where namedt <= '{date}' and nameenddt >= '{date}' "
            "and primaryexch in ('N','A','Q') "
            "and sharetype = 'NS' "
            "and securitytype = 'EQTY' "
            "and securitysubtype = 'COM' "
            "and usincflg = 'Y' "
            "and tradingstatusflg = 'A' "
            "and conditionaltype = 'RW' "
            "order by permno, namedt"
        )
        return clean(self.client.query(_limit(sql, limit)))

    def history(self, *, limit: int | None = None) -> pd.DataFrame:
        sql = (
            "select "
            "permno, permco, ticker as crsp_ticker, ticker as trade_ticker, "
            "issuernm as company, hdrcusip, hdrcusip9, cusip, cusip9, "
            "shareclass, sharetype, securitytype, securitysubtype, usincflg, issuertype, siccd, "
            "primaryexch, "
            "case primaryexch when 'N' then 'NYSE' when 'A' then 'AMEX' when 'Q' then 'NASDAQ' else null end as exchange, "
            "conditionaltype, tradingstatusflg, namedt, nameenddt as nameendt, securitybegdt, securityenddt "
            f"from {NAME_TABLE} "
            "where primaryexch in ('N','A','Q') "
            "and sharetype = 'NS' "
            "and securitytype = 'EQTY' "
            "and securitysubtype = 'COM' "
            "and usincflg = 'Y' "
            "order by permno, namedt"
        )
        return clean(self.client.query(_limit(sql, limit)))


class FactSetLinks:
    name = "factsets"

    def __init__(self, client) -> None:
        self.client = client

    def latest_date(self) -> pd.Timestamp:
        frame = self.client.query(
            "select max(link_edate) as date "
            f"from {FACTSET_TABLE} "
            "where link_edate is not null"
        )
        date = _date_value(frame, "date")
        if date is None:
            raise ValueError("no FactSet link date is available")
        return date

    def current(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
        sql = (
            "select "
            "permno, permco, ticker as factset_ticker, ticker_exchange, "
            "fsym_regional_id, fsym_security_id, factset_entity_id, link_bdate, link_edate "
            f"from {FACTSET_TABLE} "
            f"where link_bdate <= '{date}' "
            f"and (link_edate is null or link_edate >= '{date}') "
            "and fsym_id_kind = 'R' "
            "order by permno, link_bdate"
        )
        return clean(self.client.query(_limit(sql, limit)))

    def history(self, *, limit: int | None = None) -> pd.DataFrame:
        sql = (
            "select "
            "permno, permco, ticker as factset_ticker, ticker_exchange, "
            "fsym_regional_id, fsym_security_id, factset_entity_id, link_bdate, link_edate "
            f"from {FACTSET_TABLE} "
            "where fsym_id_kind = 'R' "
            "order by permno, link_bdate"
        )
        return clean(self.client.query(_limit(sql, limit)))


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return Cleaner(Columns(dates=DATE_COLUMNS, ints=INT_COLUMNS)).frame(frame)
