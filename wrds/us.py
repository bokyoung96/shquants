from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mapping import Cleaner, Columns, Linker


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


@dataclass
class Coverage:
    stock_date: pd.Timestamp
    factset_date: pd.Timestamp

    @property
    def common_date(self) -> str:
        return str(min(self.stock_date, self.factset_date).date())


class StockNames:
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


class UniverseBuilder:
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


class US:
    def __init__(self, client) -> None:
        self.client = client
        self.stocks = StockNames(client) if client is not None else None
        self.factsets = FactSetLinks(client) if client is not None else None
        self.builder = UniverseBuilder()

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

    def save_current(self, *, date: str = "latest", output: str | Path = "wrds/output/us/current", limit: int | None = None) -> None:
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
        names.to_csv(output / "names.csv", index=False)
        factset.to_csv(output / "factset_links.csv", index=False)
        universe.to_csv(output / "universe.csv", index=False)
        steps.update()
        steps.close()
        print(f"date={date}")
        print(f"names={len(names)} {output / 'names.csv'}")
        print(f"factset_links={len(factset)} {output / 'factset_links.csv'}")
        print(f"universe={len(universe)} {output / 'universe.csv'}")

    def save_history(self, *, output: str | Path = "wrds/output/us/history", limit: int | None = None) -> None:
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
        names.to_csv(output / "names.csv", index=False)
        factset.to_csv(output / "factset_links.csv", index=False)
        history.to_csv(output / "history.csv", index=False)
        latest.to_csv(output / "latest.csv", index=False)
        steps.update()
        steps.close()
        print(f"names={len(names)} {output / 'names.csv'}")
        print(f"factset_links={len(factset)} {output / 'factset_links.csv'}")
        print(f"history={len(history)} {output / 'history.csv'}")
        print(f"latest={len(latest)} {output / 'latest.csv'}")

    def save_at(self, *, date: str, output: str | Path = "wrds/output/us/at.csv", history_path: str | Path | None = None) -> None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        history = self.history() if history_path is None else clean(pd.read_csv(history_path))
        frame = self.at(date, history=history)
        frame.to_csv(output, index=False)
        print(f"date={date}")
        print(f"rows={len(frame)} {output}")

    def save(self, *, date: str = "latest", output: str | Path = "wrds/output/us", limit: int | None = None) -> None:
        self.save_current(date=date, output=output, limit=limit)

    @staticmethod
    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        return clean(frame)


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    return Cleaner(Columns(dates=DATE_COLUMNS, ints=INT_COLUMNS)).frame(frame)
