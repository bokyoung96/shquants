from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import Client
from options import Options
from us import US


class FakeDb:
    def __init__(self) -> None:
        self.sql: list[str] = []
        self.closed = False

    def raw_sql(self, sql: str) -> pd.DataFrame:
        self.sql.append(sql)
        if "max(link_edate)" in sql:
            return pd.DataFrame({"date": ["2025-12-31"]})
        return pd.DataFrame(
            {
                "permno": [10000, 10000],
                "permco": [7952, 7952],
                "ticker": ["OMFGA", "OMFGA"],
                "issuernm": ["OPTIMUM MANUFACTURING INC", "OPTIMUM MANUFACTURING INC"],
                "fsym_regional_id": ["T0GD7S-R", "T0GD7S-R"],
                "fsym_security_id": ["S143T7-S", "S143T7-S"],
                "factset_entity_id": [pd.NA, pd.NA],
                "link_bdate": ["1986-01-07", "1986-01-08"],
                "link_edate": ["1987-06-11", "1987-06-11"],
            }
        )

    def close(self) -> None:
        self.closed = True


def test_client_loads_login_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"id": "user", "pwd": "secret"}')

    client = Client(config)
    client.login()

    assert client.user == "user"
    assert client.password == "secret"


def test_client_downloads_query_to_csv(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeDb()

    path = client.download("select * from demo", tmp_path / "out.csv")

    assert path.exists()
    assert "select * from demo" in client.db.sql[-1]


def test_client_builds_table_query_with_limit(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeDb()

    client.table("wrdsapps.fscrsplink", tmp_path / "table.csv", limit=3)

    assert client.db.sql[-1] == "select * from wrdsapps.fscrsplink limit 3"


def test_client_builds_universe_from_latest_links() -> None:
    client = Client()
    client.db = FakeDb()

    links = client.links()
    universe = client.universe(links)

    assert "link_edate >= '2025-12-31'" in client.db.sql[-1]
    assert list(universe["permno"]) == [10000]
    assert universe.loc[0, "link_bdate"] == pd.Timestamp("1986-01-08")


class FakeUsDb:
    def __init__(self) -> None:
        self.sql: list[str] = []

    def raw_sql(self, sql: str) -> pd.DataFrame:
        self.sql.append(sql)
        if "max(nameenddt)" in sql:
            return pd.DataFrame({"date": ["2025-12-31"]})
        if "max(link_edate)" in sql:
            return pd.DataFrame({"date": ["2025-12-31"]})
        if "from crsp.stocknames_v2" in sql:
            return pd.DataFrame(
                {
                    "permno": [10001, 10002, 10003],
                    "permco": [7953, 7954, 7955],
                    "crsp_ticker": ["AAA", "BBB", "DEL"],
                    "trade_ticker": ["AAA", "BBB", "DEL"],
                    "company": ["AAA INC", "BBB INC", "DEL INC"],
                    "hdrcusip": ["11111111", "22222222", "33333333"],
                    "hdrcusip9": ["111111111", "222222222", "333333333"],
                    "cusip": ["11111111", "22222222", "33333333"],
                    "cusip9": ["111111111", "222222222", "333333333"],
                    "shareclass": [pd.NA, pd.NA, pd.NA],
                    "sharetype": ["NS", "NS", "NS"],
                    "securitytype": ["EQTY", "EQTY", "EQTY"],
                    "securitysubtype": ["COM", "COM", "COM"],
                    "usincflg": ["Y", "Y", "Y"],
                    "issuertype": ["CORP", "CORP", "CORP"],
                    "siccd": [1000, 2000, 3000],
                    "primaryexch": ["N", "Q", "A"],
                    "exchange": ["NYSE", "NASDAQ", "AMEX"],
                    "conditionaltype": ["RW", "RW", "RW"],
                    "tradingstatusflg": ["A", "A", "A"],
                    "namedt": ["2020-01-01", "2020-01-01", "2024-03-01"],
                    "nameendt": ["2025-12-31", "2025-12-31", "2024-03-13"],
                    "securitybegdt": ["2020-01-01", "2020-01-01", "2024-03-01"],
                    "securityenddt": ["2025-12-31", "2025-12-31", "2024-03-13"],
                }
            )
        if "from wrdsapps.fscrsplink" in sql:
            return pd.DataFrame(
                {
                    "permno": [10001, 10003],
                    "permco": [7953, 7955],
                    "factset_ticker": ["AAA", "DEL"],
                    "ticker_exchange": ["AAA-NYS", "DEL-ASE"],
                    "fsym_regional_id": ["AAA-R", "DEL-R"],
                    "fsym_security_id": ["AAA-S", "DEL-S"],
                    "factset_entity_id": ["AAA-E", "DEL-E"],
                    "link_bdate": ["2020-01-01", "2024-03-01"],
                    "link_edate": ["2025-12-31", "2024-03-13"],
                }
            )
        raise AssertionError(sql)


def test_us_latest_uses_common_coverage_date() -> None:
    client = Client()
    client.db = FakeUsDb()

    assert US(client).latest() == "2025-12-31"


def test_us_builds_exchange_and_vendor_mapping() -> None:
    client = Client()
    client.db = FakeUsDb()

    us = US(client)
    names = us.names(date="2025-12-31")
    links = us.factset(date="2025-12-31")
    universe = us.build(names, links)

    assert "primaryexch in ('N','A','Q')" in client.db.sql[-2]
    assert "tradingstatusflg = 'A'" in client.db.sql[-2]
    assert "fsym_id_kind = 'R'" in client.db.sql[-1]
    assert list(universe["market"]) == ["NYSE", "NASDAQ", "AMEX"]
    assert universe.loc[0, "factset_ticker"] == "AAA"
    assert pd.isna(universe.loc[1, "factset_ticker"])


def test_us_history_preserves_effective_mapping_dates() -> None:
    client = Client()
    client.db = FakeUsDb()
    us = US(client)

    history = us.history()

    delisted = history[history["permno"].eq(10003)].iloc[0]
    assert delisted["start_date"] == pd.Timestamp("2024-03-01")
    assert delisted["end_date"] == pd.Timestamp("2024-03-13")
    assert delisted["factset_ticker"] == "DEL"


def test_us_at_filters_daily_membership_from_history() -> None:
    client = Client()
    client.db = FakeUsDb()
    us = US(client)
    history = us.history()

    before_delist = us.at("2024-03-13", history=history)
    after_delist = us.at("2024-03-14", history=history)

    assert 10003 in set(before_delist["permno"])
    assert 10003 not in set(after_delist["permno"])


def test_us_latest_keeps_one_representative_row_per_permno() -> None:
    client = Client()
    client.db = FakeUsDb()
    us = US(client)

    latest = us.latest_rows(us.history())

    assert list(latest["permno"]) == [10001, 10002, 10003]


class FakeOptionsDb:
    def __init__(self) -> None:
        self.sql: list[str] = []

    def raw_sql(self, sql: str) -> pd.DataFrame:
        self.sql.append(sql)
        if "from wrdsapps.opcrsphist" in sql:
            return pd.DataFrame(
                {
                    "secid": [501.0, 502.0, 503.0],
                    "sdate": ["2025-01-01", "2025-01-01", "2025-01-01"],
                    "edate": ["2025-12-31", "2025-12-31", "2025-12-31"],
                    "permno": [10001, 10001, 10002],
                    "score": [2.0, 1.0, 1.0],
                }
            )
        if "from optionm.securd" in sql:
            return pd.DataFrame(
                {
                    "secid": [501.0],
                    "cusip": ["11111111"],
                    "ticker": ["AAA"],
                    "sic": [1000],
                    "index_flag": [0],
                    "exchange_d": [1],
                    "class": [pd.NA],
                    "issue_type": ["0"],
                    "industry_group": [10],
                }
            )
        if "from optionm.secnmd" in sql:
            return pd.DataFrame(
                {
                    "secid": [501.0],
                    "effect_date": ["2025-01-01"],
                    "cusip": ["11111111"],
                    "ticker": ["AAA"],
                    "class": [pd.NA],
                    "issuer": ["AAA INC"],
                    "issue": ["COM"],
                    "sic": [1000],
                }
            )
        if "from optionm.secprd2025" in sql:
            return pd.DataFrame(
                {
                    "secid": [501.0],
                    "date": ["2025-08-29"],
                    "low": [10.0],
                    "high": [11.0],
                    "close": [10.5],
                    "volume": [100],
                    "return": [0.01],
                    "cfadj": [1.0],
                    "open": [10.2],
                    "cfret": [0.01],
                    "shrout": [1000],
                }
            )
        if "from optionm.opprcd2025" in sql:
            return pd.DataFrame(
                {
                    "secid": [501.0],
                    "date": ["2025-08-29"],
                    "symbol": ["AAA250919C00010000"],
                    "exdate": ["2025-09-19"],
                    "cp_flag": ["C"],
                    "strike_price": [10000],
                    "best_bid": [1.0],
                    "best_offer": [1.1],
                    "volume": [10],
                    "open_interest": [100],
                    "impl_volatility": [0.2],
                    "delta": [0.5],
                    "gamma": [0.1],
                    "vega": [0.2],
                    "theta": [-0.01],
                    "optionid": [999],
                    "forward_price": [10.5],
                    "root": ["AAA"],
                    "suffix": [pd.NA],
                }
            )
        if "from optionm.stdopd2025" in sql:
            return pd.DataFrame(
                {
                    "secid": [501.0],
                    "date": ["2025-08-29"],
                    "days": [30],
                    "forward_price": [10.5],
                    "strike_price": [10000],
                    "premium": [1.05],
                    "impl_volatility": [0.2],
                    "delta": [0.5],
                    "gamma": [0.1],
                    "theta": [-0.01],
                    "vega": [0.2],
                    "cp_flag": ["C"],
                }
            )
        raise AssertionError(sql)


def test_options_raw_downloads_table_named_files(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeOptionsDb()

    Options(client).save_raw(date="2025-08-29", output=tmp_path / "raw", limit=1)

    assert (tmp_path / "raw" / "opcrsphist.csv").exists()
    assert (tmp_path / "raw" / "securd.csv").exists()
    assert (tmp_path / "raw" / "secnmd.csv").exists()
    assert (tmp_path / "raw" / "secprd2025.csv").exists()
    assert (tmp_path / "raw" / "opprcd2025.csv").exists()
    assert (tmp_path / "raw" / "stdopd2025.csv").exists()
    assert "from wrdsapps.opcrsphist" in client.db.sql[0]
    assert len(pd.read_csv(tmp_path / "raw" / "opcrsphist.csv")) == 3
