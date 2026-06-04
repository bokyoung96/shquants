from __future__ import annotations

import pandas as pd

from wrds.client import Client
from wrds.universes.factset.registry import UniverseRegistry
from wrds.universes.factset.service import Universe
from wrds.universes.us.registry import USRegistry
from wrds.universes.us.service import US



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


def test_us_registry_composes_sources_and_builder() -> None:
    client = Client()
    client.db = FakeUsDb()

    registry = USRegistry.default(client)
    us = US.from_registry(registry)

    assert registry.get("stocks").latest_date() == pd.Timestamp("2025-12-31")
    assert us.latest() == "2025-12-31"


def test_universe_uses_injected_source_and_strategy() -> None:
    class Source:
        def latest(self) -> str:
            return "2025-12-31"

        def links(self, *, date: str, limit: int | None = None) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "permno": [10001, 10001],
                    "permco": [1, 1],
                    "ticker": ["OLD", "NEW"],
                    "issuernm": ["OLD INC", "NEW INC"],
                    "fsym_regional_id": ["OLD-R", "NEW-R"],
                    "fsym_security_id": ["OLD-S", "NEW-S"],
                    "factset_entity_id": ["OLD-E", "NEW-E"],
                    "link_bdate": ["2020-01-01", "2021-01-01"],
                    "link_edate": ["2025-12-31", "2025-12-31"],
                }
            )

    universe = Universe(source=Source())
    rows = universe.build(universe.links())

    assert list(rows["ticker"]) == ["NEW"]


def test_universe_registry_builds_default_source() -> None:
    client = Client()
    client.db = FakeDb()

    registry = UniverseRegistry.default(client)

    assert registry.get("links").latest() == "2025-12-31"
