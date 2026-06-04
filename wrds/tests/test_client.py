from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from wrds.client import Client, load_wrds_library



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


def test_client_loads_external_wrds_library_when_local_package_exists(tmp_path: Path, monkeypatch) -> None:
    site = tmp_path / "site"
    site.mkdir()
    external = site / "wrds.py"
    external.write_text("class Connection: pass\n", encoding="utf-8")
    import wrds as local_wrds

    monkeypatch.setitem(sys.modules, "wrds", local_wrds)
    monkeypatch.syspath_prepend(str(site))

    module = load_wrds_library()

    assert Path(module.__file__).resolve() == external.resolve()
    assert hasattr(module, "Connection")


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
