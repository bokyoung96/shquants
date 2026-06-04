from __future__ import annotations

from pathlib import Path

import pandas as pd

from wrds.client import Client
from wrds.derivatives.options.registry import OptionRegistry
from wrds.derivatives.options.service import Options



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


def test_options_registry_composes_sources() -> None:
    client = Client()
    client.db = FakeOptionsDb()

    registry = OptionRegistry.default(client)
    options = Options.from_registry(registry)
    links = options.links.at(date="2025-08-29", limit=1)

    assert len(links) == 3
