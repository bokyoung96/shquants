from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import Client
from options import Options
from options import OptionRegistry
from data import (
    BrokerRegistry,
    FlowRegistry,
    Plan,
    Pipeline,
    Registry,
    Source,
    StrategyRegistry,
    Table,
)
from run import command_handlers, parse_args, split_csv
from us import US
from us import USRegistry
from universe import Universe, UniverseRegistry


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


def test_options_registry_composes_sources() -> None:
    client = Client()
    client.db = FakeOptionsDb()

    registry = OptionRegistry.default(client)
    options = Options.from_registry(registry)
    links = options.links.at(date="2025-08-29", limit=1)

    assert len(links) == 3


class FakeDataDb:
    def __init__(self) -> None:
        self.sql: list[str] = []

    def raw_sql(self, sql: str) -> pd.DataFrame:
        self.sql.append(sql)
        table = sql.split("from ", 1)[1].split()[0].replace(".", "_")
        return pd.DataFrame({"source": [table], "rownum": [1]})


def test_data_catalog_resolves_rank_numbers_to_libraries() -> None:
    catalog = Registry.default()

    plan = catalog.plan(["1", "2", "4", "12"])

    assert [library.name for library in plan.libraries] == [
        "crsp",
        "comp",
        "ibes",
        "crsp_a_indexes",
    ]
    assert plan.libraries[0].tables[0].name == "stkdlysecuritydata"


def test_data_catalog_rejects_unknown_selection() -> None:
    catalog = Registry.default()

    try:
        catalog.plan(["99"])
    except ValueError as exc:
        assert "unknown data library selection" in str(exc)
    else:
        raise AssertionError("expected invalid selection to fail")


def test_data_pipeline_saves_selected_tables_with_limit(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeDataDb()
    catalog = Registry.default()
    plan = catalog.plan(["1", "4"], tables=["stkdlysecuritydata", "det_epsus"])

    results = Pipeline(client).save(plan, output=tmp_path / "data", limit=5)

    assert (tmp_path / "data" / "crsp" / "stkdlysecuritydata.csv").exists()
    assert (tmp_path / "data" / "ibes" / "det_epsus.csv").exists()
    assert client.db.sql == [
        "select * from crsp.stkdlysecuritydata limit 5",
        "select * from ibes.det_epsus limit 5",
    ]
    assert [result.table for result in results] == ["crsp.stkdlysecuritydata", "ibes.det_epsus"]


def test_data_pipeline_writes_manifest_for_final_review(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeDataDb()
    plan = Registry.default().plan(["1", "2"], tables=["stkmthsecuritydata", "company"])

    Pipeline(client).save(plan, output=tmp_path / "data_sample", limit=2)

    manifest = pd.read_csv(tmp_path / "data_sample" / "manifest.csv")
    assert list(manifest["rank"]) == [1, 2]
    assert list(manifest["library"]) == ["crsp", "comp"]
    assert list(manifest["table"]) == ["stkmthsecuritydata", "company"]
    assert list(manifest["rows"]) == [1, 1]
    assert list(manifest["status"]) == ["saved", "saved"]
    assert list(manifest["path"]) == [
        "crsp/stkmthsecuritydata.csv",
        "comp/company.csv",
    ]


def test_data_pipeline_skips_existing_files_unless_overwrite(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeDataDb()
    path = tmp_path / "data" / "crsp" / "stkmthsecuritydata.csv"
    path.parent.mkdir(parents=True)
    path.write_text("value\nexisting\n")
    plan = Registry.default().plan(["1"], tables=["stkmthsecuritydata"])

    results = Pipeline(client).save(plan, output=tmp_path / "data", limit=1)

    assert client.db.sql == []
    assert results[0].status == "skipped"
    assert path.read_text() == "value\nexisting\n"


class FakeChunkDataDb:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def raw_sql(self, sql: str, **kwargs) -> list[pd.DataFrame]:
        self.calls.append({"sql": sql, **kwargs})
        return [
            pd.DataFrame({"rownum": [1]}),
            pd.DataFrame({"rownum": [2]}),
        ]


class EmptyChunkDataDb:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def raw_sql(self, sql: str, **kwargs) -> list[pd.DataFrame]:
        self.calls.append({"sql": sql, **kwargs})
        return [pd.DataFrame(columns=["rownum"])]


def test_data_pipeline_streams_chunks_to_csv(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeChunkDataDb()
    plan = Plan(
        (
            Source(
                rank=1,
                name="crsp",
                about="test",
                tables=(Table("stocknames"),),
            ),
        )
    )

    results = Pipeline(client).save(plan, output=tmp_path / "data", chunksize=1)

    assert client.db.calls[0]["return_iter"] is True
    assert client.db.calls[0]["chunksize"] == 1
    assert len(pd.read_csv(tmp_path / "data" / "crsp" / "stocknames.csv")) == 2
    assert results[0].rows == 2


def test_data_pipeline_partitions_large_tables_by_year(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeChunkDataDb()
    plan = Plan(
        (
            Source(
                rank=1,
                name="crsp",
                about="test",
                tables=(Table("dsf", date="date", start=2020, end=2021),),
            ),
        )
    )

    results = Pipeline(client).save(plan, output=tmp_path / "data", chunksize=1)

    sql = [call["sql"] for call in client.db.calls]
    assert sql == [
        "select * from crsp.dsf where date >= '2020-01-01' and date < '2021-01-01'",
        "select * from crsp.dsf where date >= '2021-01-01' and date < '2022-01-01'",
    ]
    assert len(pd.read_csv(tmp_path / "data" / "crsp" / "dsf" / "year=2020.csv")) == 2
    assert len(pd.read_csv(tmp_path / "data" / "crsp" / "dsf" / "year=2021.csv")) == 2
    assert results[0].rows == 4


def test_data_pipeline_redownloads_header_only_partitions(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeChunkDataDb()
    path = tmp_path / "data" / "crsp" / "stkdlysecuritydata" / "year=2025.csv"
    path.parent.mkdir(parents=True)
    path.write_text("rownum\n")
    plan = Plan((Source(1, "crsp", "test", (Table("stkdlysecuritydata", date="dlycaldt", start=2025, end=2025),)),))

    results = Pipeline(client).save(plan, output=tmp_path / "data", chunksize=1)

    assert len(client.db.calls) == 1
    assert len(pd.read_csv(path)) == 2
    assert results[0].status == "saved"


def test_data_pipeline_removes_empty_partitions(tmp_path: Path) -> None:
    client = Client()
    client.db = EmptyChunkDataDb()
    plan = Plan((Source(1, "crsp", "test", (Table("stkdlysecuritydata", date="dlycaldt", start=2026, end=2026),)),))

    results = Pipeline(client).save(plan, output=tmp_path / "data", chunksize=1)

    assert not (tmp_path / "data" / "crsp" / "stkdlysecuritydata" / "year=2026.csv").exists()
    assert results[0].rows == 0
    assert results[0].status == "empty"


class FailingDataDb:
    def raw_sql(self, sql: str, **kwargs) -> list[pd.DataFrame]:
        raise RuntimeError("connection lost")


class RecoveringDataClient(Client):
    def __init__(self) -> None:
        super().__init__()
        self.db = FailingDataDb()
        self.closed_count = 0
        self.connected_count = 0

    def close(self) -> None:
        self.closed_count += 1

    def connect(self) -> None:
        self.connected_count += 1
        self.db = FakeChunkDataDb()


def test_data_pipeline_reconnects_and_retries_partition_failures(tmp_path: Path) -> None:
    client = RecoveringDataClient()
    plan = Plan(
        (
            Source(
                rank=1,
                name="crsp",
                about="test",
                tables=(Table("dsf", date="date", start=2020, end=2020),),
            ),
        )
    )

    results = Pipeline(client).save(plan, output=tmp_path / "data", chunksize=1, retries=1)

    assert client.closed_count == 1
    assert client.connected_count == 1
    assert len(pd.read_csv(tmp_path / "data" / "crsp" / "dsf" / "year=2020.csv")) == 2
    assert results[0].rows == 2


def test_split_csv_ignores_empty_table_filters() -> None:
    assert split_csv("stkmthsecuritydata, stkdlysecuritydata,,") == ["stkmthsecuritydata", "stkdlysecuritydata"]
    assert split_csv(None) is None


def test_data_command_defaults_to_final_data_output(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", "data", "1", "--limit", "1"])

    args = parse_args()

    assert args.command == "data"
    assert args.output == Path("wrds/output/datas")


def test_data_registry_defaults_to_2015_through_current_year() -> None:
    plan = Registry.default().plan(["1"], tables=["stkdlysecuritydata"])
    table = plan.sources[0].tables[0]

    assert table.start == 2015
    assert table.end >= 2026
    assert table.parts("crsp")[0][0] == 2015
    assert "dlycaldt" in table.parts("crsp")[0][1]


def test_data_registry_uses_current_crsp_price_tables_without_legacy_duplicates() -> None:
    tables = {table.name for table in Registry.default().get("crsp").tables}

    assert {"stkdlysecuritydata", "stkmthsecuritydata", "stksecurityinfohist"} <= tables
    assert "dsf" not in tables
    assert "msf" not in tables


def test_data_package_uses_simple_module_file_names() -> None:
    root = Path(__file__).resolve().parents[1]

    assert not (root / "data_catalog.py").exists()
    assert not (root / "data_pipeline.py").exists()
    assert (root / "data" / "pipeline.py").exists()
    assert (root / "data" / "registry.py").exists()


def test_strategy_and_broker_registries_manage_composed_objects() -> None:
    class Strategy:
        name = "value"

    class Broker:
        name = "paper"

    strategies = StrategyRegistry([Strategy()])
    brokers = BrokerRegistry([Broker()])

    assert strategies.get("value").name == "value"
    assert brokers.get("paper").name == "paper"


def test_pipeline_accepts_injected_writer(tmp_path: Path) -> None:
    class Writer:
        def __init__(self) -> None:
            self.paths: list[Path] = []

        def write(self, chunks, path: Path) -> int:
            self.paths.append(path)
            frames = list(chunks)
            pd.concat(frames).to_csv(path, index=False)
            return sum(len(frame) for frame in frames)

    client = Client()
    client.db = FakeChunkDataDb()
    writer = Writer()
    plan = Registry.default().plan(["1"], tables=["stksecurityinfohist"])

    results = Pipeline(client, writer=writer).save(plan, output=tmp_path / "data", chunksize=1)

    assert writer.paths == [tmp_path / "data" / "crsp" / "stksecurityinfohist.csv"]
    assert results[0].rows == 2


def test_flow_registry_dispatches_us_and_universe_workflows() -> None:
    registry = FlowRegistry.default()

    assert registry.get("us").name == "us"
    assert registry.get("universe").name == "universe"


def test_run_command_handlers_include_data_workflows() -> None:
    handlers = command_handlers()

    assert {"check", "query", "table", "data", "us", "universe", "options"} <= set(handlers)
