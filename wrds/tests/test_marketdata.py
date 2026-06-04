from __future__ import annotations

from pathlib import Path

import pandas as pd

from wrds.client import Client
from backtesting.data import Csv, Plan, Pipeline, Source, SourceRegistry, Table
from wrds.marketdata.catalog import source_registry



class FakeDataDb:
    def __init__(self) -> None:
        self.sql: list[str] = []

    def raw_sql(self, sql: str) -> pd.DataFrame:
        self.sql.append(sql)
        table = sql.split("from ", 1)[1].split()[0].replace(".", "_")
        return pd.DataFrame({"source": [table], "rownum": [1]})


def test_data_catalog_resolves_rank_numbers_to_libraries() -> None:
    catalog = source_registry()

    plan = catalog.plan(["1", "2", "4", "12"])

    assert [library.name for library in plan.libraries] == [
        "crsp",
        "comp",
        "ibes",
        "crsp_a_indexes",
    ]
    assert plan.libraries[0].tables[0].name == "stkdlysecuritydata"


def test_data_catalog_rejects_unknown_selection() -> None:
    catalog = source_registry()

    try:
        catalog.plan(["99"])
    except ValueError as exc:
        assert "unknown data source selection" in str(exc)
    else:
        raise AssertionError("expected invalid selection to fail")


def test_data_pipeline_saves_selected_tables_with_limit(tmp_path: Path) -> None:
    client = Client()
    client.db = FakeDataDb()
    catalog = source_registry()
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
    plan = source_registry().plan(["1", "2"], tables=["stkmthsecuritydata", "company"])

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
    plan = source_registry().plan(["1"], tables=["stkmthsecuritydata"])

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


def test_wrds_uses_backtesting_data_interfaces_instead_of_private_data_package() -> None:
    root = Path(__file__).resolve().parents[1]
    project = root.parent

    assert not (root / "data").exists()
    assert not (root / "data_catalog.py").exists()
    assert not (root / "data_pipeline.py").exists()
    assert (project / "backtesting" / "data" / "download.py").exists()
    assert (project / "backtesting" / "data" / "source.py").exists()


def test_source_registry_is_shared_by_backtesting_data_package() -> None:
    registry = SourceRegistry([Source(1, "demo", "demo", (Table("table"),))])

    assert registry.plan(["demo"]).table_count == 1


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
    plan = source_registry().plan(["1"], tables=["stksecurityinfohist"])

    results = Pipeline(client, writer=writer).save(plan, output=tmp_path / "data", chunksize=1)

    assert writer.paths == [tmp_path / "data" / "crsp" / "stksecurityinfohist.csv"]
    assert results[0].rows == 2


def test_csv_writer_is_shared_by_backtesting_data_package() -> None:
    assert Csv().write
