from __future__ import annotations

from pathlib import Path

import pandas as pd

import subprocess
import sys

from backtesting.data import BrokerRegistry, StrategyRegistry
from wrds.derivatives.options.service import Options
from wrds.provider import flow_registry, source_registry
from wrds.run import command_handlers, parse_args, split_csv
from wrds.universes.factset.service import Universe
from wrds.universes.us.service import US



def test_split_csv_ignores_empty_table_filters() -> None:
    assert split_csv("stkmthsecuritydata, stkdlysecuritydata,,") == ["stkmthsecuritydata", "stkdlysecuritydata"]
    assert split_csv(None) is None


def test_data_command_defaults_to_final_data_output(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", "data", "1", "--limit", "1"])

    args = parse_args()

    assert args.command == "data"
    assert args.output == Path("wrds/output/datas")


def test_data_registry_defaults_to_2015_through_current_year() -> None:
    plan = source_registry().plan(["1"], tables=["stkdlysecuritydata"])
    table = plan.sources[0].tables[0]

    assert table.start == 2015
    assert table.end >= 2026
    assert table.parts("crsp")[0][0] == 2015
    assert "dlycaldt" in table.parts("crsp")[0][1]


def test_data_registry_uses_current_crsp_price_tables_without_legacy_duplicates() -> None:
    tables = {table.name for table in source_registry().get("crsp").tables}

    assert {"stkdlysecuritydata", "stkmthsecuritydata", "stksecurityinfohist"} <= tables
    assert "dsf" not in tables
    assert "msf" not in tables


def test_strategy_and_broker_registries_manage_composed_objects() -> None:
    class Strategy:
        name = "value"

    class Broker:
        name = "paper"

    strategies = StrategyRegistry([Strategy()])
    brokers = BrokerRegistry([Broker()])

    assert strategies.get("value").name == "value"
    assert brokers.get("paper").name == "paper"


def test_flow_registry_dispatches_us_and_universe_workflows() -> None:
    registry = flow_registry()

    assert registry.get("us").name == "us"
    assert registry.get("universe").name == "universe"


def test_run_command_handlers_include_data_workflows() -> None:
    handlers = command_handlers()

    assert {"check", "query", "table", "data", "us", "universe", "options"} <= set(handlers)


def test_wrds_package_structure_exposes_data_domains() -> None:
    from wrds.derivatives.options.service import Options as PackagedOptions
    from wrds.downloads.batch import BatchCsvWriter, OutputFile
    from wrds.marketdata.catalog import source_registry as marketdata_source_registry
    from wrds.marketdata.consensus import sources as consensus_sources
    from wrds.marketdata.fundamentals import sources as fundamental_sources
    from wrds.marketdata.indexes import sources as index_sources
    from wrds.marketdata.prices import sources as price_sources
    from wrds.universes.factset.sources import FactSetSource
    from wrds.universes.factset.strategies import LatestLinkStrategy
    from wrds.universes.factset.service import Universe as PackagedUniverse
    from wrds.universes.us.sources import FactSetLinks, StockNames
    from wrds.universes.us.strategies import UniverseBuilder
    from wrds.universes.us.service import US as PackagedUS
    from wrds.derivatives.options.sources import OptionLinks, OptionMeta, OptionPrices

    assert BatchCsvWriter is not None
    assert OutputFile is not None
    assert PackagedOptions is Options
    assert PackagedUniverse is Universe
    assert PackagedUS is US
    assert FactSetSource is not None
    assert LatestLinkStrategy is not None
    assert FactSetLinks is not None
    assert StockNames is not None
    assert UniverseBuilder is not None
    assert OptionLinks is not None
    assert OptionMeta is not None
    assert OptionPrices is not None

    assert [source.name for source in price_sources()] == ["crsp"]
    assert [source.name for source in consensus_sources()] == ["ibes"]
    assert [source.name for source in fundamental_sources()] == ["comp"]
    assert [source.name for source in index_sources()] == ["crsp_a_indexes"]
    assert [source.name for source in marketdata_source_registry().sources] == [
        "crsp",
        "comp",
        "ibes",
        "crsp_a_indexes",
    ]


def test_batch_writer_saves_named_dataframes(tmp_path: Path) -> None:
    from wrds.downloads.batch import BatchCsvWriter, OutputFile

    writer = BatchCsvWriter()
    results = writer.write(
        tmp_path,
        (
            OutputFile("names", "names.csv", pd.DataFrame({"permno": [1]})),
            OutputFile("universe", "universe.csv", pd.DataFrame({"permno": [1]})),
        ),
    )

    assert [result.name for result in results] == ["names", "universe"]
    assert [result.rows for result in results] == [1, 1]
    assert (tmp_path / "names.csv").exists()
    assert (tmp_path / "universe.csv").exists()
    assert pd.read_csv(tmp_path / "names.csv").to_dict("list") == {"permno": [1]}


def test_domain_workflows_use_shared_batch_writer_for_csv_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "provider.py",
        root / "derivatives" / "options" / "service.py",
        root / "universes" / "factset" / "service.py",
        root / "universes" / "us" / "service.py",
    ]

    for path in files:
        assert ".to_csv(" not in path.read_text(encoding="utf-8")


def test_provider_is_thin_workflow_assembler() -> None:
    from wrds.derivatives.options.workflow import OptionsWorkflow
    from wrds.marketdata.workflow import DataWorkflow
    from wrds.universes.factset.workflow import UniverseWorkflow
    from wrds.universes.us.workflow import USWorkflow

    provider_source = (Path(__file__).resolve().parents[1] / "provider.py").read_text(encoding="utf-8")

    assert "class UniverseFlow" not in provider_source
    assert "class USFlow" not in provider_source
    assert "class OptionsFlow" not in provider_source
    assert flow_registry().get("universe").__class__ is UniverseWorkflow
    assert flow_registry().get("us").__class__ is USWorkflow
    assert flow_registry().get("options").__class__ is OptionsWorkflow
    assert flow_registry().get("data").__class__ is DataWorkflow


def test_run_delegates_marketdata_pipeline_to_workflow() -> None:
    run_source = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding="utf-8")

    assert "Pipeline(" not in run_source
    assert "source_registry().plan" not in run_source


def test_wrds_package_imports_work_from_repo_root() -> None:
    project = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-c",
        (
            "import wrds.download; "
            "import wrds.provider; "
            "import wrds.run; "
            "import wrds.derivatives.options.registry; "
            "import wrds.universes.us.service; "
            "import wrds.derivatives.options.service; "
            "import wrds.marketdata.catalog"
        ),
    ]

    result = subprocess.run(command, cwd=project, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr


def test_wrds_script_help_still_works_from_repo_root() -> None:
    project = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "wrds/run.py", "--help"],
        cwd=project,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "WRDS login, query, and download helper." in result.stdout
