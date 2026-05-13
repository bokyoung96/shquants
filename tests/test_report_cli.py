from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from backtesting.engine import BacktestResult
from backtesting.reporting import ReportCli
from backtesting.reporting.writer import RunWriter
from backtesting.run import RunConfig, RunReport


def test_report_cli_parses_run_ids() -> None:
    cli = ReportCli()
    args = cli.parser().parse_args(["--runs", "run-a", "run-b", "--name", "compare-report"])

    assert args.runs == ["run-a", "run-b"]
    assert args.name == "compare-report"


def test_report_cli_builds_report_bundle(tmp_path: Path) -> None:
    runs_root = tmp_path / "backtests"
    reports_root = tmp_path / "reports"
    run_dir = RunWriter(runs_root).write(_build_report())

    payload = ReportCli(runs_root=runs_root, reports_root=reports_root).run(
        ["--runs", run_dir.name, "--name", "sample-report", "--title", "Sample Report"]
    )

    out_dir = reports_root / "sample-report"
    assert payload["run_ids"] == [run_dir.name]
    assert payload["output_dir"] == str(out_dir)
    assert "pdf_ok" not in payload
    assert "pdf_path" not in payload
    assert "pdf_error" not in payload
    assert not (out_dir / "report.pdf").exists()
    assert (out_dir / "report.html").exists()
    assert (out_dir / "report.json").exists()
    saved = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert saved["report_name"] == "sample-report"
    assert saved["run_ids"] == [run_dir.name]
    assert saved["html_path"].endswith("report.html")


def _build_report() -> RunReport:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    result = BacktestResult(
        equity=pd.Series([100.0, 110.0], index=index, name="equity"),
        returns=pd.Series([0.0, 0.1], index=index, name="returns"),
        weights=pd.DataFrame({"A": [-0.25, -0.25], "B": [0.75, 0.75]}, index=index),
        qty=pd.DataFrame({"A": [0.0, 2.0], "B": [0.0, -5.0]}, index=index),
        turnover=pd.Series([0.0, 0.05], index=index, name="turnover"),
    )
    config = RunConfig(start="2024-01-02", end="2024-01-03", strategy="trend_rank", name="cli-roundtrip")
    summary = {"cagr": 0.1, "mdd": -0.2, "sharpe": 1.2, "final_equity": 110.0, "avg_turnover": 0.05}
    return RunReport(config=config, summary=summary, result=result)
