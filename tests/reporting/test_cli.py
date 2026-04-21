import json
from pathlib import Path

import pytest

from backtesting.reporting.cli import ReportCli
from backtesting.reporting.models import ReportKind


def test_cli_parser_supports_kind_and_benchmark_options(tmp_path: Path) -> None:
    cli = ReportCli(runs_root=tmp_path, reports_root=tmp_path / "reports")

    args = cli.parser().parse_args(
        [
            "--runs",
            "a",
            "b",
            "--name",
            "compare",
            "--kind",
            "comparison",
            "--benchmark-code",
            "IKS200",
            "--benchmark-name",
            "KOSPI200",
        ]
    )

    assert args.runs == ["a", "b"]
    assert args.kind == "comparison"
    assert args.benchmark_code == "IKS200"
    assert args.benchmark_name == "KOSPI200"


def test_cli_builds_report_spec_with_auto_kind_and_benchmark(tmp_path: Path, monkeypatch) -> None:
    cli = ReportCli(runs_root=tmp_path, reports_root=tmp_path / "reports")

    captured = {}

    monkeypatch.setattr(cli.reader, "read", lambda path: f"run:{Path(path).name}")

    def _build(spec, runs):  # type: ignore[no-untyped-def]
        captured["spec"] = spec
        captured["runs"] = runs
        return type("Bundle", (), {"out_dir": tmp_path / "reports" / spec.name})()

    monkeypatch.setattr(cli.builder, "build", _build)
    monkeypatch.setattr(cli.html, "render", lambda bundle: bundle.out_dir / "report.html")
    monkeypatch.setattr(cli.pdf, "render_with_status", lambda html_path: (html_path.with_suffix(".pdf"), {"pdf_ok": True}))

    payload = cli.run(
        [
            "--runs",
            "a",
            "b",
            "--name",
            "compare",
            "--benchmark-code",
            "IKS200",
            "--benchmark-name",
            "KOSPI200",
        ]
    )

    assert captured["spec"].kind is ReportKind.COMPARISON
    assert captured["spec"].benchmark.code == "IKS200"
    assert captured["spec"].benchmark.name == "KOSPI200"
    assert captured["runs"] == ["run:a", "run:b"]
    assert payload["report_name"] == "compare"
    assert payload["kind"] == "comparison"
    assert payload["benchmark_code"] == "IKS200"
    assert payload["benchmark_name"] == "KOSPI200"
    report_json = json.loads((tmp_path / "reports" / "compare" / "report.json").read_text(encoding="utf-8"))
    assert report_json["kind"] == "comparison"
    assert report_json["benchmark_code"] == "IKS200"
    assert report_json["benchmark_name"] == "KOSPI200"


def test_cli_rejects_invalid_kind_and_run_count_combination(tmp_path: Path) -> None:
    cli = ReportCli(runs_root=tmp_path, reports_root=tmp_path / "reports")

    with pytest.raises(SystemExit):
        cli.parser().parse_args(["--runs", "a", "b", "--name", "bad", "--kind", "tearsheet"])
