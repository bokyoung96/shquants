from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

from backtesting.reporting.models import SavedRun
from backtesting.reporting.plots import PlotExportError, PlotLibrary


def _sample_run(tmp_path: Path) -> SavedRun:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    return SavedRun(
        run_id="sample",
        path=tmp_path,
        config={"strategy": "momentum"},
        summary={"cagr": 0.1, "mdd": -0.2, "sharpe": 1.0, "final_equity": 110.0, "avg_turnover": 0.05},
        equity=pd.Series([100.0, 110.0], index=index),
        returns=pd.Series([0.0, 0.1], index=index),
        turnover=pd.Series([0.0, 0.05], index=index),
        weights=pd.DataFrame({"A": [1.0, 1.0]}, index=index),
        qty=pd.DataFrame({"A": [10.0, 10.0]}, index=index),
    )


def _sample_run_named(tmp_path: Path, run_id: str, scale: float = 1.0) -> SavedRun:
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"])
    return SavedRun(
        run_id=run_id,
        path=tmp_path / run_id,
        config={"strategy": run_id},
        summary={
            "cagr": 0.1 * scale,
            "mdd": -0.2,
            "sharpe": 1.0,
            "final_equity": 110.0 * scale,
            "avg_turnover": 0.05,
        },
        equity=pd.Series([100.0 * scale, 110.0 * scale, 108.0 * scale], index=index),
        returns=pd.Series([0.0, 0.1, -0.01818181818181818], index=index),
        turnover=pd.Series([0.0, 0.05, 0.02], index=index),
        weights=pd.DataFrame({"A": [1.0, 0.0, 0.5], "B": [0.0, -0.25, 0.0]}, index=index),
        qty=pd.DataFrame({"A": [10.0, 0.0, 5.0], "B": [0.0, -5.0, 0.0]}, index=index),
        monthly_returns=pd.Series([0.05, -0.02], index=pd.to_datetime(["2024-01-31", "2024-02-29"])),
    )


def _write_image_success(self, path, *args, **kwargs):  # type: ignore[no-untyped-def]
    Path(path).write_bytes(b"png")


def test_plot_library_writes_equity_plot(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "write_image", _write_image_success)

    path = PlotLibrary(tmp_path).equity([_sample_run(tmp_path)])

    assert path.exists()
    assert path.suffix == ".png"


def test_plot_library_writes_all_expected_plots(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "write_image", _write_image_success)

    run = _sample_run_named(tmp_path, "sample")

    plotter = PlotLibrary(tmp_path)

    for path in (
        plotter.drawdown([run]),
        plotter.turnover([run]),
        plotter.top_weights([run]),
        plotter.monthly_heatmap([run]),
    ):
        assert path.exists()
        assert path.suffix == ".png"


def test_plot_library_preserves_flat_plot_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "write_image", _write_image_success)

    run = _sample_run_named(tmp_path, "sample")
    plotter = PlotLibrary(tmp_path)

    assets = {
        "equity": plotter.equity([run]),
        "drawdown": plotter.drawdown([run]),
        "turnover": plotter.turnover([run]),
        "top_weights": plotter.top_weights([run]),
        "monthly_heatmap": plotter.monthly_heatmap([run]),
    }

    assert assets.keys() == {"equity", "drawdown", "turnover", "top_weights", "monthly_heatmap"}


def test_plot_library_supports_multi_run_equity_chart(tmp_path: Path, monkeypatch) -> None:
    captured = {}

    def capture_write_image(self, path, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["trace_count"] = len(self.data)
        Path(path).write_bytes(b"png")

    monkeypatch.setattr(go.Figure, "write_image", capture_write_image)

    runs = [_sample_run_named(tmp_path, "run-a", 1.0), _sample_run_named(tmp_path, "run-b", 2.0)]
    path = PlotLibrary(tmp_path).equity(runs)

    assert path.exists()
    assert captured["trace_count"] == 2


def test_plot_library_supports_many_run_monthly_heatmap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(go.Figure, "write_image", _write_image_success)

    runs = [_sample_run_named(tmp_path, f"run-{i}", float(i + 1)) for i in range(14)]

    path = PlotLibrary(tmp_path).monthly_heatmap(runs)

    assert path.exists()
    assert path.suffix == ".png"


def test_plot_library_writes_html_fallback_when_image_export_fails(tmp_path: Path, monkeypatch) -> None:
    run = _sample_run(tmp_path)

    def fail_write_image(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("no image export")

    monkeypatch.setattr(go.Figure, "write_image", fail_write_image)

    path = PlotLibrary(tmp_path).equity([run])

    assert path.exists()
    assert path.suffix == ".html"
    assert not path.with_suffix(".png").exists()


@pytest.mark.parametrize(
    "method_name",
    ["equity", "drawdown", "turnover", "top_weights", "monthly_heatmap"],
)
def test_plot_library_strict_png_mode_raises_controlled_exception(
    tmp_path: Path, monkeypatch, method_name: str
) -> None:
    run = _sample_run(tmp_path)

    def fail_write_image(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("no image export")

    monkeypatch.setattr(go.Figure, "write_image", fail_write_image)

    with pytest.raises(PlotExportError) as excinfo:
        getattr(PlotLibrary(tmp_path), method_name)([run], require_png=True)

    assert excinfo.value.png_path.suffix == ".png"
    assert excinfo.value.html_path.suffix == ".html"
    assert excinfo.value.html_path.exists()
