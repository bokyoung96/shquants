from pathlib import Path

import pandas as pd
import pytest

from backtesting.reporting.writer import RunWriter, _EMPTY_PNG


@pytest.fixture(autouse=True)
def _stub_plot_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    def _write_placeholder_plot(path: Path, series: pd.Series, title: str, ylabel: str) -> None:
        path.write_bytes(_EMPTY_PNG)

    monkeypatch.setattr(RunWriter, "_plot_series", staticmethod(_write_placeholder_plot))