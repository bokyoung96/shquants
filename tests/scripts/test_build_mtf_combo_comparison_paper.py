from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from scripts.build_mtf_combo_comparison_paper import plot_combo_comparison_paper


def test_combo_comparison_paper_uses_four_performance_panels(tmp_path, monkeypatch) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01"])
    ledgers = {
        "current": pd.DataFrame({"equity": [1.0, 1.04, 1.03], "drawdown": [0.0, 0.0, -0.01], "active_positions": [1, 2, 1]}, index=dates),
        "weekly_sector_rs_plus_daily_vol_compression": pd.DataFrame(
            {"equity": [1.0, 1.02, 1.05], "drawdown": [0.0, -0.005, 0.0], "active_positions": [1, 1, 0]},
            index=dates,
        ),
    }
    trades = {
        "current": pd.DataFrame({"net_return": [-0.01, 0.02, 0.04]}),
        "weekly_sector_rs_plus_daily_vol_compression": pd.DataFrame({"net_return": [-0.005, 0.025, 0.05]}),
    }
    metrics = pd.DataFrame(
        {
            "strategy": ["current", "weekly_sector_rs_plus_daily_vol_compression"],
            "fixed_return": [0.03, 0.05],
            "mdd": [-0.01, -0.005],
        }
    )
    saved: list[Figure] = []

    def capture_savefig(self, *_args, **_kwargs) -> None:
        saved.append(self)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    plot_combo_comparison_paper(ledgers=ledgers, trades=trades, metrics=metrics, path=tmp_path / "paper.png")

    assert len(saved) == 1
    titles = [axis.get_title(loc="left") for axis in saved[0].axes]
    assert titles == ["Performance", "Drawdown", "Position", "Return distribution"]
    labeled_lines = [line for line in saved[0].axes[0].lines if not line.get_label().startswith("_")]
    assert len(labeled_lines) == 2
    plt.close("all")
