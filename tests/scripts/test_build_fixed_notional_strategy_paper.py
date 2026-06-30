from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from scripts.build_fixed_notional_strategy_paper import plot_strategy_paper


def test_strategy_paper_uses_minimal_linkedin_panels_without_header(tmp_path, monkeypatch) -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-03-01"])
    ledger = pd.DataFrame(
        {
            "equity": [1.0, 1.04, 1.02, 1.12],
            "drawdown": [0.0, 0.0, -0.0192, 0.0],
            "active_positions": [1, 2, 1, 0],
        },
        index=dates,
    )
    trades = pd.DataFrame(
        {
            "net_return": [-0.012, -0.006, -0.004, 0.009, 0.028, 0.11],
            "exit_reason": ["atr_stop", "atr_stop", "new_high_lost", "new_high_lost", "new_high_lost", "new_high_lost"],
        }
    )
    audit = {
        "selected_trades": 6,
        "fixed_notional_final_return": 0.12,
        "fixed_notional_mdd": -0.0192,
        "selected_avg_trade_return": 0.0208,
        "selected_hit_rate": 0.5,
        "selected_profit_factor": 2.0,
        "max_active_positions": 2,
        "slot_weight": 0.05,
    }
    saved: list[Figure] = []

    def capture_savefig(self, *_args, **_kwargs) -> None:
        saved.append(self)

    monkeypatch.setattr(Figure, "savefig", capture_savefig)

    plot_strategy_paper(
        ledger=ledger,
        trades=trades,
        audit=audit,
        path=tmp_path / "paper.png",
        strategy_name="multi timeframe various momentum indicators strategy",
    )

    assert len(saved) == 1
    titles = [axis.get_title(loc="left") for axis in saved[0].axes]
    assert titles == ["Performance", "Drawdown", "Position", "Return distribution"]
    figure_text = "\n".join(text.get_text() for text in saved[0].texts)
    assert "multi timeframe various momentum indicators strategy" not in figure_text
    assert "FINAL RETURN" not in figure_text
    assert "PROFIT FACTOR" not in figure_text
    plt.close("all")
