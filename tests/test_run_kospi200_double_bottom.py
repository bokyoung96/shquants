import numpy as np
import pandas as pd

from scripts.run_kospi200_double_bottom import main


def test_cli_writes_reproducible_output_contract(tmp_path, monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    low = np.linspace(110.0, 120.0, len(dates))
    high = low + 5.0
    close = low + 2.0
    low[5], high[5], close[5] = 100.0, 102.0, 101.0
    high[10:20] = 135.0
    low[20], high[20], close[20] = 101.5, 104.0, 102.0
    close[25] = 136.0
    open_ = close + 1.0
    rows = pd.DataFrame(
        {
            "ts": dates.tz_localize("UTC"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )
    source = tmp_path / "source.parquet"
    rows.to_parquet(source)
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv", ["run", "--input", str(source), "--output-dir", str(output)]
    )

    main()

    for name in (
        "config.json",
        "signals.csv",
        "event_returns.csv",
        "event_summary.csv",
        "sensitivity_summary.csv",
        "portfolio_summary.csv",
        "portfolio_equity.csv",
        "report.md",
    ):
        assert (output / name).exists()
    assert len(pd.read_csv(output / "signals.csv")) == 1
    assert "KOSPI200" in (output / "report.md").read_text(encoding="utf-8")
