import json
from pathlib import Path

import pandas as pd

from .models import SavedRun


class RunReader:
    def read(self, run_dir: Path) -> SavedRun:
        run_dir = Path(run_dir)
        series_dir = run_dir / "series"
        positions_dir = run_dir / "positions"
        return SavedRun(
            run_id=run_dir.name,
            path=run_dir,
            config=json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
            summary=json.loads((run_dir / "summary.json").read_text(encoding="utf-8")),
            equity=self._read_series(series_dir / "equity.csv", "equity"),
            returns=self._read_series(series_dir / "returns.csv", "returns"),
            turnover=self._read_series(series_dir / "turnover.csv", "turnover"),
            weights=pd.read_parquet(positions_dir / "weights.parquet"),
            qty=pd.read_parquet(positions_dir / "qty.parquet"),
            monthly_returns=self._read_optional_series(series_dir / "monthly_returns.csv", "monthly_returns"),
            latest_qty=self._read_optional_frame(positions_dir / "latest_qty.csv"),
            latest_weights=self._read_optional_frame(positions_dir / "latest_weights.csv"),
            bucket_ledger=self._read_optional_parquet(positions_dir / "bucket_ledger.parquet"),
            validation=self._read_optional_json(run_dir / "validation.json"),
            split=self._read_optional_json(run_dir / "split.json"),
            factor=self._read_optional_json(run_dir / "factor.json"),
        )

    @staticmethod
    def _read_series(path: Path, column: str) -> pd.Series:
        frame = pd.read_csv(path, parse_dates=["date"])
        return frame.set_index("date")[column]

    @staticmethod
    def _read_optional_series(path: Path, column: str) -> pd.Series | None:
        if not path.exists():
            return None
        frame = pd.read_csv(path, parse_dates=["date"])
        return frame.set_index("date")[column]

    @staticmethod
    def _read_optional_frame(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        return pd.read_csv(path)

    @staticmethod
    def _read_optional_parquet(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        return pd.read_parquet(path)

    @staticmethod
    def _read_optional_json(path: Path) -> dict[str, object] | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
