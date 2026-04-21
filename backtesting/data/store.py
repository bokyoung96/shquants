from pathlib import Path

import pandas as pd


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, stem: str, frame: pd.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{stem}.parquet"
        frame.to_parquet(path, engine="pyarrow")
        return path

    def read(self, stem: str) -> pd.DataFrame:
        return pd.read_parquet(self.root / f"{stem}.parquet", engine="pyarrow")
