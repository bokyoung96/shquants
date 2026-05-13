from pathlib import Path

import pandas as pd


class ParquetStore:
    def __init__(self, root: Path, cache: bool = True) -> None:
        self.root = root
        self.cache = cache
        self._frames: dict[str, pd.DataFrame] = {}

    def write(self, stem: str, frame: pd.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{stem}.parquet"
        frame.to_parquet(path, engine="pyarrow")
        if self.cache:
            self._frames[stem] = frame.copy(deep=True)
        return path

    def read(self, stem: str) -> pd.DataFrame:
        if self.cache and stem in self._frames:
            return self._frames[stem].copy(deep=True)

        frame = pd.read_parquet(self.root / f"{stem}.parquet", engine="pyarrow")
        if self.cache:
            self._frames[stem] = frame.copy(deep=True)
            return frame.copy(deep=True)
        return frame
