from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd


class Writer(Protocol):
    def write(self, chunks: Iterable[pd.DataFrame], path: Path) -> int:
        ...


class Csv:
    def write(self, chunks: Iterable[pd.DataFrame], path: Path) -> int:
        tmp = path.with_suffix(f"{path.suffix}.tmp")
        rows = 0
        header = True
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            for chunk in chunks:
                chunk.to_csv(tmp, mode="a", header=header, index=False)
                rows += len(chunk)
                header = False
            if header:
                tmp.write_text("")
            tmp.replace(path)
            return rows
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise
