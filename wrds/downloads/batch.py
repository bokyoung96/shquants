from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from ..core.io import SavedFile
except ImportError:  # pragma: no cover - direct script compatibility
    from core.io import SavedFile


@dataclass(frozen=True)
class OutputFile:
    name: str
    path: str | Path
    frame: pd.DataFrame


class BatchCsvWriter:
    def write(self, root: str | Path, files: Iterable[OutputFile]) -> list[SavedFile]:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        saved: list[SavedFile] = []
        for item in files:
            path = root / item.path
            path.parent.mkdir(parents=True, exist_ok=True)
            item.frame.to_csv(path, index=False)
            saved.append(SavedFile(item.name, path, len(item.frame)))
        return saved
