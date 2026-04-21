import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class IngestResult:
    stem: str
    rows: int
    columns: int
    date_start: date
    date_end: date
    missing: int
    shape: list[int]
    dtypes: dict[str, str]

    @classmethod
    def from_frame(cls, stem: str, frame: pd.DataFrame) -> "IngestResult":
        return cls(
            stem=stem,
            rows=len(frame),
            columns=len(frame.columns),
            date_start=frame.index.min().date(),
            date_end=frame.index.max().date(),
            missing=int(frame.isna().sum().sum()),
            shape=[len(frame), len(frame.columns)],
            dtypes={column: str(dtype) for column, dtype in frame.dtypes.items()},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "stem": self.stem,
            "rows": self.rows,
            "columns": self.columns,
            "missing": self.missing,
            "date_start": self.date_start.isoformat(),
            "date_end": self.date_end.isoformat(),
            "shape": self.shape,
            "dtypes": self.dtypes,
        }

    def write_json(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path
