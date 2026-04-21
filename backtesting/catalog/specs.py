from dataclasses import dataclass

from .enums import DatasetGroup, DatasetId


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    id: DatasetId
    stem: str
    group: DatasetGroup
    freq: str
    kind: str
    fill: str
    validity: str
    lag: int
    dtype: str
    axis: str = "date_symbol"
