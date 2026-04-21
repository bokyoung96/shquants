from __future__ import annotations

from dataclasses import dataclass

from .enums import DatasetGroup, DatasetId


@dataclass(frozen=True, slots=True)
class DatasetGroups:
    price: tuple[DatasetId, ...]
    fundamental: tuple[DatasetId, ...]
    estimate: tuple[DatasetId, ...]
    flow: tuple[DatasetId, ...]
    flag: tuple[DatasetId, ...]
    meta: tuple[DatasetId, ...]
    benchmark: tuple[DatasetId, ...]

    def get(self, group: DatasetGroup) -> tuple[DatasetId, ...]:
        mapping = {
            DatasetGroup.PRICE: self.price,
            DatasetGroup.FUNDAMENTAL: self.fundamental,
            DatasetGroup.ESTIMATE: self.estimate,
            DatasetGroup.FLOW: self.flow,
            DatasetGroup.FLAG: self.flag,
            DatasetGroup.META: self.meta,
            DatasetGroup.BENCHMARK: self.benchmark,
        }
        return mapping[group]
