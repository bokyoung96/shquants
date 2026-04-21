from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd


@dataclass(slots=True)
class ValidationSession:
    def run(
        self,
        signal: pd.DataFrame,
        *,
        lag_sensitive_datasets: Iterable[str] = (),
        lag_map: Mapping[str, int] | None = None,
        benchmark: pd.Series | pd.DataFrame | None = None,
        sparse_threshold: float = 1.0,
        stale_gap_datasets: Iterable[str] = (),
    ) -> list[str]:
        warnings: list[str] = []
        lag_map = dict(lag_map or {})

        for dataset in self._unique_sorted(lag_sensitive_datasets):
            if dataset not in lag_map:
                warnings.append(f"missing_lag:{dataset}")

        if signal.index.has_duplicates:
            warnings.append("duplicate_index")

        if benchmark is not None and not self._covers_index(benchmark, signal.index):
            warnings.append("short_benchmark")

        if self._has_sparse_row(signal, sparse_threshold):
            warnings.append("sparse_signal")

        for dataset in self._unique_sorted(stale_gap_datasets):
            warnings.append(f"stale_gap:{dataset}")

        return warnings

    @staticmethod
    def _covers_index(
        benchmark: pd.Series | pd.DataFrame,
        signal_index: pd.Index,
    ) -> bool:
        benchmark_index = pd.Index(benchmark.index)
        return signal_index.difference(benchmark_index).empty

    @staticmethod
    def _has_sparse_row(frame: pd.DataFrame, sparse_threshold: float) -> bool:
        if frame.shape[0] == 0:
            return True

        if frame.shape[1] == 0:
            return True

        coverage = frame.notna().mean(axis=1)
        return bool((coverage < sparse_threshold).any())

    @staticmethod
    def _unique_sorted(values: Iterable[str]) -> list[str]:
        return sorted(set(values))
