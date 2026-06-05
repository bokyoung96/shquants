from dataclasses import dataclass
from pathlib import Path

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data.benchmarks import read_quantwise_benchmark_frame
from backtesting.data.store import ParquetStore

from .bm_weights import read_kospi200_bm_weights
from .io import find_raw_path, read_raw_frame
from .normalize import normalize_frame
from .report import IngestResult


@dataclass(slots=True)
class IngestJob:
    catalog: DataCatalog
    raw_dir: Path
    parquet_dir: Path

    def run(self, dataset_id: DatasetId) -> IngestResult:
        spec = self.catalog.get(dataset_id)
        if dataset_id is DatasetId.QW_BM_WEIGHTS:
            frame = read_kospi200_bm_weights(self.raw_dir)
        else:
            raw_path = find_raw_path(self.raw_dir, spec.stem)
            if dataset_id is DatasetId.QW_BM and raw_path.suffix == ".xlsx":
                frame = read_quantwise_benchmark_frame(raw_path)
            else:
                frame = normalize_frame(read_raw_frame(raw_path))

        store = ParquetStore(self.parquet_dir)
        parquet_path = store.write(spec.stem, frame)

        result = IngestResult.from_frame(spec.stem, frame)
        result.write_json(parquet_path.with_suffix(".json"))
        return result
