from dataclasses import dataclass
from pathlib import Path

from backtesting.catalog import DataCatalog, DatasetId
from backtesting.data.store import ParquetStore

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
        raw_path = find_raw_path(self.raw_dir, spec.stem)
        frame = normalize_frame(read_raw_frame(raw_path))

        store = ParquetStore(self.parquet_dir)
        parquet_path = store.write(spec.stem, frame)

        result = IngestResult.from_frame(spec.stem, frame)
        result.write_json(parquet_path.with_suffix(".json"))
        return result
