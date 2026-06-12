import os
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List


class DataConverter:
    def __init__(self, excel_path: str, output_dir: str) -> None:
        self.excel_path = excel_path
        self.output_dir = output_dir
        self._output_dir()

    def _output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _data_format(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if df.shape[1] < 1:
            return None
        df.columns = pd.to_datetime(df.columns, errors="coerce")
        return df

    def data_convert(self) -> None:
        xls = pd.ExcelFile(self.excel_path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(
                xls,
                sheet_name=sheet,
                header=7,
                index_col=0
            ).iloc[:, 5:]
            if df.empty:
                continue
            df = self._data_format(df)
            if df is not None:
                out_path = os.path.join(self.output_dir, f"{sheet}.parquet")
                df.to_parquet(out_path, engine="pyarrow")


class DataLoader:
    @dataclass(frozen=True)
    class DataEntry:
        name: str
        path: str

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self._registry: Dict[str, DataLoader.DataEntry] = self._register()

    def _register(self) -> Dict[str, 'DataLoader.DataEntry']:
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
        return {
            os.path.splitext(f)[0]: DataLoader.DataEntry(
                name=os.path.splitext(f)[0],
                path=os.path.join(self.data_dir, f)
            ) for f in files
        }

    def available(self) -> List[str]:
        return list(self._registry.keys())

    def load(self, name: str) -> pd.DataFrame:
        if name not in self._registry:
            raise ValueError(f"Dataset '{name}' not found. Available: {self.available()}")
        return pd.read_parquet(self._registry[name].path, engine="pyarrow")

    def __getattr__(self, name: str) -> pd.DataFrame:
        if name in self._registry:
            return self.load(name)
        raise AttributeError(f"No such dataset: {name}")


if __name__ == "__main__":
    # NOTE: DATA CONVERTER
    converter = DataConverter(
        excel_path=os.path.join(os.path.dirname(__file__), "DATA.xlsx"),
        output_dir=os.path.join(os.path.dirname(__file__), "DATA")
    )
    converter.data_convert()

    # NOTE: DATA LOADER
    loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), "DATA"))
    print(loader.available())
