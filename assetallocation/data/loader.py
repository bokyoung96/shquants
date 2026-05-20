from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

import pandas as pd


OHLC_FIELDS: tuple[tuple[str, str], ...] = (
    ("PX_OPEN", "open"),
    ("PX_HIGH", "high"),
    ("PX_LOW", "low"),
    ("PX_LAST", "close"),
)

_FIELD_TO_OUTPUT = dict(OHLC_FIELDS)


@dataclass(frozen=True, slots=True)
class OHLCConversionResult:
    source_copy_path: Path
    parquet_paths: list[Path]
    rows: int
    symbols: list[str]


@dataclass(slots=True)
class BloombergOHLCExcelConverter:
    source_path: Path
    output_dir: Path
    sheet_name: str | int = 0

    def __post_init__(self) -> None:
        self.source_path = Path(self.source_path)
        self.output_dir = Path(self.output_dir)

    def convert(self) -> OHLCConversionResult:
        frames = self.read_ohlc_frames()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        source_copy_path = self._copy_source()

        parquet_paths: list[Path] = []
        for _, stem in OHLC_FIELDS:
            path = self.output_dir / f"{stem}.parquet"
            frames[stem].to_parquet(path, engine="pyarrow")
            parquet_paths.append(path)

        first_frame = frames[OHLC_FIELDS[0][1]]
        return OHLCConversionResult(
            source_copy_path=source_copy_path,
            parquet_paths=parquet_paths,
            rows=len(first_frame),
            symbols=list(first_frame.columns),
        )

    def read_ohlc_frames(self) -> dict[str, pd.DataFrame]:
        raw = pd.read_excel(self.source_path, sheet_name=self.sheet_name, header=None)
        self._validate_layout(raw)

        ticker_row = raw.iloc[3]
        field_row = raw.iloc[5]
        data = raw.iloc[6:].copy()
        dates = pd.to_datetime(data.iloc[:, 0], errors="coerce").dt.normalize()
        valid_rows = dates.notna()
        dates = dates.loc[valid_rows]
        data = data.loc[valid_rows]

        if dates.duplicated().any():
            raise ValueError("duplicate date values in Bloomberg OHLC workbook")

        frames: dict[str, dict[str, pd.Series]] = {stem: {} for _, stem in OHLC_FIELDS}
        symbols: list[str] = []
        current_symbol: str | None = None

        for column in range(1, raw.shape[1]):
            ticker = ticker_row.iloc[column]
            if pd.notna(ticker):
                current_symbol = str(ticker).strip()
                if current_symbol in symbols:
                    raise ValueError(f"duplicate symbol in Bloomberg OHLC workbook: {current_symbol}")
                symbols.append(current_symbol)

            field = field_row.iloc[column]
            if current_symbol is None or pd.isna(field):
                continue

            output_stem = _FIELD_TO_OUTPUT.get(str(field).strip())
            if output_stem is None:
                continue

            series = pd.to_numeric(data.iloc[:, column], errors="coerce")
            series.index = dates
            frames[output_stem][current_symbol] = series

        missing = [stem for _, stem in OHLC_FIELDS if not frames[stem]]
        if missing:
            raise ValueError(f"missing OHLC fields in Bloomberg workbook: {', '.join(missing)}")

        return {
            stem: self._build_frame(columns_by_symbol, symbols)
            for _, stem in OHLC_FIELDS
            for columns_by_symbol in (frames[stem],)
        }

    def _copy_source(self) -> Path:
        destination = self.output_dir / self.source_path.name
        if self.source_path.resolve() != destination.resolve():
            copy2(self.source_path, destination)
        return destination

    @staticmethod
    def _validate_layout(raw: pd.DataFrame) -> None:
        if raw.shape[0] < 7 or raw.shape[1] < 5:
            raise ValueError("Bloomberg OHLC workbook is too small to contain ticker and field headers")
        if str(raw.iloc[5, 0]).strip() != "Dates":
            raise ValueError("Bloomberg OHLC workbook must have a Dates column in row 6")

    @staticmethod
    def _build_frame(columns_by_symbol: dict[str, pd.Series], symbols: list[str]) -> pd.DataFrame:
        missing_symbols = [symbol for symbol in symbols if symbol not in columns_by_symbol]
        if missing_symbols:
            raise ValueError(f"missing OHLC values for symbols: {', '.join(missing_symbols)}")

        frame = pd.DataFrame({symbol: columns_by_symbol[symbol] for symbol in symbols})
        frame.index.name = "date"
        return frame.sort_index()


def default_converter() -> BloombergOHLCExcelConverter:
    root = Path(__file__).resolve().parent
    return BloombergOHLCExcelConverter(
        source_path=root / "data_bb.xlsx",
        output_dir=root,
    )


def main() -> None:
    result = default_converter().convert()
    print(f"source={result.source_copy_path}")
    print(f"rows={result.rows}")
    print(f"symbols={len(result.symbols)}")
    for path in result.parquet_paths:
        print(f"parquet={path}")


if __name__ == "__main__":
    main()
