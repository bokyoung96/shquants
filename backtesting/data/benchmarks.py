from __future__ import annotations

from pathlib import Path

import pandas as pd


BENCHMARK_FIELD_BY_ITEM_CODE = {
    "I100110": "open",
    "I100120": "high",
    "I100130": "low",
    "I100100": "close",
}

BENCHMARK_FIELD_BY_LABEL = {
    "시가지수": "open",
    "고가지수": "high",
    "저가지수": "low",
    "종가지수": "close",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
}


def benchmark_price_series(frame: pd.DataFrame, code: str, *, field: str = "close") -> pd.Series:
    if isinstance(frame.columns, pd.MultiIndex):
        return _multiindex_benchmark_price_series(frame, code, field=field)
    if code in frame.columns:
        return frame[code].astype(float)
    return frame.iloc[:, 0].astype(float)


def benchmark_ohlc_frame(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        return _multiindex_benchmark_ohlc_frame(frame, code)
    close = benchmark_price_series(frame, code, field="close")
    return pd.DataFrame({"close": close.astype(float)}, index=close.index).rename_axis("date")


def read_quantwise_benchmark_frame(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    leading = raw.iloc[:, 0].astype(str).str.strip().str.upper()

    code_rows = leading[leading.eq("CODE")]
    date_rows = leading[leading.eq("D A T E")]
    if code_rows.empty or date_rows.empty:
        raise KeyError(f"unable to locate QuantWise benchmark headers in {path.name}")

    code_row = int(code_rows.index[0])
    date_row = int(date_rows.index[0])
    item_code_row = _first_header_row(leading, "ITEM CODE")
    codes = raw.iloc[code_row, 1:]
    valid_columns = [int(column) for column, value in codes.items() if pd.notna(value)]

    parsed_columns = [
        (
            str(codes[column]).strip(),
            _benchmark_field(raw=raw, column=column, date_row=date_row, item_code_row=item_code_row),
        )
        for column in valid_columns
    ]

    dates = raw.loc[date_row + 1 :, 0].copy()
    frame = raw.loc[date_row + 1 :, valid_columns].copy()
    if _requires_field_axis(parsed_columns):
        frame.columns = pd.MultiIndex.from_tuples(parsed_columns, names=["code", "field"])
    else:
        frame.columns = [code for code, _field in parsed_columns]

    valid_dates = dates.notna()
    frame = frame.loc[valid_dates].copy()
    frame.index = pd.to_datetime(dates.loc[valid_dates]).dt.normalize()
    frame = frame.sort_index()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame.index.name = "date"
    return frame


def _multiindex_benchmark_price_series(frame: pd.DataFrame, code: str, *, field: str) -> pd.Series:
    code_level = _column_level(frame.columns, "code", default=0)
    field_level = _column_level(frame.columns, "field", default=1)
    codes = frame.columns.get_level_values(code_level)
    fields = frame.columns.get_level_values(field_level)

    exact = (codes == code) & (fields == field)
    if exact.any():
        return frame.loc[:, exact].iloc[:, 0].astype(float)

    code_match = codes == code
    if code_match.any():
        by_code = frame.loc[:, code_match]
        if by_code.shape[1] == 1:
            return by_code.iloc[:, 0].astype(float)

    field_match = fields == field
    if field_match.any():
        return frame.loc[:, field_match].iloc[:, 0].astype(float)
    return frame.iloc[:, 0].astype(float)


def _multiindex_benchmark_ohlc_frame(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    code_level = _column_level(frame.columns, "code", default=0)
    field_level = _column_level(frame.columns, "field", default=1)
    codes = frame.columns.get_level_values(code_level)
    code_match = codes == code
    if not code_match.any():
        code_match = pd.Series([True] * len(frame.columns)).to_numpy()

    selected = frame.loc[:, code_match].copy()
    fields = selected.columns.get_level_values(field_level)
    selected.columns = pd.Index([str(field) for field in fields])
    selected = selected.loc[:, ~selected.columns.duplicated(keep="first")]
    ordered = [field for field in ("open", "high", "low", "close") if field in selected.columns]
    return selected.loc[:, ordered].astype(float).rename_axis("date")


def _column_level(columns: pd.MultiIndex, name: str, *, default: int) -> int:
    try:
        return columns.names.index(name)
    except ValueError:
        return default


def _first_header_row(leading: pd.Series, label: str) -> int | None:
    rows = leading[leading.eq(label)]
    if rows.empty:
        return None
    return int(rows.index[0])


def _benchmark_field(
    *,
    raw: pd.DataFrame,
    column: int,
    date_row: int,
    item_code_row: int | None,
) -> str:
    if item_code_row is not None:
        item_code = raw.iat[item_code_row, column]
        field = BENCHMARK_FIELD_BY_ITEM_CODE.get(str(item_code).strip().upper())
        if field is not None:
            return field

    label = raw.iat[date_row, column]
    return BENCHMARK_FIELD_BY_LABEL.get(str(label).strip().lower(), "close")


def _requires_field_axis(columns: list[tuple[str, str]]) -> bool:
    codes = [code for code, _field in columns]
    fields = {field for _code, field in columns}
    return len(codes) != len(set(codes)) or fields != {"close"}
