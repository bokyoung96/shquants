from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


PRICE_COLUMNS = ["dt", "date", "time", "open", "high", "low", "close", "volume"]
EVENT_COLUMNS = [
    "no",
    "event_dt",
    "date",
    "time",
    "market",
    "kind",
    "action",
    "kospi_price",
    "kospi_change",
    "kospi_return",
    "kospi200_price",
    "kospi200_change",
    "kospi200_return",
    "futures_price",
    "futures_change",
    "futures_return",
]


def build_parquet_bundle(source_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Normalize the sidecar Excel files under source_dir into parquet files."""
    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    event_path, leverage_path, inverse_path = locate_source_files(source)
    events = load_event_history(event_path)
    leverage = load_price_history(leverage_path)
    inverse = load_price_history(inverse_path)

    paths = {
        "events": output / "sidecar_events.parquet",
        "leverage_prices": output / "kodex_leverage_1m.parquet",
        "inverse_prices": output / "kodex_inverse_1m.parquet",
    }
    events.to_parquet(paths["events"], index=False, engine="pyarrow")
    leverage.to_parquet(paths["leverage_prices"], index=False, engine="pyarrow")
    inverse.to_parquet(paths["inverse_prices"], index=False, engine="pyarrow")
    return paths


def locate_source_files(source_dir: str | Path) -> tuple[Path, Path, Path]:
    source = Path(source_dir)
    excels = [p for p in source.glob("*.xlsx") if not p.name.startswith("~$")]
    event_path: Path | None = None
    price_paths: list[Path] = []

    for path in excels:
        sheet = pd.ExcelFile(path).sheet_names[0]
        head = pd.read_excel(path, sheet_name=sheet, nrows=2)
        columns = [str(col) for col in head.columns]
        if columns and columns[0] == "No":
            event_path = path
        elif columns and columns[0].startswith("[일시]"):
            price_paths.append(path)

    if event_path is None:
        raise FileNotFoundError(f"No sidecar event Excel file found in {source}")
    if len(price_paths) < 2:
        raise FileNotFoundError(f"Expected two KODEX price Excel files in {source}, found {len(price_paths)}")

    leverage_path = _match_price_file(price_paths, "레버리지")
    inverse_path = _match_price_file(price_paths, "인버스")
    if leverage_path is None or inverse_path is None:
        priced = [(path, load_price_history(path)["close"].median()) for path in price_paths]
        priced.sort(key=lambda item: item[1], reverse=True)
        leverage_path = priced[0][0]
        inverse_path = priced[-1][0]
    return event_path, leverage_path, inverse_path


def load_price_history(path: str | Path) -> pd.DataFrame:
    """Return normalized 1-minute OHLC prices from a KODEX Excel file."""
    raw = pd.read_excel(path)
    frame = pd.DataFrame(
        {
            "dt": [_combine_date_time(day, minute) for day, minute in zip(raw.iloc[:, 0], raw.iloc[:, 1])],
            "open": pd.to_numeric(raw.iloc[:, 2], errors="coerce"),
            "high": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
            "low": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
            "close": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
        }
    )
    if raw.shape[1] >= 9:
        frame["volume"] = pd.to_numeric(raw.iloc[:, 8], errors="coerce")
    else:
        frame["volume"] = pd.NA
    frame = frame.dropna(subset=["dt", "open", "high", "low", "close"]).copy()
    frame["date"] = frame["dt"].dt.date
    frame["time"] = frame["dt"].dt.time
    frame = frame.drop_duplicates("dt").sort_values("dt").reset_index(drop=True)
    return frame[PRICE_COLUMNS]


def load_event_history(path: str | Path, include_cb: bool = False) -> pd.DataFrame:
    """Return normalized sidecar event history from the exchange event Excel file."""
    raw = pd.read_excel(path)
    rows = raw[raw.iloc[:, 0].notna()].copy()
    if not include_cb:
        rows = rows[rows.iloc[:, 4].astype(str).str.strip() == "사이드카"]

    frame = pd.DataFrame(
        {
            "no": pd.to_numeric(rows.iloc[:, 0], errors="coerce").astype("Int64"),
            "event_dt": [_combine_date_time(day, minute) for day, minute in zip(rows.iloc[:, 1], rows.iloc[:, 2])],
            "market": rows.iloc[:, 3].astype(str).str.strip(),
            "kind": rows.iloc[:, 4].astype(str).str.strip(),
            "action": rows.iloc[:, 5].astype(str).str.strip(),
            "kospi_price": pd.to_numeric(rows.iloc[:, 6], errors="coerce"),
            "kospi_change": pd.to_numeric(rows.iloc[:, 7], errors="coerce"),
            "kospi_return": pd.to_numeric(rows.iloc[:, 8], errors="coerce"),
            "kospi200_price": pd.to_numeric(rows.iloc[:, 9], errors="coerce"),
            "kospi200_change": pd.to_numeric(rows.iloc[:, 10], errors="coerce"),
            "kospi200_return": pd.to_numeric(rows.iloc[:, 11], errors="coerce"),
            "futures_price": pd.to_numeric(rows.iloc[:, 12], errors="coerce"),
            "futures_change": pd.to_numeric(rows.iloc[:, 13], errors="coerce"),
            "futures_return": pd.to_numeric(rows.iloc[:, 14], errors="coerce"),
        }
    )
    frame["date"] = frame["event_dt"].dt.date
    frame["time"] = frame["event_dt"].dt.time
    frame = frame.drop_duplicates(["event_dt", "kind", "action"])
    frame = frame.sort_values("event_dt").reset_index(drop=True)
    return frame[EVENT_COLUMNS]


def run_event_study(
    parquet_dir: str | Path,
    horizons: Iterable[int] = (1, 3, 5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate ETF returns from sidecar activation to release plus each horizon."""
    parquet = Path(parquet_dir)
    events = pd.read_parquet(parquet / "sidecar_events.parquet", engine="pyarrow")
    leverage = pd.read_parquet(parquet / "kodex_leverage_1m.parquet", engine="pyarrow")
    inverse = pd.read_parquet(parquet / "kodex_inverse_1m.parquet", engine="pyarrow")
    trades = calculate_event_returns(events, leverage, inverse, tuple(horizons))
    summary = summarize_event_returns(trades, tuple(horizons))
    return trades, summary


def calculate_event_returns(
    events: pd.DataFrame,
    leverage_prices: pd.DataFrame,
    inverse_prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    events = events.copy()
    events["event_dt"] = pd.to_datetime(events["event_dt"])
    events = events.sort_values("event_dt").reset_index(drop=True)
    leverage = _price_lookup(leverage_prices)
    inverse = _price_lookup(inverse_prices)

    rows: list[dict[str, object]] = []
    used_release_indices: set[int] = set()
    activations = events[events["action"] == "발동"]
    for activation_index, activation in activations.iterrows():
        release = _next_release(events, activation, used_release_indices)
        if release is None:
            rows.append(
                {
                    "activation_dt": activation["event_dt"],
                    "release_dt": pd.NaT,
                    "issue": "missing release",
                }
            )
            continue

        direction = "buy_sidecar" if activation["futures_return"] > 0 else "sell_sidecar"
        prices = leverage if direction == "buy_sidecar" else inverse
        etf = "KODEX leverage" if direction == "buy_sidecar" else "KODEX inverse"
        entry_dt = _minute_floor(activation["event_dt"])
        entry_price = _close_at(prices, entry_dt)
        row = {
            "date": activation["event_dt"].date(),
            "activation_dt": activation["event_dt"],
            "release_dt": release["event_dt"],
            "direction": direction,
            "etf": etf,
            "futures_return_at_trigger_pct": activation["futures_return"] * 100,
            "minutes_to_release": (release["event_dt"] - activation["event_dt"]).total_seconds() / 60,
            "entry_dt": entry_dt,
            "entry_price": entry_price,
            "issue": pd.NA,
        }
        for horizon in horizons:
            exit_dt = _minute_floor(release["event_dt"] + timedelta(minutes=horizon))
            exit_price = _close_at(prices, exit_dt)
            row[f"exit_{horizon}m_dt"] = exit_dt
            row[f"exit_{horizon}m_price"] = exit_price
            row[f"ret_{horizon}m_pct"] = _pct_return(entry_price, exit_price)
        rows.append(row)

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades
    required = ["entry_price", *[f"ret_{horizon}m_pct" for horizon in horizons]]
    return trades.dropna(subset=required).reset_index(drop=True)


def summarize_event_returns(trades: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    groups: list[tuple[str, pd.DataFrame]] = [("all", trades)]
    groups.extend((name, frame) for name, frame in trades.groupby("direction", sort=True))

    rows: list[dict[str, object]] = []
    for group, frame in groups:
        for horizon in horizons:
            returns = frame[f"ret_{horizon}m_pct"].dropna()
            rows.append(
                {
                    "group": group,
                    "horizon": f"{horizon}m",
                    "n": int(returns.count()),
                    "mean_pct": returns.mean(),
                    "median_pct": returns.median(),
                    "win_rate_pct": (returns > 0).mean() * 100 if not returns.empty else pd.NA,
                    "min_pct": returns.min(),
                    "max_pct": returns.max(),
                    "sum_pct": returns.sum(),
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(
    source_dir: str | Path = "sidecar",
    parquet_dir: str | Path = "sidecar/parquet",
    results_dir: str | Path = "sidecar/results",
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    paths = build_parquet_bundle(source_dir, parquet_dir)
    trades, summary = run_event_study(parquet_dir)
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    trades.to_csv(results / "sidecar_event_study_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results / "sidecar_event_study_summary.csv", index=False, encoding="utf-8-sig")
    write_excel_report(trades, results / "sidecar_event_study_report.xlsx")
    return paths, trades, summary


def write_excel_report(trades: pd.DataFrame, output_path: str | Path) -> Path:
    """Write full and yearly event-study detail/summary sheets to Excel."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = trades.copy()
    frame["activation_dt"] = pd.to_datetime(frame["activation_dt"])
    frame["year"] = frame["activation_dt"].dt.year

    detail = _korean_detail_frame(frame)
    summary_all = _korean_summary_frame(summarize_event_returns(frame))
    yearly_summaries = []
    for year, yearly in frame.groupby("year", sort=True):
        summary = _korean_summary_frame(summarize_event_returns(yearly))
        summary.insert(0, "연도", year)
        yearly_summaries.append(summary)
    summary_by_year = pd.concat(yearly_summaries, ignore_index=True) if yearly_summaries else pd.DataFrame()

    with pd.ExcelWriter(output, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm:ss", date_format="yyyy-mm-dd") as writer:
        detail.to_excel(writer, sheet_name="거래내역_전체", index=False)
        summary_all.to_excel(writer, sheet_name="요약_전체", index=False)
        summary_by_year.to_excel(writer, sheet_name="요약_연도별", index=False)
        for year in sorted(frame["year"].dropna().unique()):
            year_frame = frame[frame["year"] == year]
            _korean_detail_frame(year_frame).to_excel(writer, sheet_name=f"거래내역_{int(year)}", index=False)
            _korean_summary_frame(summarize_event_returns(year_frame)).to_excel(
                writer,
                sheet_name=f"요약_{int(year)}",
                index=False,
            )
        _format_workbook(writer.book)
    return output


def _korean_detail_frame(trades: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "date": "날짜",
        "year": "연도",
        "activation_dt": "발동시간",
        "release_dt": "종료시간",
        "direction": "사이드카방향",
        "etf": "매수ETF",
        "futures_return_at_trigger_pct": "발동시_K200선물등락률(%)",
        "minutes_to_release": "발동후_종료까지(분)",
        "entry_dt": "매수기준분봉",
        "entry_price": "매수가(종가)",
        "exit_1m_dt": "종료후1분_매도기준분봉",
        "exit_1m_price": "종료후1분_매도가(종가)",
        "ret_1m_pct": "종료후1분_수익률(%)",
        "exit_3m_dt": "종료후3분_매도기준분봉",
        "exit_3m_price": "종료후3분_매도가(종가)",
        "ret_3m_pct": "종료후3분_수익률(%)",
        "exit_5m_dt": "종료후5분_매도기준분봉",
        "exit_5m_price": "종료후5분_매도가(종가)",
        "ret_5m_pct": "종료후5분_수익률(%)",
    }
    detail = trades[list(columns)].rename(columns=columns).copy()
    detail["사이드카방향"] = detail["사이드카방향"].replace(
        {
            "buy_sidecar": "매수 사이드카",
            "sell_sidecar": "매도 사이드카",
        }
    )
    detail["매수ETF"] = detail["매수ETF"].replace(
        {
            "KODEX leverage": "KODEX 레버리지",
            "KODEX inverse": "KODEX 인버스",
        }
    )
    return detail


def _korean_summary_frame(summary: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "group": "구분",
        "horizon": "보유기간",
        "n": "표본수",
        "mean_pct": "평균수익률(%)",
        "median_pct": "중앙값수익률(%)",
        "win_rate_pct": "승률(%)",
        "min_pct": "최저수익률(%)",
        "max_pct": "최고수익률(%)",
        "sum_pct": "단순합계수익률(%)",
    }
    frame = summary[list(columns)].rename(columns=columns).copy()
    frame["구분"] = frame["구분"].replace(
        {
            "all": "전체",
            "buy_sidecar": "매수 사이드카",
            "sell_sidecar": "매도 사이드카",
        }
    )
    return frame


def _format_workbook(workbook) -> None:
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column in worksheet.columns:
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
            worksheet.column_dimensions[column[0].column_letter].width = min(max(max_length + 2, 10), 28)


def _match_price_file(paths: list[Path], token: str) -> Path | None:
    for path in paths:
        if token in path.name:
            return path
        sheet = pd.ExcelFile(path).sheet_names[0]
        head = pd.read_excel(path, sheet_name=sheet, nrows=0)
        if head.columns.size and token in str(head.columns[0]):
            return path
    return None


def _combine_date_time(day, minute) -> pd.Timestamp:
    date_part = pd.Timestamp(day).date()
    if isinstance(minute, time):
        time_part = minute
    else:
        time_part = pd.to_datetime(str(minute)).time()
    return pd.Timestamp(datetime.combine(date_part, time_part))


def _price_lookup(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["dt"] = pd.to_datetime(frame["dt"])
    return frame.set_index("dt").sort_index()


def _next_release(events: pd.DataFrame, activation: pd.Series, used_release_indices: set[int]) -> pd.Series | None:
    releases = events[
        (events["action"] == "발동해제")
        & (events["event_dt"].dt.date == activation["event_dt"].date())
        & (events["event_dt"] > activation["event_dt"])
    ].sort_values("event_dt")
    for index, release in releases.iterrows():
        if index not in used_release_indices:
            used_release_indices.add(index)
            return release
    return None


def _minute_floor(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).floor("min")


def _close_at(prices: pd.DataFrame, timestamp: pd.Timestamp) -> float | pd.NA:
    if timestamp not in prices.index:
        return pd.NA
    return float(prices.loc[timestamp, "close"])


def _pct_return(entry_price, exit_price) -> float | pd.NA:
    if pd.isna(entry_price) or pd.isna(exit_price):
        return pd.NA
    return (float(exit_price) / float(entry_price) - 1) * 100


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build sidecar parquet data and event-study outputs.")
    parser.add_argument("--source", default="sidecar", help="Directory containing source Excel files.")
    parser.add_argument("--parquet", default="sidecar/parquet", help="Directory for normalized parquet outputs.")
    parser.add_argument("--results", default="sidecar/results", help="Directory for event-study CSV outputs.")
    args = parser.parse_args(argv)

    paths, trades, summary = run_pipeline(args.source, args.parquet, args.results)
    print("parquet:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("\nsummary:")
    print(summary.round(4).to_string(index=False))
    print(f"\ntrades: {len(trades)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
