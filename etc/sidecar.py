from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
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
DEFAULT_ENTRY_DELAY_MINUTES = 3
DEFAULT_EXIT_DELAY_MINUTES = 3
DEFAULT_SOURCE_DIR = Path("etc/data/sidecar")
DEFAULT_PARQUET_DIR = Path("etc/data/sidecar/parquet")
DEFAULT_RESULTS_DIR = Path("etc/results/sidecar")


@dataclass(frozen=True, slots=True)
class SidecarRunConfig:
    entry_delay_minutes: int = DEFAULT_ENTRY_DELAY_MINUTES
    exit_delay_minutes: int = DEFAULT_EXIT_DELAY_MINUTES


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
    *,
    entry_delay_minutes: int = DEFAULT_ENTRY_DELAY_MINUTES,
    exit_delay_minutes: int = DEFAULT_EXIT_DELAY_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate ETF returns from activation plus entry delay to release plus exit delay."""
    parquet = Path(parquet_dir)
    events = pd.read_parquet(parquet / "sidecar_events.parquet", engine="pyarrow")
    leverage = pd.read_parquet(parquet / "kodex_leverage_1m.parquet", engine="pyarrow")
    inverse = pd.read_parquet(parquet / "kodex_inverse_1m.parquet", engine="pyarrow")
    config = SidecarRunConfig(entry_delay_minutes=entry_delay_minutes, exit_delay_minutes=exit_delay_minutes)
    trades = calculate_event_returns(events, leverage, inverse, config=config)
    summary = summarize_event_returns(trades, config=config)
    return trades, summary


def calculate_event_returns(
    events: pd.DataFrame,
    leverage_prices: pd.DataFrame,
    inverse_prices: pd.DataFrame,
    *,
    config: SidecarRunConfig | None = None,
) -> pd.DataFrame:
    config = config or SidecarRunConfig()
    events = events.copy()
    events["event_dt"] = pd.to_datetime(events["event_dt"])
    events = events.sort_values("event_dt").reset_index(drop=True)
    leverage = _price_lookup(leverage_prices)
    inverse = _price_lookup(inverse_prices)

    rows: list[dict[str, object]] = []
    used_release_indices: set[int] = set()
    activations = events[events["action"] == "발동"]
    for _, activation in activations.iterrows():
        release = _next_release(events, activation, used_release_indices)
        if release is None:
            rows.append({"activation_dt": activation["event_dt"], "release_dt": pd.NaT, "issue": "missing release"})
            continue

        direction = "buy_sidecar" if activation["futures_return"] > 0 else "sell_sidecar"
        prices = leverage if direction == "buy_sidecar" else inverse
        etf = "KODEX leverage" if direction == "buy_sidecar" else "KODEX inverse"
        entry_dt = _minute_floor(activation["event_dt"] + timedelta(minutes=config.entry_delay_minutes))
        exit_dt = _minute_floor(release["event_dt"] + timedelta(minutes=config.exit_delay_minutes))
        entry_price = _close_at(prices, entry_dt)
        exit_price = _close_at(prices, exit_dt)
        ret_pct = _pct_return(entry_price, exit_price)
        rows.append(
            {
                "date": activation["event_dt"].date(),
                "year": activation["event_dt"].year,
                "activation_dt": activation["event_dt"],
                "release_dt": release["event_dt"],
                "direction": direction,
                "etf": etf,
                "futures_return_at_trigger_pct": activation["futures_return"] * 100,
                "minutes_to_release": (release["event_dt"] - activation["event_dt"]).total_seconds() / 60,
                "entry_delay_m": config.entry_delay_minutes,
                "entry_dt": entry_dt,
                "entry_price": entry_price,
                "exit_delay_m": config.exit_delay_minutes,
                "exit_dt": exit_dt,
                "exit_price": exit_price,
                "ret_pct": ret_pct,
                "issue": pd.NA,
            }
        )

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades
    return trades.dropna(subset=["entry_price", "exit_price", "ret_pct"]).reset_index(drop=True)


def summarize_event_returns(
    trades: pd.DataFrame,
    *,
    config: SidecarRunConfig | None = None,
) -> pd.DataFrame:
    config = config or SidecarRunConfig()
    rows: list[dict[str, object]] = []
    scopes = [("all_years", trades)]
    if "year" in trades.columns:
        scopes.extend((str(year), frame) for year, frame in trades.groupby("year", sort=True))
    for scope, scope_frame in scopes:
        groups: list[tuple[str, pd.DataFrame]] = [("all", scope_frame)]
        groups.extend((name, frame) for name, frame in scope_frame.groupby("direction", sort=True))
        for group, frame in groups:
            returns = frame["ret_pct"].dropna()
            rows.append(
                {
                    "scope": scope,
                    "group": group,
                    "entry_delay_m": config.entry_delay_minutes,
                    "exit_delay_m": config.exit_delay_minutes,
                    "n": int(returns.count()),
                    "wins": int((returns > 0).sum()),
                    "losses": int((returns <= 0).sum()),
                    "mean_pct": returns.mean(),
                    "median_pct": returns.median(),
                    "win_rate_pct": (returns > 0).mean() * 100 if not returns.empty else pd.NA,
                    "min_pct": returns.min(),
                    "max_pct": returns.max(),
                    "sum_pct": returns.sum(),
                }
            )
    return pd.DataFrame(rows)


def collect_next_day_reaction_files(source_dir: str | Path, results_dir: str | Path) -> dict[str, Path]:
    source = Path(source_dir)
    reaction_source = source / "next_day_reaction"
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    mapping = {
        "sidecar_only": (
            [
                reaction_source / "next_day_reaction_sidecar_only.csv",
                source / "sidecar_event_summary_no_cb.csv",
            ],
            results / "next_day_reaction_sidecar_only.csv",
        ),
        "with_cb": (
            [
                reaction_source / "next_day_reaction_with_cb.csv",
                source / "sidecar_event_summary_with_cb.csv",
            ],
            results / "next_day_reaction_with_cb.csv",
        ),
        "legacy_full": (
            [
                reaction_source / "next_day_reaction_legacy_full.csv",
                source / "sidecar_event_summary.csv",
            ],
            results / "next_day_reaction_legacy_full.csv",
        ),
    }
    written: dict[str, Path] = {}
    for name, (src_candidates, dst) in mapping.items():
        src = next((candidate for candidate in src_candidates if candidate.exists()), None)
        if src is not None:
            shutil.copyfile(src, dst)
            written[name] = dst
    summary = summarize_next_day_reactions(written.values())
    if not summary.empty:
        summary.to_csv(results / "next_day_reaction_summary.csv", index=False, encoding="utf-8-sig")
    return written


def summarize_next_day_reactions(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        frame = pd.read_csv(path)
        if "kospi200_next_return_pct" not in frame.columns or "direction" not in frame.columns:
            continue
        clean = frame.dropna(subset=["kospi200_next_return_pct"]).copy()
        if clean.empty:
            continue
        trigger_sign = clean["direction"].map({"Up trigger": 1, "Down trigger": -1})
        next_sign = clean["kospi200_next_return_pct"].map(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
        clean["same_as_trigger"] = next_sign.eq(trigger_sign)
        clean["opposite_trigger"] = next_sign.eq(-trigger_sign)
        for group, group_frame in [("all", clean), *list(clean.groupby("direction", sort=True))]:
            rows.append(
                {
                    "file": path.name,
                    "group": group,
                    "n": len(group_frame),
                    "same_as_trigger_pct": group_frame["same_as_trigger"].mean() * 100,
                    "opposite_trigger_pct": group_frame["opposite_trigger"].mean() * 100,
                    "mean_kospi200_next_return_pct": group_frame["kospi200_next_return_pct"].mean(),
                    "median_kospi200_next_return_pct": group_frame["kospi200_next_return_pct"].median(),
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    parquet_dir: str | Path = DEFAULT_PARQUET_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    *,
    entry_delay_minutes: int = DEFAULT_ENTRY_DELAY_MINUTES,
    exit_delay_minutes: int = DEFAULT_EXIT_DELAY_MINUTES,
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    paths = build_parquet_bundle(source_dir, parquet_dir)
    trades, summary = run_event_study(
        parquet_dir,
        entry_delay_minutes=entry_delay_minutes,
        exit_delay_minutes=exit_delay_minutes,
    )
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    suffix = f"entry{entry_delay_minutes}_exit{exit_delay_minutes}"
    trades.to_csv(results / f"event_study_{suffix}_trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results / f"event_study_{suffix}_summary.csv", index=False, encoding="utf-8-sig")
    write_excel_report(trades, summary, results / f"event_study_{suffix}_report.xlsx")
    collect_next_day_reaction_files(source_dir, results)
    return paths, trades, summary


def write_excel_report(trades: pd.DataFrame, summary: pd.DataFrame, output_path: str | Path) -> Path:
    """Write event-study trades and summaries to Excel."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm:ss", date_format="yyyy-mm-dd") as writer:
        trades.to_excel(writer, sheet_name="trades", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        if "year" in trades.columns:
            for year in sorted(trades["year"].dropna().unique()):
                trades[trades["year"] == year].to_excel(writer, sheet_name=f"trades_{int(year)}", index=False)
        _format_workbook(writer.book)
    return output


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
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_DIR), help="Directory containing source Excel files.")
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET_DIR), help="Directory for normalized parquet outputs.")
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_DIR), help="Directory for event-study and next-day outputs.")
    parser.add_argument("--entry-delay-minutes", type=int, default=DEFAULT_ENTRY_DELAY_MINUTES)
    parser.add_argument("--exit-delay-minutes", type=int, default=DEFAULT_EXIT_DELAY_MINUTES)
    args = parser.parse_args(argv)

    paths, trades, summary = run_pipeline(
        args.source,
        args.parquet,
        args.results,
        entry_delay_minutes=args.entry_delay_minutes,
        exit_delay_minutes=args.exit_delay_minutes,
    )
    print("parquet:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("\nsummary:")
    print(summary.round(4).to_string(index=False))
    print(f"\ntrades: {len(trades)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
