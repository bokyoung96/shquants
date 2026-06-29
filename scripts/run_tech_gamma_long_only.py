from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backtesting.data.kr_stock_5m import KrStock5mDataset, normalize_ticker, read_tickers_bars
from root import ROOT
from scripts.tech_gamma_holding import simulate_continuation_holding
from scripts.tech_gamma_intraday import TradeSide, simulate_intraday
from scripts.tech_gamma_plots import write_performance_outputs
from scripts.tech_gamma_research_filters import apply_research_features, load_research_feature_data
from scripts.tech_gamma_universe import filter_kospi200_historical_members, kospi200_tickers
from scripts.tech_gamma_schemes import get_scheme, scheme_names


ROUND_TRIP_BPS = 3.0


@dataclass(frozen=True, slots=True)
class TechGammaConfig:
    scheme: str = "tech_gamma"
    side: TradeSide = TradeSide.LONG
    universe: str = "custom"
    tickers: tuple[str, ...] = ("005930", "000660")
    start: str = "2024-01-01"
    end: str = "2026-12-31 23:59:59"
    range_end_hhmm: str = "1000"
    exit_hhmm: str = "1455"
    overnight_entry_hhmm: str = "1455"
    range_buffer_bps: float = 8.0
    min_score: float = 4.0
    min_overnight_score: float = 5.0
    stop_bps: float = 55.0
    trailing_bps: float = 45.0
    atr_lookback_bars: int = 14
    atr_stop_multiplier: float = 1.0
    min_holding_days: int = 1
    holding_mode: str = "intraday"
    use_positivity: bool = False
    positivity_lookback_days: int = 252
    min_daily_positivity: float = 0.55
    positivity_benchmark: str = "absolute"
    positivity_margin: float = 0.0
    factor_filter: str = "none"
    factor_lookback_days: int = 60
    overnight_enabled: bool = True
    high_lookback_days: int = 370


def load_strategy_frame(dataset: KrStock5mDataset, config: TechGammaConfig) -> pd.DataFrame:
    load_start = _load_start(config)
    raw = read_tickers_bars(dataset, _tickers_for_config(dataset, config), start=load_start, end=config.end)
    usable = raw.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
    frame = build_features(usable, config)
    if config.universe == "kospi200_historical":
        frame = filter_kospi200_historical_members(frame, dataset.root.parent)
    data = load_research_feature_data(dataset.root.parent, tuple(usable["ticker"].drop_duplicates()))
    frame = apply_research_features(frame, config, data)
    return frame.loc[frame["ts"].ge(pd.Timestamp(config.start))].reset_index(drop=True)


def build_features(raw: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    return get_scheme(config.scheme).build_features(raw, config)


def simulate_overnight(frame: pd.DataFrame, config: TechGammaConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scheme = get_scheme(config.scheme)
    for ticker, group in frame.sort_values("ts").groupby("ticker", sort=True):
        days = list(group.groupby("date", sort=True))
        for index, (date, day) in enumerate(days[:-1]):
            candidates = day[day["hhmm"].ge(config.overnight_entry_hhmm)]
            if candidates.empty:
                continue
            signal = candidates.iloc[-1]
            if not scheme.overnight_entry_ok(signal, config):
                continue
            next_day = days[index + 1][1]
            exit_row = next_day.iloc[0]
            gross = float(exit_row["open"]) / float(signal["close"]) - 1.0
            rows.append(
                {
                    "ticker": str(ticker),
                    "signal_date": pd.Timestamp(date).date().isoformat(),
                    "entry_time": signal["ts"],
                    "exit_time": exit_row["ts"],
                    "entry_price": float(signal["close"]),
                    "exit_price": float(exit_row["open"]),
                    "signal_score": float(signal["signal_score"]),
                    "gross_return": gross,
                    "net_return": gross - ROUND_TRIP_BPS / 10_000.0,
                }
            )
    return pd.DataFrame(rows)


def summarize(intraday: pd.DataFrame, overnight: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _summary_row("intraday", intraday, "exit_time"),
            _summary_row("overnight", overnight, "exit_time"),
            _summary_row("intraday_2024_2025", _period(intraday, "exit_time", 2024, 2025), "exit_time"),
            _summary_row("intraday_2026", _period(intraday, "exit_time", 2026, 2026), "exit_time"),
            _summary_row("overnight_2024_2025", _period(overnight, "exit_time", 2024, 2025), "exit_time"),
            _summary_row("overnight_2026", _period(overnight, "exit_time", 2026, 2026), "exit_time"),
        ]
    )


def run(config: TechGammaConfig, output_dir: Path | None = None) -> Path:
    dataset = KrStock5mDataset(ROOT.parquet_path / "KR_STOCK_5m")
    scheme = get_scheme(config.scheme)
    tickers = _tickers_for_config(dataset, config)
    frame = load_strategy_frame(dataset, config)
    intraday = simulate_continuation_holding(frame, config) if config.holding_mode == "continuation" else simulate_intraday(frame, config)
    overnight = simulate_overnight(frame, config) if config.overnight_enabled else _empty_trades()
    output = output_dir or ROOT.results_path / "tech_gamma_long_only" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    signal_columns = [*scheme.signal_columns]
    for column in (
        "daily_positivity",
        "positivity_benchmark",
        "positivity_spread",
        "positivity_filter_ok",
        "factor_filter_ok",
        "op_revision",
        "foreign_flow_to_cap",
        "institution_flow_to_cap",
        "sector_name",
    ):
        if column in frame.columns and column not in signal_columns:
            signal_columns.append(column)
    frame[["ts", "ticker", "close", *signal_columns]].to_csv(output / "signals.csv", index=False)
    intraday.to_csv(output / "intraday_trades.csv", index=False)
    overnight.to_csv(output / "overnight_trades.csv", index=False)
    summarize(intraday, overnight).to_csv(output / "summary.csv", index=False)
    pd.Series(tickers, name="ticker").to_csv(output / "universe_tickers.csv", index=False)
    write_performance_outputs(intraday, overnight, output, f"{config.side.value.title()} Scheme Performance")
    (output / "config.json").write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _tickers_for_config(dataset: KrStock5mDataset, config: TechGammaConfig) -> tuple[str, ...]:
    if config.universe == "custom":
        return tuple(normalize_ticker(ticker) for ticker in config.tickers)
    return kospi200_tickers(dataset.root.parent, config)


def _load_start(config: TechGammaConfig) -> pd.Timestamp:
    if config.scheme == "52w_high_breakout":
        return pd.Timestamp(config.start) - pd.Timedelta(days=config.high_lookback_days)
    return pd.Timestamp(config.start)


def _period(frame: pd.DataFrame, time_column: str, start_year: int, end_year: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    years = pd.to_datetime(frame[time_column]).dt.year
    return frame[years.between(start_year, end_year)]


def _summary_row(name: str, frame: pd.DataFrame, time_column: str) -> dict[str, object]:
    if frame.empty:
        return {"segment": name, "trades": 0, "net_return_sum": 0.0, "avg_net_bps": 0.0, "hit_rate": 0.0}
    return {
        "segment": name,
        "trades": int(len(frame)),
        "net_return_sum": float(frame["net_return"].sum()),
        "avg_net_bps": float(frame["net_return"].mean() * 10_000.0),
        "hit_rate": float(frame["net_return"].gt(0.0).mean()),
        "first_exit": str(pd.to_datetime(frame[time_column]).min()),
        "last_exit": str(pd.to_datetime(frame[time_column]).max()),
    }


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "signal_date",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "signal_score",
            "gross_return",
            "net_return",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Samsung/SK Hynix 5-minute tech gamma long-only research.")
    parser.add_argument("--scheme", choices=scheme_names(), default="tech_gamma")
    parser.add_argument("--side", choices=tuple(side.value for side in TradeSide), default=TradeSide.LONG.value)
    parser.add_argument("--universe", choices=("custom", "kospi200_latest", "kospi200_ever", "kospi200_historical"), default="custom")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-12-31 23:59:59")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-score", type=float, default=4.0)
    parser.add_argument("--min-overnight-score", type=float, default=5.0)
    parser.add_argument("--atr-lookback-bars", type=int, default=14)
    parser.add_argument("--atr-stop-multiplier", type=float, default=1.0)
    parser.add_argument("--min-holding-days", type=int, default=1)
    parser.add_argument("--holding-mode", choices=("intraday", "continuation"), default="intraday")
    parser.add_argument("--use-positivity", action="store_true")
    parser.add_argument("--positivity-lookback-days", type=int, default=252)
    parser.add_argument("--min-daily-positivity", type=float, default=0.55)
    parser.add_argument(
        "--positivity-benchmark",
        choices=("absolute", "sector_cap_weighted", "sector_equal_weight", "index_cap_weighted", "index_equal_weight"),
        default="absolute",
    )
    parser.add_argument("--positivity-margin", type=float, default=0.0)
    parser.add_argument(
        "--factor-filter",
        choices=(
            "none",
            "op_revision_positive",
            "op_sector_rank_positive",
            "foreign_positive",
            "institution_positive",
            "foreign_flow_positive",
            "institution_flow_positive",
            "foreign_or_institution_positive",
            "foreign_and_institution_positive",
            "op_or_flow_positive",
        ),
        default="none",
    )
    parser.add_argument("--factor-lookback-days", type=int, default=60)
    parser.add_argument("--intraday-only", action="store_true")
    parser.add_argument("--high-lookback-days", type=int, default=370)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(
        TechGammaConfig(
            start=args.start,
            end=args.end,
            scheme=args.scheme,
            side=TradeSide(args.side),
            universe=args.universe,
            min_score=args.min_score,
            min_overnight_score=args.min_overnight_score,
            atr_lookback_bars=args.atr_lookback_bars,
            atr_stop_multiplier=args.atr_stop_multiplier,
            min_holding_days=args.min_holding_days,
            holding_mode=args.holding_mode,
            use_positivity=args.use_positivity,
            positivity_lookback_days=args.positivity_lookback_days,
            min_daily_positivity=args.min_daily_positivity,
            positivity_benchmark=args.positivity_benchmark,
            positivity_margin=args.positivity_margin,
            factor_filter=args.factor_filter,
            factor_lookback_days=args.factor_lookback_days,
            overnight_enabled=not args.intraday_only,
            high_lookback_days=args.high_lookback_days,
        ),
        output_dir=args.output_dir,
    )
    print(output)


if __name__ == "__main__":
    main()
