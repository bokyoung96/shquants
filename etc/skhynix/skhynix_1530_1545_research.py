from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_WORKBOOK = BASE_DIR / "SK Hynix analysis.xlsx"
ROUND_TRIP_COST_BPS = 2.0


def _sign(series: pd.Series) -> pd.Series:
    return np.sign(series.fillna(0.0)).astype(int)


def _ret_bps(end: pd.Series | float, start: pd.Series | float) -> pd.Series | float:
    return (end / start - 1.0) * 10_000.0


def prepare_parquet() -> dict:
    raw = pd.read_excel(SOURCE_WORKBOOK, sheet_name="data", header=None, engine="openpyxl")
    investor_raw = pd.read_excel(
        SOURCE_WORKBOOK, sheet_name="data2", header=None, engine="openpyxl"
    )

    spot = raw.iloc[2:, 0:3].copy()
    spot.columns = [
        "timestamp",
        "sk_hynix_intraday_close",
        "kodex_sk_hynix_single_stock_leverage_intraday_close",
    ]
    spot = spot.dropna(how="all")
    spot = spot[spot["timestamp"].notna()].reset_index(drop=True)
    spot["timestamp"] = pd.to_datetime(spot["timestamp"])
    for col in spot.columns.drop("timestamp"):
        spot[col] = pd.to_numeric(spot[col], errors="coerce")
    spot.to_parquet(BASE_DIR / "intraday_spot_etf.parquet", index=False, engine="pyarrow")

    futures = raw.iloc[2:, 4:7].copy()
    futures.columns = [
        "timestamp",
        "sk_hynix_futures_2607_intraday_close",
        "sk_hynix_futures_2607_intraday_cumulative_volume",
    ]
    futures = futures.dropna(how="all")
    futures = futures[futures["timestamp"].notna()].reset_index(drop=True)
    futures["timestamp"] = pd.to_datetime(futures["timestamp"])
    for col in futures.columns.drop("timestamp"):
        futures[col] = pd.to_numeric(futures[col], errors="coerce")
    futures.to_parquet(BASE_DIR / "intraday_futures_2607.parquet", index=False, engine="pyarrow")

    investor = investor_raw.iloc[2:, 0:9].copy()
    investor.columns = [
        "date",
        "institutional_net_buy_qty_ex_spread",
        "foreign_net_buy_qty_ex_spread",
        "individual_net_buy_qty_ex_spread",
        "securities_net_buy_qty_ex_spread",
        "investment_trust_net_buy_qty_ex_spread",
        "insurance_net_buy_qty_ex_spread",
        "pension_fund_net_buy_qty_ex_spread",
        "other_corporation_net_buy_qty_ex_spread",
    ]
    investor = investor.dropna(how="all")
    investor = investor[investor["date"].notna()].reset_index(drop=True)
    investor["date"] = pd.to_datetime(investor["date"]).dt.date
    for col in investor.columns.drop("date"):
        investor[col] = pd.to_numeric(investor[col], errors="coerce").astype("Int64")
    investor.to_parquet(BASE_DIR / "futures_investor_net_buy.parquet", index=False, engine="pyarrow")

    manifest = {
        "source_workbook": "etc/skhynix/SK Hynix analysis.xlsx",
        "tables": [
            {
                "file": "intraday_spot_etf.parquet",
                "source_sheet": "data",
                "source_range": "A:C, rows 3+",
                "rows": int(len(spot)),
                "columns": list(spot.columns),
                "original_headers": {
                    "timestamp": "시간/항목",
                    "sk_hynix_intraday_close": "SK하이닉스 / Intra종가",
                    "kodex_sk_hynix_single_stock_leverage_intraday_close": (
                        "KODEX SK하이닉스단일종목레버리지 / Intra종가"
                    ),
                },
            },
            {
                "file": "intraday_futures_2607.parquet",
                "source_sheet": "data",
                "source_range": "E:G, rows 3+",
                "rows": int(len(futures)),
                "columns": list(futures.columns),
                "original_headers": {
                    "timestamp": "시간/항목",
                    "sk_hynix_futures_2607_intraday_close": "SK하이닉스 선물 2607 / Intra종가",
                    "sk_hynix_futures_2607_intraday_cumulative_volume": (
                        "SK하이닉스 선물 2607 / Intra누적거래량"
                    ),
                },
            },
            {
                "file": "futures_investor_net_buy.parquet",
                "source_sheet": "data2",
                "source_range": "A:I, rows 3+",
                "rows": int(len(investor)),
                "columns": list(investor.columns),
                "original_headers": {
                    "date": "시간/항목",
                    "institutional_net_buy_qty_ex_spread": "종목투자자별SPREAD제외순매수수량 - 기관계",
                    "foreign_net_buy_qty_ex_spread": "종목투자자별SPREAD제외순매수수량 - 외국인",
                    "individual_net_buy_qty_ex_spread": "종목투자자별SPREAD제외순매수수량 - 개인",
                    "securities_net_buy_qty_ex_spread": "종목투자자별SPREAD제외순매수수량 - 금융투자(증권)",
                    "investment_trust_net_buy_qty_ex_spread": "종목투자자별SPREAD제외순매수수량 - 투신",
                    "insurance_net_buy_qty_ex_spread": "종목투자자별SPREAD제외순매수수량 - 보험",
                    "pension_fund_net_buy_qty_ex_spread": "종목투자자별SPREAD제외순매수수량 - 연기금",
                    "other_corporation_net_buy_qty_ex_spread": "종목투자자별SPREAD제외순매수수량 - 기타법인",
                },
            },
        ],
    }
    (BASE_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _value_at_time(
    df: pd.DataFrame,
    value_col: str,
    time: str,
    *,
    require_exact: bool = True,
) -> pd.Series:
    target = pd.Timestamp(time).time()
    data = df[df["timestamp"].dt.time == target].copy()
    if require_exact:
        return data.set_index("date")[value_col]
    data = df[df["timestamp"].dt.time <= target].sort_values("timestamp")
    return data.groupby("date").tail(1).set_index("date")[value_col]


def build_daily_features() -> pd.DataFrame:
    spot = pd.read_parquet(BASE_DIR / "intraday_spot_etf.parquet")
    futures = pd.read_parquet(BASE_DIR / "intraday_futures_2607.parquet")
    investor = pd.read_parquet(BASE_DIR / "futures_investor_net_buy.parquet")

    spot["date"] = spot["timestamp"].dt.date
    futures["date"] = futures["timestamp"].dt.date
    investor = investor.sort_values("date").reset_index(drop=True)
    investor_prev = investor.copy()
    investor_prev["date"] = investor_prev["date"].shift(-1)
    investor_prev = investor_prev.dropna(subset=["date"]).add_prefix("prev_")
    investor_prev = investor_prev.rename(columns={"prev_date": "date"})
    investor_same_day = investor.add_prefix("same_day_").rename(columns={"same_day_date": "date"})

    features = pd.DataFrame(index=sorted(set(spot["date"]) & set(futures["date"])))
    features.index.name = "date"

    stock = "sk_hynix_intraday_close"
    etf = "kodex_sk_hynix_single_stock_leverage_intraday_close"
    fut_px = "sk_hynix_futures_2607_intraday_close"
    fut_vol = "sk_hynix_futures_2607_intraday_cumulative_volume"

    spot_open = spot.groupby("date").head(1).set_index("date")
    fut_open = futures.groupby("date").head(1).set_index("date")

    stock_1525 = _value_at_time(spot, stock, "15:25")
    stock_1530 = _value_at_time(spot, stock, "15:30")
    etf_1525 = _value_at_time(spot, etf, "15:25")
    etf_1530 = _value_at_time(spot, etf, "15:30")
    fut_1525 = _value_at_time(futures, fut_px, "15:25")
    fut_1530 = _value_at_time(futures, fut_px, "15:30")
    fut_1535 = _value_at_time(futures, fut_px, "15:35")
    fut_1545 = _value_at_time(futures, fut_px, "15:45")
    fut_vol_1525 = _value_at_time(futures, fut_vol, "15:25")
    fut_vol_1530 = _value_at_time(futures, fut_vol, "15:30")

    features["target_fut_1530_1545_bps"] = _ret_bps(fut_1545, fut_1530)
    features["target_fut_1530_1535_bps"] = _ret_bps(fut_1535, fut_1530)
    features["target_fut_1535_1545_bps"] = _ret_bps(fut_1545, fut_1535)
    features["stock_day_to_1530_bps"] = _ret_bps(stock_1530, spot_open[stock])
    features["etf_day_to_1530_bps"] = _ret_bps(etf_1530, spot_open[etf])
    features["fut_day_to_1530_bps"] = _ret_bps(fut_1530, fut_open[fut_px])
    features["stock_1525_1530_bps"] = _ret_bps(stock_1530, stock_1525)
    features["etf_1525_1530_bps"] = _ret_bps(etf_1530, etf_1525)
    features["fut_1525_1530_bps"] = _ret_bps(fut_1530, fut_1525)
    features["fut_volume_1525_1530"] = fut_vol_1530 - fut_vol_1525
    features["fut_signed_volume_pressure_1525_1530"] = (
        _sign(features["fut_1525_1530_bps"]) * features["fut_volume_1525_1530"]
    )
    features["fut_close_volume_prior_median"] = (
        features["fut_volume_1525_1530"].expanding(min_periods=3).median().shift(1)
    )
    features["fut_close_volume_is_heavy"] = (
        features["fut_volume_1525_1530"] > features["fut_close_volume_prior_median"]
    )
    features["etf_excess_vs_2x_stock_day_bps"] = (
        features["etf_day_to_1530_bps"] - 2.0 * features["stock_day_to_1530_bps"]
    )
    features["etf_excess_vs_2x_stock_close5_bps"] = (
        features["etf_1525_1530_bps"] - 2.0 * features["stock_1525_1530_bps"]
    )
    features["fut_vs_stock_day_bps"] = (
        features["fut_day_to_1530_bps"] - features["stock_day_to_1530_bps"]
    )

    features = features.reset_index()
    features = features.merge(investor_prev, on="date", how="left")
    features = features.merge(investor_same_day, on="date", how="left")
    features = features.dropna(subset=["target_fut_1530_1545_bps"]).reset_index(drop=True)
    return features


@dataclass(frozen=True)
class Strategy:
    name: str
    description: str
    signal: pd.Series


def build_strategies(features: pd.DataFrame) -> list[Strategy]:
    rally = features["fut_day_to_1530_bps"] > 0
    strong_rally = features["fut_day_to_1530_bps"] > 100

    same_trust = features["same_day_investment_trust_net_buy_qty_ex_spread"] > 0
    same_foreign = features["same_day_foreign_net_buy_qty_ex_spread"] > 0
    same_securities = features["same_day_securities_net_buy_qty_ex_spread"] > 0
    same_institutional = features["same_day_institutional_net_buy_qty_ex_spread"] > 0

    prev_trust = features["prev_investment_trust_net_buy_qty_ex_spread"] > 0
    prev_foreign = features["prev_foreign_net_buy_qty_ex_spread"] > 0
    prev_securities = features["prev_securities_net_buy_qty_ex_spread"] > 0
    prev_institutional = features["prev_institutional_net_buy_qty_ex_spread"] > 0

    heavy_volume_pressure = (
        features["fut_close_volume_is_heavy"]
        & (_sign(features["fut_signed_volume_pressure_1525_1530"]) > 0)
    )

    def long_when(condition: pd.Series) -> pd.Series:
        return pd.Series(np.where(condition.fillna(False), 1, 0), index=features.index)

    def short_when(condition: pd.Series) -> pd.Series:
        return -long_when(condition)

    return [
        Strategy(
            "same_day_oracle_rally_trust_long",
            "Needs intraday investor-flow feed: long when futures has rallied by 15:30 and investment-trust futures flow is net buying.",
            long_when(rally & same_trust),
        ),
        Strategy(
            "same_day_oracle_rally_trust_short",
            "Needs intraday investor-flow feed: fade when futures has rallied by 15:30 and investment-trust futures flow is net buying.",
            short_when(rally & same_trust),
        ),
        Strategy(
            "same_day_oracle_strong_rally_trust_long",
            "Needs intraday investor-flow feed: long when futures rally is above 100 bps and investment-trust futures flow is net buying.",
            long_when(strong_rally & same_trust),
        ),
        Strategy(
            "same_day_oracle_strong_rally_trust_short",
            "Needs intraday investor-flow feed: fade when futures rally is above 100 bps and investment-trust futures flow is net buying.",
            short_when(strong_rally & same_trust),
        ),
        Strategy(
            "same_day_oracle_rally_trust_foreign_long",
            "Needs intraday investor-flow feed: long when rally, investment-trust buying, and foreign futures buying line up.",
            long_when(rally & same_trust & same_foreign),
        ),
        Strategy(
            "same_day_oracle_rally_trust_foreign_short",
            "Needs intraday investor-flow feed: fade when rally, investment-trust buying, and foreign futures buying line up.",
            short_when(rally & same_trust & same_foreign),
        ),
        Strategy(
            "same_day_oracle_rally_trust_institutional_long",
            "Needs intraday investor-flow feed: long when rally, investment-trust buying, and institutional futures buying line up.",
            long_when(rally & same_trust & same_institutional),
        ),
        Strategy(
            "same_day_oracle_rally_trust_securities_sell_short",
            "Needs intraday investor-flow feed: short when rally and investment-trust buying happen while securities futures flow is selling.",
            short_when(rally & same_trust & ~same_securities),
        ),
        Strategy(
            "t1_rally_prev_trust_long",
            "Tradable with current file: long when futures has rallied by 15:30 and T-1 investment-trust futures flow was net buying.",
            long_when(rally & prev_trust),
        ),
        Strategy(
            "t1_rally_prev_trust_short",
            "Tradable with current file: fade when futures has rallied by 15:30 and T-1 investment-trust futures flow was net buying.",
            short_when(rally & prev_trust),
        ),
        Strategy(
            "t1_strong_rally_prev_trust_long",
            "Tradable with current file: long when futures rally is above 100 bps and T-1 investment-trust futures flow was net buying.",
            long_when(strong_rally & prev_trust),
        ),
        Strategy(
            "t1_strong_rally_prev_trust_short",
            "Tradable with current file: fade when futures rally is above 100 bps and T-1 investment-trust futures flow was net buying.",
            short_when(strong_rally & prev_trust),
        ),
        Strategy(
            "t1_rally_prev_trust_foreign_long",
            "Tradable with current file: long when rally, T-1 investment-trust buying, and T-1 foreign buying line up.",
            long_when(rally & prev_trust & prev_foreign),
        ),
        Strategy(
            "t1_rally_prev_trust_foreign_short",
            "Tradable with current file: fade when rally, T-1 investment-trust buying, and T-1 foreign buying line up.",
            short_when(rally & prev_trust & prev_foreign),
        ),
        Strategy(
            "t1_rally_prev_trust_institutional_long",
            "Tradable with current file: long when rally, T-1 investment-trust buying, and T-1 institutional buying line up.",
            long_when(rally & prev_trust & prev_institutional),
        ),
        Strategy(
            "t1_rally_prev_trust_securities_sell_short",
            "Tradable with current file: short when rally and T-1 investment-trust buying happen while T-1 securities futures flow was selling.",
            short_when(rally & prev_trust & ~prev_securities),
        ),
        Strategy(
            "heavy_futures_pressure_after_rally_short",
            "Proxy from futures tape only: short when the market rallied and 15:25-15:30 futures pressure was positive on heavy volume.",
            short_when(rally & heavy_volume_pressure),
        ),
    ]


def _max_drawdown(cumulative: pd.Series) -> float:
    if cumulative.empty:
        return 0.0
    return float((cumulative - cumulative.cummax()).min())


def evaluate_strategy(features: pd.DataFrame, strategy: Strategy) -> tuple[dict, pd.DataFrame]:
    daily = features[["date", "target_fut_1530_1545_bps"]].copy()
    daily["strategy"] = strategy.name
    daily["signal"] = strategy.signal.to_numpy(dtype=int)
    daily["gross_bps"] = daily["signal"] * daily["target_fut_1530_1545_bps"]
    daily["cost_bps"] = daily["signal"].abs() * ROUND_TRIP_COST_BPS
    daily["net_bps"] = daily["gross_bps"] - daily["cost_bps"]
    trades = daily[daily["signal"] != 0].copy()

    if trades.empty:
        metrics = {
            "strategy": strategy.name,
            "description": strategy.description,
            "n_days": int(len(daily)),
            "n_trades": 0,
            "hit_rate": np.nan,
            "avg_gross_bps": np.nan,
            "avg_net_bps": np.nan,
            "median_net_bps": np.nan,
            "total_net_bps": 0.0,
            "max_drawdown_bps": 0.0,
            "worst_leave_one_out_total_net_bps": 0.0,
            "largest_win_share_of_total_net": np.nan,
        }
        return metrics, daily

    total_net = float(trades["net_bps"].sum())
    positive_net = trades.loc[trades["net_bps"] > 0, "net_bps"]
    largest_win_share = (
        float(positive_net.max() / total_net)
        if total_net > 0 and not positive_net.empty
        else np.nan
    )
    metrics = {
        "strategy": strategy.name,
        "description": strategy.description,
        "n_days": int(len(daily)),
        "n_trades": int(len(trades)),
        "hit_rate": float((trades["gross_bps"] > 0).mean()),
        "avg_gross_bps": float(trades["gross_bps"].mean()),
        "avg_net_bps": float(trades["net_bps"].mean()),
        "median_net_bps": float(trades["net_bps"].median()),
        "total_net_bps": total_net,
        "max_drawdown_bps": _max_drawdown(trades["net_bps"].cumsum()),
        "worst_leave_one_out_total_net_bps": float((total_net - trades["net_bps"]).min()),
        "largest_win_share_of_total_net": largest_win_share,
    }
    return metrics, daily


def feature_correlations(features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "fut_day_to_1530_bps",
        "fut_1525_1530_bps",
        "fut_volume_1525_1530",
        "fut_signed_volume_pressure_1525_1530",
        "fut_vs_stock_day_bps",
        "prev_institutional_net_buy_qty_ex_spread",
        "prev_foreign_net_buy_qty_ex_spread",
        "prev_individual_net_buy_qty_ex_spread",
        "prev_securities_net_buy_qty_ex_spread",
        "prev_investment_trust_net_buy_qty_ex_spread",
        "prev_pension_fund_net_buy_qty_ex_spread",
        "same_day_institutional_net_buy_qty_ex_spread",
        "same_day_foreign_net_buy_qty_ex_spread",
        "same_day_securities_net_buy_qty_ex_spread",
        "same_day_investment_trust_net_buy_qty_ex_spread",
        "same_day_pension_fund_net_buy_qty_ex_spread",
    ]
    rows = []
    target = features["target_fut_1530_1545_bps"]
    for col in feature_cols:
        if col not in features or features[col].isna().all():
            continue
        rows.append(
            {
                "feature": col,
                "pearson_corr": float(features[col].corr(target, method="pearson")),
                "spearman_corr": float(features[col].corr(target, method="spearman")),
                "n": int(features[[col, "target_fut_1530_1545_bps"]].dropna().shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_corr", key=lambda s: s.abs(), ascending=False)


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        else:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))

    columns = list(display.columns)
    rows = display.to_dict("records")
    widths = {
        col: max([len(col), *[len(row[col]) for row in rows]]) if rows else len(col)
        for col in columns
    }
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in columns) + " |"
    body = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_report(
    features: pd.DataFrame,
    summary: pd.DataFrame,
    corr: pd.DataFrame,
) -> None:
    best = summary.sort_values("total_net_bps", ascending=False).iloc[0]
    tradable = summary[summary["strategy"].str.startswith("t1_")].sort_values(
        "total_net_bps",
        ascending=False,
    )
    oracle = summary[summary["strategy"].str.startswith("same_day_oracle_")].sort_values(
        "total_net_bps",
        ascending=False,
    )
    liquid_candidates = summary[summary["n_trades"] >= 3].sort_values(
        "total_net_bps",
        ascending=False,
    )
    best_tradable = tradable.iloc[0] if not tradable.empty else best
    best_oracle = oracle.iloc[0] if not oracle.empty else best
    best_liquid = liquid_candidates.iloc[0] if not liquid_candidates.empty else best
    window_diag = (
        features[
            [
                "target_fut_1530_1545_bps",
                "target_fut_1530_1535_bps",
                "target_fut_1535_1545_bps",
            ]
        ]
        .agg(["count", "mean", "median", "min", "max"])
        .T.reset_index()
        .rename(columns={"index": "window"})
    )
    markdown = [
        "# SK Hynix 15:30-15:45 Futures Strategy Research",
        "",
        "## Data",
        "",
        f"- Complete 15:30 entry / 15:45 exit samples: {len(features)} trading days.",
        "- Signal timestamp discipline: signals use information available by 15:30 only.",
        "- Cost assumption: 2.0 bps round-trip per futures trade.",
        "- 2026-06-11 is excluded because the futures sheet stops before 15:45.",
        "- The supplied investor-flow table is daily close-level futures net buy by investor type.",
        "- `t1_` strategies are tradable with this file alone because they use T-1 futures flow plus same-day futures rally by 15:30.",
        "- `same_day_oracle_` strategies test the intended live workflow: if a real-time/intraday futures investor-flow feed is available by 15:30, use same-day flow. The current workbook does not prove that availability.",
        "- ETF price/NAV/creation flow is not used in this version; the focus is futures 수급.",
        "",
        "## Window Diagnostic",
        "",
        window_diag.round(4).pipe(markdown_table),
        "",
        "- The 15:35-15:45 leg is material, so these tests are partly about the post-close futures auction/settlement print rather than continuous 15:30-15:45 liquidity.",
        "",
        "## Futures Flow Strategy Ideas Tested",
        "",
        "- Rally + investment trust: trade only when futures has rallied by 15:30 and investment-trust futures flow is net buying.",
        "- Strong rally + investment trust: same rule but require day-to-15:30 futures rally above 100 bps.",
        "- Rally + investment trust + foreign/institutional confirmation: require additional buyer participation.",
        "- Rally + investment trust + securities sell: test whether apparent buy-side demand is exhausted/offset by securities selling.",
        "- Heavy futures tape pressure after rally: use 15:25-15:30 futures price/volume pressure as a non-investor-type proxy.",
        "- Each idea is tested as both continuation/long and fade/short where economically relevant.",
        "",
        "## Backtest Summary",
        "",
        summary[
            [
                "strategy",
                "n_trades",
                "hit_rate",
                "avg_net_bps",
                "median_net_bps",
                "total_net_bps",
                "max_drawdown_bps",
                "worst_leave_one_out_total_net_bps",
                "largest_win_share_of_total_net",
            ]
        ].round(4).pipe(markdown_table),
        "",
        "## Feature Correlations To 15:30-15:45 Futures Return",
        "",
        corr.head(10).round(4).pipe(markdown_table),
        "",
        "## Read",
        "",
        f"- Best overall in-sample futures-flow result: `{best['strategy']}` at {best['total_net_bps']:.2f} bps net over {int(best['n_trades'])} trades.",
        f"- Best same-day/oracle version: `{best_oracle['strategy']}` at {best_oracle['total_net_bps']:.2f} bps net over {int(best_oracle['n_trades'])} trades.",
        f"- Best T-1 tradable version with this workbook alone: `{best_tradable['strategy']}` at {best_tradable['total_net_bps']:.2f} bps net over {int(best_tradable['n_trades'])} trades.",
        f"- Best candidate with at least 3 trades: `{best_liquid['strategy']}` at {best_liquid['total_net_bps']:.2f} bps net over {int(best_liquid['n_trades'])} trades.",
        f"- For that best strategy, the worst leave-one-day-out total is {best['worst_leave_one_out_total_net_bps']:.2f} bps; largest winning day share is {best['largest_win_share_of_total_net']:.2%}.",
        "- Treat two-trade results as event diagnostics, not deployable strategy evidence.",
        "- This is too small a sample for deployment. Treat positive results as hypotheses to retest with more post-listing days.",
        "- The most useful next data addition is intraday futures investor-flow by type up to 15:30, especially investment trust, foreign, securities, and institutional cumulative flow.",
        "",
    ]
    (BASE_DIR / "strategy_1530_1545_report.md").write_text("\n".join(markdown), encoding="utf-8")


def main() -> None:
    manifest = prepare_parquet()
    features = build_daily_features()
    features.to_csv(BASE_DIR / "strategy_1530_1545_daily_features.csv", index=False, encoding="utf-8-sig")

    strategies = build_strategies(features)
    metric_rows = []
    daily_rows = []
    for strategy in strategies:
        metrics, daily = evaluate_strategy(features, strategy)
        metric_rows.append(metrics)
        daily_rows.append(daily)

    summary = pd.DataFrame(metric_rows).sort_values("total_net_bps", ascending=False)
    daily = pd.concat(daily_rows, ignore_index=True)
    corr = feature_correlations(features)

    summary.to_csv(BASE_DIR / "strategy_1530_1545_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(BASE_DIR / "strategy_1530_1545_daily_returns.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(BASE_DIR / "strategy_1530_1545_feature_correlations.csv", index=False, encoding="utf-8-sig")
    write_report(features, summary, corr)

    print(json.dumps({"manifest_tables": manifest["tables"], "n_backtest_days": len(features)}, indent=2, ensure_ascii=False))
    print(summary[["strategy", "n_trades", "hit_rate", "avg_net_bps", "total_net_bps"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
