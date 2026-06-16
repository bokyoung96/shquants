from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtesting.run import BacktestRunner
from backtesting.strategies.emp008.comparison import _benchmark_returns, excess_summary_bps, performance_metrics
from backtesting.strategies.emp008.mfbt_emp008 import run_mfbt_emp008
from backtesting.strategies.emp008.mfbt_emp008_data import MfbtEmp008Config, load_mfbt_emp008_market
from backtesting.strategies.emp008.mfbt_emp008_factors import build_raw_mfbt_factors
from backtesting.strategies.emp008.mfbt_emp008_preprocess import preprocess_factor_frame
from backtesting.strategies.emp008.run_backtest import (
    active_share_summary,
    build_target_weight_spec,
    write_active_share,
)
from backtesting.strategies.emp008.run_weights import build_emp008_config, write_target_weights_csv


DEFAULT_VARIANTS = (
    "baseline",
    "value_zcap_5",
    "value_zcap_3",
    "value_winsor_1pct",
)


@dataclass(frozen=True, slots=True)
class Variant:
    name: str
    config: MfbtEmp008Config


def build_variants(base_config: MfbtEmp008Config) -> dict[str, Variant]:
    return {
        "baseline": Variant("baseline", base_config),
        "value_zcap_5": Variant("value_zcap_5", replace(base_config, value_zscore_cap=5.0)),
        "value_zcap_3": Variant("value_zcap_3", replace(base_config, value_zscore_cap=3.0)),
        "value_winsor_1pct": Variant(
            "value_winsor_1pct",
            replace(base_config, value_raw_winsor_quantile=0.01),
        ),
        "value_winsor_1pct_zcap_5": Variant(
            "value_winsor_1pct_zcap_5",
            replace(base_config, value_raw_winsor_quantile=0.01, value_zscore_cap=5.0),
        ),
    }


def run_value_factor_robustness(
    *,
    parquet_dir: Path,
    output_dir: Path,
    start: str,
    end: str,
    tracking_error_annual: float,
    risk_model: str,
    variants: tuple[str, ...],
    force: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = build_emp008_config(
        tracking_error_annual=tracking_error_annual,
        risk_model=risk_model,
        factor_set="mfbt",
    )
    variant_map = build_variants(base_config)
    unknown = tuple(name for name in variants if name not in variant_map)
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    value_diagnostics = value_exposure_diagnostics(
        parquet_dir=parquet_dir,
        start=start,
        end=end,
        variants=tuple(variant_map[name] for name in variants),
    )
    value_diagnostics.to_csv(output_dir / "value_exposure_diagnostics.csv", index=False)

    returns_by_variant: dict[str, pd.Series] = {}
    summary_rows: list[dict[str, object]] = []
    for variant_name in variants:
        variant = variant_map[variant_name]
        variant_dir = output_dir / variant.name
        result = run_variant(
            variant=variant,
            parquet_dir=parquet_dir,
            variant_dir=variant_dir,
            start=start,
            end=end,
            force=force,
        )
        returns = result["returns"]
        bm_returns = _benchmark_returns(parquet_dir / "qw_BM.parquet", "IKS200", returns.index)
        excess_returns = returns.sub(bm_returns.reindex(returns.index).fillna(0.0))
        returns_by_variant[variant.name] = returns

        metrics = performance_metrics(returns, periods_per_year=252)
        excess = excess_summary_bps(excess_returns.rename("gross_excess").to_frame(), periods_per_year=252).loc[
            "gross_excess"
        ]
        active_share = active_share_summary(variant_dir / "weights" / "active_share.parquet")
        row = {
            "variant": variant.name,
            "value_raw_winsor_quantile": variant.config.value_raw_winsor_quantile,
            "value_zscore_cap": variant.config.value_zscore_cap,
            "returns_csv": str(result["returns_csv"]),
            "weights_dir": str(variant_dir / "weights"),
            **{f"strategy_{key}": value for key, value in metrics.items()},
            **{f"excess_{key}": float(value) for key, value in excess.items()},
            **{f"active_share_{key}": value for key, value in active_share.items()},
        }
        summary_rows.append(row)
        (variant_dir / "variant_summary.json").write_text(
            json.dumps(_json_ready(row), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary = pd.DataFrame(summary_rows).sort_values("excess_total_excess_bp", ascending=False)
    summary.to_csv(output_dir / "variant_summary.csv", index=False)
    with pd.ExcelWriter(output_dir / "variant_summary.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        value_diagnostics.to_excel(writer, sheet_name="value_exposure", index=False)

    plot_cumulative_excess(
        returns_by_variant=returns_by_variant,
        benchmark_returns=_benchmark_returns(parquet_dir / "qw_BM.parquet", "IKS200"),
        path=output_dir / "cumulative_excess_by_variant.png",
    )
    plot_active_share(output_dir=output_dir, variants=variants, path=output_dir / "active_share_by_variant.png")

    payload = {
        "output_dir": str(output_dir),
        "summary_csv": str(output_dir / "variant_summary.csv"),
        "summary_xlsx": str(output_dir / "variant_summary.xlsx"),
        "value_exposure_diagnostics_csv": str(output_dir / "value_exposure_diagnostics.csv"),
        "cumulative_excess_png": str(output_dir / "cumulative_excess_by_variant.png"),
        "active_share_png": str(output_dir / "active_share_by_variant.png"),
        "variants": variants,
        "start": start,
        "end": end,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def run_variant(
    *,
    variant: Variant,
    parquet_dir: Path,
    variant_dir: Path,
    start: str,
    end: str,
    force: bool,
) -> dict[str, object]:
    weights_dir = variant_dir / "weights"
    weights_csv = weights_dir / "target_weights.csv"
    active_weights_path = weights_dir / "active_weights.parquet"
    if force or not weights_csv.exists() or not active_weights_path.exists():
        result = run_mfbt_emp008(
            parquet_dir=parquet_dir,
            start=start,
            end=end,
            config=variant.config,
            output_dir=weights_dir,
        )
        write_target_weights_csv(result.target_weights, weights_csv)
        write_active_share(active_weights_path)

    existing = _load_existing_backtest(variant_dir)
    if force or existing is None:
        dates = tuple(pd.to_datetime(pd.read_csv(weights_csv, index_col=0, usecols=[0]).index).strftime("%Y-%m-%d"))
        spec = build_target_weight_spec(
            name=f"mfbt_value_robust_{variant.name}",
            weights_csv=weights_csv,
            dates=dates,
            end=end,
            fill_mode="close",
            capital=100_000_000.0,
            fee=0.0,
            sell_tax=0.0,
            slippage=0.0,
            allow_fractional=True,
        )
        runner = BacktestRunner(result_dir=variant_dir / "backtests", write_report_assets=False, profile=True)
        report = runner.run_spec(runner.resolve_spec(spec))
        output_dir = Path(str(report.output_dir))
        returns_csv = output_dir / "series" / "returns.csv"
        metadata = {
            "variant": variant.name,
            "config": asdict(variant.config),
            "backtest_output_dir": str(output_dir),
            "returns_csv": str(returns_csv),
        }
        (variant_dir / "backtest_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        returns_csv = existing

    returns = pd.read_csv(returns_csv, parse_dates=["date"]).set_index("date")["returns"].astype(float).sort_index()
    return {"returns": returns, "returns_csv": returns_csv}


def value_exposure_diagnostics(
    *,
    parquet_dir: Path,
    start: str,
    end: str,
    variants: tuple[Variant, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in variants:
        market = load_mfbt_emp008_market(parquet_dir=parquet_dir, start=start, end=end, config=variant.config)
        raw_value = build_raw_mfbt_factors(market, variant.config)["value"]
        close = market.frames["close"].astype(float)
        float_mktcap = market.frames["float_market_cap"].reindex(index=close.index, columns=close.columns).astype(float)
        universe = market.frames["k200_yn"].reindex(index=close.index, columns=close.columns).fillna(0).astype(bool)
        exposure = preprocess_factor_frame(
            raw_value,
            float_mktcap,
            universe,
            winsor_quantile=variant.config.value_raw_winsor_quantile,
            zscore_cap=variant.config.value_zscore_cap,
        )
        abs_exposure = exposure.abs().where(universe)
        stacked = abs_exposure.stack().sort_values(ascending=False)
        top_date, top_ticker = stacked.index[0]
        rows.append(
            {
                "variant": variant.name,
                "value_raw_winsor_quantile": variant.config.value_raw_winsor_quantile,
                "value_zscore_cap": variant.config.value_zscore_cap,
                "max_abs_value_zscore": float(stacked.iloc[0]),
                "max_abs_value_zscore_date": pd.Timestamp(top_date).date().isoformat(),
                "max_abs_value_zscore_ticker": str(top_ticker),
                "p99_abs_value_zscore": float(stacked.quantile(0.99)),
                "p95_abs_value_zscore": float(stacked.quantile(0.95)),
                "mean_abs_value_zscore": float(stacked.mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_cumulative_excess(
    *,
    returns_by_variant: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    for name, returns in returns_by_variant.items():
        bm = benchmark_returns.reindex(returns.index).fillna(0.0)
        cumulative = (1.0 + returns.sub(bm)).cumprod().sub(1.0).mul(100.0)
        ax.plot(cumulative.index, cumulative.to_numpy(), label=name)
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_title("MFBT value robustification cumulative gross excess")
    ax.set_ylabel("Cumulative excess return (%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_active_share(*, output_dir: Path, variants: tuple[str, ...], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    for name in variants:
        active_share_path = output_dir / name / "weights" / "active_share.parquet"
        if not active_share_path.exists():
            continue
        active_share = pd.read_parquet(active_share_path)
        active_share.index = pd.to_datetime(active_share.index)
        ax.plot(active_share.index, active_share["active_share_pct"].astype(float), label=name)
    ax.set_title("MFBT value robustification active share")
    ax.set_ylabel("Active share (%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_existing_backtest(variant_dir: Path) -> Path | None:
    metadata_path = variant_dir / "backtest_metadata.json"
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    returns_csv = Path(payload["returns_csv"])
    return returns_csv if returns_csv.exists() else None


def _json_ready(row: dict[str, object]) -> dict[str, object]:
    ready: dict[str, object] = {}
    for key, value in row.items():
        if pd.isna(value):
            ready[key] = None
        elif hasattr(value, "item"):
            ready[key] = value.item()
        else:
            ready[key] = value
    return ready


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest MFBT EMP008 value factor robustification variants.")
    parser.add_argument("--parquet-dir", type=Path, default=Path("parquet"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtesting/strategies/emp008/mfbt_emp008_experiments/results/value_factor_robustness"),
    )
    parser.add_argument("--start", default="2022-12-29")
    parser.add_argument("--end", default="2026-05-28")
    parser.add_argument("--tracking-error-annual", type=float, default=0.007)
    parser.add_argument("--risk-model", choices=("factor_idio", "direct_covariance"), default="factor_idio")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = run_value_factor_robustness(
        parquet_dir=args.parquet_dir,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        tracking_error_annual=args.tracking_error_annual,
        risk_model=args.risk_model,
        variants=tuple(args.variants),
        force=args.force,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
