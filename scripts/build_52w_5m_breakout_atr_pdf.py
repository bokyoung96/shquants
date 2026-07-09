from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = (
    ROOT
    / "results"
    / "flow_filtered_breakout_single"
    / "sector_pos90_margin002_flow_or_60d_2019start_confirmed_episode"
    / "research"
    / "52w_5m_breakout_atr_final"
)
RESULT_PDF = REPORT_DIR / "professional_strategy_report.pdf"
OUTPUT_PDF_DIR = ROOT / "output" / "pdf"
OUTPUT_PDF = OUTPUT_PDF_DIR / "52w_5m_breakout_atr_strategy_report.pdf"


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100.0:.{digits}f}%"


def bps(value: float, digits: int = 2) -> str:
    return f"{value * 10_000.0:.{digits}f} bps"


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/malgun.ttf")
    bold = Path("C:/Windows/Fonts/malgunbd.ttf")
    if not regular.exists() or not bold.exists():
        regular = Path("C:/Windows/Fonts/NotoSansKR-VF.ttf")
        bold = regular
    pdfmetrics.registerFont(TTFont("KRRegular", str(regular)))
    pdfmetrics.registerFont(TTFont("KRBold", str(bold)))
    return "KRRegular", "KRBold"


def build_styles() -> dict[str, ParagraphStyle]:
    regular, bold = register_fonts()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKR",
            parent=base["Title"],
            fontName=bold,
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f2f35"),
            spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleKR",
            parent=base["Normal"],
            fontName=regular,
            fontSize=10,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#54636a"),
            spaceAfter=22,
        ),
        "h1": ParagraphStyle(
            "H1KR",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#23343a"),
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2KR",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#2f4f4f"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyKR",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=9,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallKR",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7.6,
            leading=10.5,
            textColor=colors.HexColor("#4f5d62"),
        ),
        "table": ParagraphStyle(
            "TableKR",
            parent=base["BodyText"],
            fontName=regular,
            fontSize=7.4,
            leading=9.5,
        ),
        "table_bold": ParagraphStyle(
            "TableBoldKR",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=7.4,
            leading=9.5,
            textColor=colors.white,
        ),
    }


def para(text: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(text), style)


def styled_table(
    rows: list[list[object]],
    styles: dict[str, ParagraphStyle],
    *,
    col_widths: list[float] | None = None,
    header: bool = True,
) -> Table:
    converted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        row_style = styles["table_bold"] if row_index == 0 and header else styles["table"]
        converted.append([para(cell, row_style) for cell in row])
    table = Table(converted, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d7dcde")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2f4f4f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ]
        )
    for row in range(1 if header else 0, len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#f5f6f4")))
    table.setStyle(TableStyle(commands))
    return table


def add_section(story: list[object], title: str, styles: dict[str, ParagraphStyle]) -> None:
    story.append(Paragraph(title, styles["h1"]))


def add_paragraphs(story: list[object], paragraphs: Iterable[str], styles: dict[str, ParagraphStyle]) -> None:
    for item in paragraphs:
        story.append(Paragraph(item, styles["body"]))


def page_footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFont("KRRegular", 7)
    canvas.setFillColor(colors.HexColor("#6c777c"))
    canvas.drawString(doc.leftMargin, 0.65 * cm, "52W High 5M Breakout + ATR Strategy")
    canvas.drawRightString(A4[0] - doc.rightMargin, 0.65 * cm, f"Page {doc.page}")
    canvas.restoreState()


def metrics_from_inputs() -> dict[str, object]:
    trades = pd.read_csv(REPORT_DIR / "selected_trades.csv", parse_dates=["signal_time", "entry_time", "exit_time"])
    ledger = pd.read_csv(REPORT_DIR / "fixed_notional_ledger.csv", parse_dates=["date"]).set_index("date")
    yearly = pd.read_csv(REPORT_DIR / "yearly_returns.csv")
    returns = trades["net_return"].astype(float)
    trades["hold_days"] = (trades["exit_time"].dt.normalize() - trades["entry_time"].dt.normalize()).dt.days
    wealth = ledger["equity"].astype(float)
    daily_ret = wealth.pct_change().fillna(0.0)
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    trough = dd.idxmin()
    peak_date = wealth.loc[:trough].idxmax()
    recovery = wealth.loc[trough:][wealth.loc[trough:] >= wealth.loc[peak_date]]
    recovery_date = recovery.index[0] if not recovery.empty else pd.NaT
    start = trades["entry_time"].min().normalize()
    end = ledger.index.max().normalize()
    years = (end - start).days / 365.25
    final_return = float(wealth.iloc[-1] - 1.0)
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0)
    wins = returns[returns > 0.0]
    losses = returns[returns < 0.0]
    monthly = wealth.resample("ME").last().dropna().pct_change().dropna()
    return {
        "trades": trades,
        "ledger": ledger,
        "yearly": yearly,
        "returns": returns,
        "start": start,
        "end": end,
        "years": years,
        "final_return": final_return,
        "cagr": cagr,
        "mdd": float(dd.min()),
        "peak_date": peak_date,
        "trough": trough,
        "recovery_date": recovery_date,
        "ann_vol": float(daily_ret.std() * (252 ** 0.5)),
        "sharpe": float((daily_ret.mean() / daily_ret.std()) * (252 ** 0.5)) if daily_ret.std() else 0.0,
        "sortino": float((daily_ret.mean() / daily_ret[daily_ret < 0.0].std()) * (252 ** 0.5))
        if not daily_ret[daily_ret < 0.0].empty and daily_ret[daily_ret < 0.0].std()
        else 0.0,
        "monthly": monthly,
        "avg_win": float(wins.mean()),
        "avg_loss": float(losses.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())),
        "top": trades.nlargest(10, "net_return"),
        "bottom": trades.nsmallest(10, "net_return"),
    }


def build_pdf(output_path: Path = RESULT_PDF) -> Path:
    OUTPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    data = metrics_from_inputs()
    trades = data["trades"]
    returns = data["returns"]
    yearly = data["yearly"]
    monthly = data["monthly"]
    top = data["top"]
    bottom = data["bottom"]
    best = top.iloc[0]

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.45 * cm,
        leftMargin=1.45 * cm,
        topMargin=1.35 * cm,
        bottomMargin=1.15 * cm,
        title="52W High 5M Breakout + ATR Strategy Report",
        author="shquants",
    )
    usable_width = A4[0] - doc.leftMargin - doc.rightMargin
    story: list[object] = []

    story.append(Paragraph("52주 신고가 5분봉 Breakout + ATR 전략 보고서", styles["title"]))
    story.append(
        Paragraph(
            f"KOSPI200 Long-only Momentum Strategy | {data['start'].date()} - {data['end'].date()} | 비용 차감 후 기준",
            styles["subtitle"],
        )
    )
    add_paragraphs(
        story,
        [
            "본 전략은 KOSPI200 유니버스에서 52주 종가 신고가 돌파가 발생한 뒤, 다음 5분봉에서도 돌파 상태가 유지되는 경우에만 진입하는 단순 모멘텀 전략이다.",
            "최종 채택된 형태는 수급, positivity, 섹터 상대강도, 변동성 압축 필터를 제외하고 가격 기반 신고가 돌파와 ATR 손절만 남긴 구조다.",
        ],
        styles,
    )

    summary_rows = [
        ["Metric", "Value", "Metric", "Value"],
        ["총 거래 수", f"{len(trades):,}", "고유 종목 수", f"{trades.ticker.nunique():,}"],
        ["Fixed 20-slot return", pct(data["final_return"]), "CAGR", pct(data["cagr"])],
        ["MDD", pct(data["mdd"]), "Calmar", f"{data['cagr'] / abs(data['mdd']):.2f}"],
        ["Avg trade", bps(returns.mean()), "Median trade", bps(returns.median())],
        ["Hit rate", pct(returns.gt(0).mean()), "Profit factor", f"{data['profit_factor']:.3f}"],
        ["Best trade", "A009150 / 삼성전기", "Best net return", pct(float(best.net_return))],
    ]
    story.append(styled_table(summary_rows, styles, col_widths=[4.2 * cm, 4.5 * cm, 4.1 * cm, 4.5 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Image(str(REPORT_DIR / "performance.png"), width=usable_width, height=usable_width * 0.60))

    story.append(PageBreak())
    add_section(story, "1. 최종 전략 Scheme", styles)
    scheme_rows = [
        ["항목", "내용"],
        ["Universe", "KOSPI200 historical members"],
        ["Direction", "Long only"],
        ["Signal", "52주 종가 신고가 돌파"],
        ["Intraday filter", "09:20 이후 발생한 돌파만 사용"],
        ["Confirmation", "다음 5분봉 close도 prior 52w close high 위에 있어야 함"],
        ["Entry fill", "confirmation 다음 5분봉 open"],
        ["Initial stop", "entry price - 1.0 * ATR"],
        ["Exit 1", "다음 거래일부터 daily low가 ATR stop 이하이면 stop price 체결"],
        ["Exit 2", "다음 거래일부터 daily close가 prior 52w close high 이하이면 종가 청산"],
        ["Position sizing", "fixed 20-slot notional, 종목당 5%, 레버리지 없음"],
        ["Cost", "round-trip 35 bps, net return에 반영"],
        ["Excluded filters", "positivity, 외인/기관 수급, weekly sector RS, daily volatility compression, close-based trailing"],
    ]
    story.append(styled_table(scheme_rows, styles, col_widths=[4.1 * cm, 12.8 * cm]))
    add_section(story, "2. 비용 및 체결 가정", styles)
    add_paragraphs(
        story,
        [
            "비용은 왕복 35bp로 고정했다. 모든 거래의 net return은 gross_return = exit_price / entry_price - 1, net_return = gross_return - 0.0035 방식으로 계산된다.",
            "ATR stop은 daily low가 stop price를 터치하면 stop price에 체결된 것으로 처리한다. 52주 고점 이탈 exit는 daily close 기준이며, 해당 일의 종가에 청산한다.",
        ],
        styles,
    )

    add_section(story, "3. Portfolio Performance", styles)
    perf_rows = [
        ["지표", "값", "지표", "값"],
        ["Final return", pct(data["final_return"]), "CAGR", pct(data["cagr"])],
        ["MDD", pct(data["mdd"]), "Annualized daily volatility", pct(data["ann_vol"])],
        ["Daily Sharpe, rf=0", f"{data['sharpe']:.2f}", "Daily Sortino, rf=0", f"{data['sortino']:.2f}"],
        ["Best month", pct(monthly.max()), "Worst month", pct(monthly.min())],
        ["Positive month ratio", pct(monthly.gt(0.0).mean()), "Average active positions", f"{data['ledger'].active_positions.mean():.2f}"],
        ["Max active positions", f"{int(data['ledger'].active_positions.max())}", "Days active", pct(data["ledger"].active_positions.gt(0).mean())],
    ]
    story.append(styled_table(perf_rows, styles, col_widths=[4.2 * cm, 4.3 * cm, 4.3 * cm, 4.1 * cm]))

    add_section(story, "4. Drawdown", styles)
    dd_rows = [
        ["항목", "값"],
        ["Max drawdown", pct(data["mdd"])],
        ["MDD peak date", str(data["peak_date"].date())],
        ["MDD trough date", str(data["trough"].date())],
        ["Recovery date", str(data["recovery_date"].date()) if not pd.isna(data["recovery_date"]) else "Not recovered"],
    ]
    story.append(styled_table(dd_rows, styles, col_widths=[6 * cm, 6 * cm]))

    story.append(PageBreak())
    add_section(story, "5. Trade-Level Statistics", styles)
    trade_rows = [
        ["지표", "값", "지표", "값"],
        ["Trades", f"{len(trades):,}", "Unique tickers", f"{trades.ticker.nunique():,}"],
        ["Average trade", bps(returns.mean()), "Median trade", bps(returns.median())],
        ["Hit rate", pct(returns.gt(0).mean()), "Profit factor", f"{data['profit_factor']:.3f}"],
        ["Average win", bps(data["avg_win"]), "Average loss", bps(data["avg_loss"])],
        ["Payoff ratio", f"{data['avg_win'] / abs(data['avg_loss']):.2f}", "Trade return stdev", pct(returns.std())],
        ["Skewness", f"{returns.skew():.2f}", "Excess kurtosis", f"{returns.kurt():.2f}"],
        ["Worst trade", pct(returns.min()), "Best trade", pct(returns.max())],
        ["Average holding days", f"{trades.hold_days.mean():.2f}", "Max holding days", f"{int(trades.hold_days.max())}"],
    ]
    story.append(styled_table(trade_rows, styles, col_widths=[4.2 * cm, 4.3 * cm, 4.3 * cm, 4.1 * cm]))

    add_section(story, "6. Return Distribution", styles)
    quantiles = pd.read_csv(REPORT_DIR / "return_distribution_summary.csv")
    q_rows = [["분위수", "net return", "bps"]]
    for row in quantiles.itertuples(index=False):
        q_rows.append([row.quantile, f"{row.net_return_pct:.2f}%", f"{row.net_return_bps:,.2f}"])
    story.append(styled_table(q_rows, styles, col_widths=[3.5 * cm, 4.5 * cm, 4.5 * cm]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Image(str(REPORT_DIR / "return_distribution_deep_dive.png"), width=usable_width, height=usable_width * 0.60))

    story.append(PageBreak())
    add_section(story, "7. Exit Reason Analysis", styles)
    exit_stats = trades.groupby("exit_reason")["net_return"].agg(["count", "mean", "median", "min", "max"]).reset_index()
    exit_rows = [["Exit reason", "Trades", "Ratio", "Avg net", "Median", "Min", "Max"]]
    for row in exit_stats.itertuples(index=False):
        exit_rows.append(
            [
                row.exit_reason,
                f"{row.count:,}",
                pct(row.count / len(trades)),
                pct(row.mean),
                pct(row.median),
                pct(row.min),
                pct(row.max),
            ]
        )
    story.append(styled_table(exit_rows, styles, col_widths=[3.4 * cm, 2 * cm, 2.2 * cm, 2.4 * cm, 2.4 * cm, 2.2 * cm, 2.2 * cm]))
    add_paragraphs(
        story,
        [
            "ATR stop은 전체의 약 75.7%를 차지한다. 반면 new_high_lost로 종료된 거래의 평균 수익률은 +5.28%로, 전체 수익의 대부분을 만든다.",
            "즉, exit 구조는 많은 작은 실패와 적은 큰 성공을 의도적으로 허용한다.",
        ],
        styles,
    )

    add_section(story, "8. Yearly Stability", styles)
    yearly_rows = [["Year", "Trades", "Fixed return", "Avg trade", "Hit rate"]]
    for row in yearly.itertuples(index=False):
        label = f"{int(row.year)}" if int(row.year) < 2026 else "2026 YTD"
        yearly_rows.append([label, f"{int(row.trades):,}", pct(row.year_return), bps(row.avg_trade_return), pct(row.hit_rate)])
    story.append(styled_table(yearly_rows, styles, col_widths=[3 * cm, 3 * cm, 3.4 * cm, 3.8 * cm, 3.4 * cm]))

    story.append(PageBreak())
    add_section(story, "9. Top Winners and Tail Dependence", styles)
    top_rows = [["Rank", "Ticker", "Entry", "Exit", "Entry price", "Exit price", "Net", "Hold"]]
    for index, row in enumerate(top.itertuples(index=False), start=1):
        ticker = "A009150 / 삼성전기" if row.ticker == "A009150" else row.ticker
        top_rows.append(
            [
                index,
                ticker,
                row.entry_time.date(),
                row.exit_time.date(),
                f"{row.entry_price:,.0f}",
                f"{row.exit_price:,.0f}",
                pct(row.net_return),
                f"{int(row.hold_days)}",
            ]
        )
    story.append(styled_table(top_rows, styles, col_widths=[1.2 * cm, 3.4 * cm, 2.3 * cm, 2.3 * cm, 2.5 * cm, 2.5 * cm, 1.7 * cm, 1.2 * cm]))
    sorted_ret = returns.sort_values(ascending=False).reset_index(drop=True)
    contrib_rows = [["Top N trades", "Portfolio contribution", "Share of final return"]]
    for n in [1, 5, 10, 25, 50, 100]:
        contribution = float(sorted_ret.head(n).sum() / 20.0)
        contrib_rows.append([n, pct(contribution), pct(contribution / data["final_return"])])
    story.append(Spacer(1, 0.2 * cm))
    story.append(styled_table(contrib_rows, styles, col_widths=[4 * cm, 5 * cm, 5 * cm]))

    add_section(story, "10. Worst Trades", styles)
    bottom_rows = [["Rank", "Ticker", "Entry", "Exit", "Entry price", "Exit price", "Net", "Reason"]]
    for index, row in enumerate(bottom.itertuples(index=False), start=1):
        bottom_rows.append(
            [
                index,
                row.ticker,
                row.entry_time.date(),
                row.exit_time.date(),
                f"{row.entry_price:,.0f}",
                f"{row.exit_price:,.0f}",
                pct(row.net_return),
                row.exit_reason,
            ]
        )
    story.append(styled_table(bottom_rows, styles, col_widths=[1.2 * cm, 2.2 * cm, 2.3 * cm, 2.3 * cm, 2.5 * cm, 2.5 * cm, 1.7 * cm, 2.2 * cm]))

    story.append(PageBreak())
    add_section(story, "11. Economic Interpretation", styles)
    add_paragraphs(
        story,
        [
            "이 전략의 경제적 가정은 단순하다. 52주 신고가는 시장 참여자들이 기존 가격 앵커를 재평가하는 구간이다. 장중 5분봉 확인을 요구하면 단순한 순간 체결이나 얇은 호가 돌파를 일부 걸러내고, 실제 수급이 신고가 위에서 유지되는 사건을 포착하게 된다.",
            "전략의 edge는 예측 정확도에서 나오지 않는다. hit rate가 22.83%에 불과하기 때문이다. edge는 손실을 ATR로 제한하면서, 일부 종목이 정보 재평가 또는 수급 쏠림으로 짧은 기간에 크게 상승할 때 그 right-tail을 포획하는 데서 나온다.",
            "따라서 이 전략은 평균회귀형 전략이 아니라 convex momentum 전략이다. 다수의 작은 손실을 사업 비용처럼 지불하고, 드물게 발생하는 큰 winner를 통해 전체 손익을 만든다. 삼성전기(A009150) 거래가 이 구조를 가장 잘 보여준다.",
        ],
        styles,
    )
    add_section(story, "12. Why Filters Were Removed", styles)
    add_paragraphs(
        story,
        [
            "이전 실험에서 positivity, 외인/기관 수급, weekly sector RS, daily volatility compression 등을 검토했지만 최종 canonical 전략에는 넣지 않았다.",
            "추가 필터가 종목 수를 줄일 수는 있어도 winner를 놓치는 비용이 컸고, 임계값 기반 필터는 쉽게 overfitting으로 변질된다. 현재 전략은 의도적으로 단순하다.",
        ],
        styles,
    )
    add_section(story, "13. Risk and Implementation Caveats", styles)
    caveats = [
        "- 비용은 왕복 35bp로 고정되어 있으며, 종목별 유동성/호가 스프레드 차이는 별도 모델링하지 않았다.",
        "- ATR stop은 stop price 체결을 가정한다. 갭다운에서는 실제 손실이 더 커질 수 있다.",
        "- KOSPI200 historical members를 사용하지만, 데이터 구성 방식에 따라 survivorship 또는 index membership timing 이슈를 별도로 점검해야 한다.",
        "- 성과는 소수 right-tail winner에 민감하다. 필터 추가나 daily cap은 반드시 winner 누락률을 함께 평가해야 한다.",
    ]
    add_paragraphs(story, caveats, styles)
    add_section(story, "14. Conclusion", styles)
    add_paragraphs(
        story,
        [
            "현재 최종 전략은 복잡한 quality score 전략이 아니라, 단순한 52주 신고가 5분봉 confirmation breakout 전략이다. 성과의 본질은 낮은 승률, 제한된 손실, 강한 우측 꼬리다.",
            "향후 개선은 필터를 더 붙이는 방식보다, winner를 덜 놓치면서 손실 tail을 과도하게 키우지 않는 방향이어야 한다. Candle body size 또는 body-to-range ratio는 사후 최적화 숫자가 아니라 breakout bar의 질을 사전에 정의하는 진단 변수로 연구하는 편이 더 적절하다.",
        ],
        styles,
    )

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    if output_path != OUTPUT_PDF:
        shutil.copy2(output_path, OUTPUT_PDF)
    return output_path


def main() -> None:
    path = build_pdf()
    print(path)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
