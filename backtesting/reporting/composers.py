from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .models import ComparisonBundle, TearsheetBundle

__all__ = (
    "ComparisonComposer",
    "ComparisonRenderContext",
    "CoverContext",
    "MetricStripItem",
    "PageContext",
    "SectionContext",
    "TableContext",
    "TearsheetComposer",
    "TearsheetRenderContext",
)

_TABLE_TITLES = {
    "performance_summary": "Performance Summary",
    "drawdown_episodes": "Worst Drawdowns",
    "top_holdings": "Top Holdings",
    "sector_weights": "Sector Weights",
    "validation_appendix": "Validation Appendix",
    "ranked_summary": "Ranked Summary",
    "benchmark_relative": "Benchmark Relative Metrics",
    "exposure_summary": "Holdings And Turnover",
    "sector_summary": "Sector Comparison",
}
_PAGE_TITLES = {
    "executive": "Executive Overview",
    "rolling": "Rolling Diagnostics",
    "calendar": "Return Shape",
    "exposure": "Holdings And Sectors",
    "performance": "Performance Dashboard",
}
_PERCENT_COLUMNS = {
    "cagr",
    "cumulative_return",
    "annual_volatility",
    "max_drawdown",
    "avg_turnover",
    "alpha",
    "tracking_error",
    "weight",
    "top_sector_weight",
    "drawdown",
    "target_weight",
    "abs_weight",
}
_NUMBER_COLUMNS = {"beta", "sharpe", "sortino", "calmar", "information_ratio"}
_INTEGER_COLUMNS = {"count", "holdings_count", "duration_days", "recovery_days"}
_MONEY_COLUMNS = {"final_equity"}
_PERCENT_METRICS = {
    "Cumulative Return",
    "CAGR",
    "Volatility",
    "Max Drawdown",
    "Avg Turnover",
    "Alpha",
    "Tracking Error",
}
_NUMBER_METRICS = {"Sharpe", "Sortino", "Calmar", "Beta", "Information Ratio"}
_MONEY_METRICS = {"Final Equity"}


@dataclass(frozen=True, slots=True)
class PageContext:
    key: str
    title: str
    path: str
    kind: str


@dataclass(frozen=True, slots=True)
class TableContext:
    key: str
    title: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class CoverContext:
    report_type: str
    title: str
    subtitle: str
    benchmark_name: str
    report_name: str
    descriptor: str


@dataclass(frozen=True, slots=True)
class MetricStripItem:
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class SectionContext:
    title: str
    pages: tuple[PageContext, ...]
    tables: tuple[TableContext, ...]


@dataclass(frozen=True, slots=True)
class TearsheetRenderContext:
    cover: CoverContext
    executive_metrics: tuple[MetricStripItem, ...]
    executive_pages: tuple[PageContext, ...]
    executive_tables: tuple[TableContext, ...]
    sections: tuple[SectionContext, ...]
    notes: tuple[str, ...]
    metric_cards: tuple[dict[str, str], ...]
    pages: tuple[PageContext, ...]
    tables: tuple[TableContext, ...]
    title: str
    report_name: str
    display_name: str
    benchmark_name: str


@dataclass(frozen=True, slots=True)
class ComparisonRenderContext:
    cover: CoverContext
    executive_metrics: tuple[MetricStripItem, ...]
    executive_pages: tuple[PageContext, ...]
    executive_tables: tuple[TableContext, ...]
    sections: tuple[SectionContext, ...]
    notes: tuple[str, ...]
    metric_cards: tuple[dict[str, str], ...]
    pages: tuple[PageContext, ...]
    tables: tuple[TableContext, ...]
    title: str
    report_name: str
    benchmark_name: str
    participants: tuple[str, ...]


class TearsheetComposer:
    def compose(self, bundle: TearsheetBundle) -> TearsheetRenderContext:
        pages = _page_contexts(bundle.pages, bundle.out_dir)
        tables = _table_contexts(bundle.tables)
        executive_pages, executive_tables, sections = _split_tearsheet_sections(pages, tables)
        executive_metrics = _metric_strip(bundle.tables.get("performance_summary", pd.DataFrame()))
        benchmark_name = _benchmark_label(bundle.spec)
        return TearsheetRenderContext(
            cover=CoverContext(
                report_type="Single-Run Tearsheet",
                title=bundle.spec.title or bundle.display_name,
                subtitle=bundle.display_name,
                benchmark_name=benchmark_name,
                report_name=bundle.spec.name,
                descriptor="PDF-first single-run performance summary",
            ),
            executive_metrics=executive_metrics,
            executive_pages=executive_pages,
            executive_tables=executive_tables,
            sections=sections,
            notes=bundle.notes,
            metric_cards=_metric_cards_from_strip(executive_metrics),
            pages=pages,
            tables=tables,
            title=bundle.spec.title or bundle.display_name,
            report_name=bundle.spec.name,
            display_name=bundle.display_name,
            benchmark_name=benchmark_name,
        )


class ComparisonComposer:
    def compose(self, bundle: ComparisonBundle) -> ComparisonRenderContext:
        pages = _page_contexts(bundle.pages, bundle.out_dir)
        tables = _table_contexts(bundle.tables)
        executive_pages, executive_tables, sections = _split_comparison_sections(pages, tables)
        ranked = bundle.tables.get("ranked_summary", pd.DataFrame())
        executive_metrics = _comparison_metric_strip(ranked)
        benchmark_name = _benchmark_label(bundle.spec)
        return ComparisonRenderContext(
            cover=CoverContext(
                report_type="Comparison Report",
                title=bundle.spec.title or bundle.spec.name,
                subtitle=", ".join(bundle.display_names),
                benchmark_name=benchmark_name,
                report_name=bundle.spec.name,
                descriptor="Cross-strategy comparison optimized for PDF review",
            ),
            executive_metrics=executive_metrics,
            executive_pages=executive_pages,
            executive_tables=executive_tables,
            sections=sections,
            notes=bundle.notes,
            metric_cards=_metric_cards_from_strip(executive_metrics),
            pages=pages,
            tables=tables,
            title=bundle.spec.title or bundle.spec.name,
            report_name=bundle.spec.name,
            benchmark_name=benchmark_name,
            participants=bundle.display_names,
        )


def _comparison_metric_strip(frame: pd.DataFrame) -> tuple[MetricStripItem, ...]:
    if frame.empty:
        return ()
    cagr_leader = frame.sort_values(["cagr", "display_name"], ascending=[False, True]).iloc[0]
    cards = [
        MetricStripItem(
            label="Top CAGR",
            value=f'{cagr_leader["display_name"]} · {_format_value("cagr", cagr_leader.get("cagr"))}',
        )
    ]
    if "sharpe" in frame.columns:
        sharpe_leader = frame.sort_values(["sharpe", "display_name"], ascending=[False, True]).iloc[0]
        cards.append(
            MetricStripItem(
                label="Top Sharpe",
                value=f'{sharpe_leader["display_name"]} · {_format_value("sharpe", sharpe_leader.get("sharpe"))}',
            )
        )
    return tuple(cards)


def _metric_strip(frame: pd.DataFrame) -> tuple[MetricStripItem, ...]:
    if frame.empty:
        return ()
    cards: list[MetricStripItem] = []
    for row in frame.head(8).to_dict(orient="records"):
        label = str(row.get("metric", ""))
        cards.append(
            MetricStripItem(
                label=label,
                value=_format_metric_value(str(row.get("metric_key", "")), label, row.get("value")),
            )
        )
    return tuple(cards)


def _metric_cards_from_strip(items: tuple[MetricStripItem, ...]) -> tuple[dict[str, str], ...]:
    return tuple({"label": item.label, "value": item.value} for item in items)


def _split_tearsheet_sections(
    pages: tuple[PageContext, ...],
    tables: tuple[TableContext, ...],
) -> tuple[tuple[PageContext, ...], tuple[TableContext, ...], tuple[SectionContext, ...]]:
    executive_pages = tuple(page for page in pages if page.key == "performance")
    executive_tables = tuple(table for table in tables if table.key in {"performance_summary", "drawdown_episodes"})
    sections = (
        SectionContext(
            title="Holdings And Sectors",
            pages=(),
            tables=tuple(table for table in tables if table.key in {"top_holdings", "sector_weights"}),
        ),
        SectionContext(
            title="Appendix",
            pages=(),
            tables=tuple(table for table in tables if table.key == "validation_appendix"),
        ),
    )
    return executive_pages, executive_tables, tuple(section for section in sections if section.pages or section.tables)


def _split_comparison_sections(
    pages: tuple[PageContext, ...],
    tables: tuple[TableContext, ...],
) -> tuple[tuple[PageContext, ...], tuple[TableContext, ...], tuple[SectionContext, ...]]:
    executive_pages = tuple(page for page in pages if page.key in {"executive", "performance"})
    executive_tables = tuple(table for table in tables if table.key in {"ranked_summary", "benchmark_relative"})
    sections = (
        SectionContext(
            title="Rolling And Relative Diagnostics",
            pages=tuple(page for page in pages if page.key == "rolling"),
            tables=(),
        ),
        SectionContext(
            title="Holdings And Sector Comparison",
            pages=tuple(page for page in pages if page.key == "exposure"),
            tables=tuple(table for table in tables if table.key in {"exposure_summary", "sector_summary"}),
        ),
    )
    return executive_pages, executive_tables, tuple(section for section in sections if section.pages or section.tables)


def _page_contexts(pages: dict[str, Path], out_dir: Path) -> tuple[PageContext, ...]:
    items: list[PageContext] = []
    for key, path in pages.items():
        items.append(
            PageContext(
                key=key,
                title=_PAGE_TITLES.get(key, key.replace("_", " ").title()),
                path=os.path.relpath(path, out_dir).replace(os.sep, "/"),
                kind=path.suffix.lower().lstrip(".") or "file",
            )
        )
    return tuple(items)


def _benchmark_label(spec) -> str:
    benchmark = spec.benchmark
    if benchmark is None:
        return "Strategy Only"
    return benchmark.name


def _table_contexts(tables: dict[str, pd.DataFrame]) -> tuple[TableContext, ...]:
    items: list[TableContext] = []
    for key, frame in tables.items():
        display_columns = tuple(str(column) for column in frame.columns if not _is_internal_column(key, str(column)))
        rows = tuple(
            {
                str(column): _format_table_cell(key, row, str(column), value)
                for column, value in row.items()
                if not _is_internal_column(key, str(column))
            }
            for row in frame.to_dict(orient="records")
        )
        items.append(
            TableContext(
                key=key,
                title=_TABLE_TITLES.get(key, key.replace("_", " ").title()),
                columns=display_columns,
                rows=rows,
            )
        )
    return tuple(items)


def _format_table_cell(table_key: str, row: dict[str, object], column: str, value: object) -> str:
    if table_key == "performance_summary" and column == "value":
        return _format_metric_value(str(row.get("metric_key", "")), str(row.get("metric", "")), value)
    return _format_value(column, value)


def _format_metric_value(metric_key: str, metric_label: str, value: object) -> str:
    key = metric_key.strip().lower()
    if key in _PERCENT_COLUMNS:
        return _format_value(key, value)
    if key in _NUMBER_COLUMNS:
        return _format_value(key, value)
    if key in _MONEY_COLUMNS:
        return _format_value(key, value)

    label = metric_label.strip()
    if label in _PERCENT_METRICS:
        return _format_value("cagr", value)
    if label in _NUMBER_METRICS:
        return _format_value("sharpe", value)
    if label in _MONEY_METRICS:
        return _format_value("final_equity", value)
    return _format_value("value", value)


def _format_value(column: str, value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "isoformat") and not isinstance(value, (str, bytes, int, float)):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, str):
        return value

    numeric = float(value)
    name = column.strip().lower()
    if name in _PERCENT_COLUMNS:
        return f"{numeric:.1%}"
    if name in _NUMBER_COLUMNS:
        return f"{numeric:.2f}"
    if name in _INTEGER_COLUMNS:
        return f"{int(round(numeric)):,}"
    if name in _MONEY_COLUMNS:
        return f"{numeric:,.0f}"
    return f"{numeric:,.2f}"


def _is_internal_column(table_key: str, column: str) -> bool:
    return table_key == "performance_summary" and column == "metric_key"
