"""Run output exports."""

from .builder import ReportBuilder
from .composers import ComparisonComposer, TearsheetComposer
from .html import HtmlRenderer
from .models import (
    BenchmarkConfig,
    ComparisonBundle,
    ReportBundle,
    ReportKind,
    ReportProfile,
    ReportSpec,
    SavedRun,
    TearsheetBundle,
)
from .reader import RunReader
from .tables_comparison import ComparisonTableBuilder
from .tables_single import TearsheetTableBuilder
from .writer import RunWriter


def __getattr__(name: str) -> object:
    if name in {"ReportCli", "main"}:
        from .cli import ReportCli, main

        return {"ReportCli": ReportCli, "main": main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "BenchmarkConfig",
    "ComparisonComposer",
    "ComparisonBundle",
    "ComparisonTableBuilder",
    "HtmlRenderer",
    "ReportBuilder",
    "ReportBundle",
    "ReportCli",
    "ReportKind",
    "ReportProfile",
    "ReportSpec",
    "RunReader",
    "RunWriter",
    "SavedRun",
    "TearsheetComposer",
    "TearsheetBundle",
    "TearsheetTableBuilder",
    "main",
)
