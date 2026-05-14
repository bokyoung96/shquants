from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pandas as pd

from backtesting.saved_runs import SavedRun


class ReportKind(str, Enum):
    TEARSHEET = "tearsheet"
    COMPARISON = "comparison"


class ReportProfile(str, Enum):
    ALPHA = "alpha"
    INDEX = "index"
    ABSOLUTE = "absolute"

    @classmethod
    def normalize(cls, value: "ReportProfile | str | None") -> "ReportProfile | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    code: str
    name: str
    dataset: str = "qw_BM"

    @classmethod
    def default_kospi200(cls) -> BenchmarkConfig:
        return cls(code="IKS200", name="KOSPI200")


@dataclass(frozen=True, slots=True)
class ReportSpec:
    name: str
    run_ids: tuple[str, ...]
    title: str | None = None
    include_factor: bool = True
    include_validation: bool = True
    include_is_oos: bool = True
    formats: tuple[str, ...] = ("html", "pdf")
    kind: ReportKind | None = None
    benchmark: BenchmarkConfig | None = field(default_factory=BenchmarkConfig.default_kospi200)
    profile: ReportProfile | None = None

    def __post_init__(self) -> None:
        if not self.run_ids:
            raise ValueError("run_ids must not be empty")
        kind = self.kind
        if kind is None:
            kind = ReportKind.TEARSHEET if len(self.run_ids) == 1 else ReportKind.COMPARISON
        else:
            try:
                kind = ReportKind(kind)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid report kind: {kind!r}") from exc

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "profile", ReportProfile.normalize(self.profile))

        if kind is ReportKind.TEARSHEET and len(self.run_ids) != 1:
            raise ValueError("TEARSHEET reports require exactly one run_id")
        if kind is ReportKind.COMPARISON and len(self.run_ids) < 2:
            raise ValueError("COMPARISON reports require at least two run_ids")


@dataclass(frozen=True, slots=True)
class TearsheetBundle:
    spec: ReportSpec
    out_dir: Path
    run_id: str
    display_name: str
    pages: dict[str, Path]
    tables: dict[str, pd.DataFrame]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonBundle:
    spec: ReportSpec
    out_dir: Path
    display_names: tuple[str, ...]
    pages: dict[str, Path]
    tables: dict[str, pd.DataFrame]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReportBundle:
    spec: ReportSpec
    out_dir: Path
    runs: tuple[SavedRun, ...]
    summary: pd.DataFrame
    appendix: pd.DataFrame
    plots: dict[str, Path]
    notes: tuple[str, ...] = ()
