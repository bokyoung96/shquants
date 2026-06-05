from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .mfbt_emp008_data import MfbtEmp008Config, load_mfbt_emp008_bm_weights


@dataclass(frozen=True, slots=True)
class MfbtEmp008Result:
    target_weights: pd.DataFrame
    active_weights: pd.DataFrame
    diagnostics: pd.DataFrame

    def weights_for_export(self) -> pd.DataFrame:
        return self.target_weights.T

    def write_outputs(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.target_weights.to_parquet(output_dir / "target_weights.parquet", engine="pyarrow")
        self.active_weights.to_parquet(output_dir / "active_weights.parquet", engine="pyarrow")
        self.diagnostics.to_parquet(output_dir / "diagnostics.parquet", engine="pyarrow")
        with pd.ExcelWriter(output_dir / "weights_export.xlsx", engine="openpyxl") as writer:
            self.weights_for_export().to_excel(writer, sheet_name="weights_ticker_by_date")
            self.diagnostics.to_excel(writer, sheet_name="summary", index=False)
            self.active_weights.T.to_excel(writer, sheet_name="active_ticker_by_date")


def run_mfbt_emp008_smoke(*, parquet_dir: Path, start: str, end: str) -> MfbtEmp008Result:
    config = MfbtEmp008Config()
    bm = load_mfbt_emp008_bm_weights(parquet_dir=parquet_dir, start=start, end=end, config=config).astype(float).copy()
    row_sum = bm.sum(axis=1)
    usable = row_sum.gt(0.0)
    if not usable.any():
        raise ValueError("no usable benchmark weight rows")
    bm = bm.loc[usable].div(row_sum.loc[usable], axis=0)
    diagnostics = pd.DataFrame(
        {
            "target_date": bm.index,
            "success": True,
            "sum_final_weight": bm.sum(axis=1).to_numpy(),
            "n_active_positions": bm.gt(0.0).sum(axis=1).to_numpy(),
        }
    )
    return MfbtEmp008Result(target_weights=bm, active_weights=bm * 0.0, diagnostics=diagnostics)
