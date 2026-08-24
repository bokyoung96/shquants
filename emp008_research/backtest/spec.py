from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CostModel:
    fee: float = 0.0
    sell_tax: float = 0.0
    slippage: float = 0.0
    borrow_fee_annual: float = 0.0
    short_cash_collateral_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class BacktestSpec:
    name: str
    data_dir: Path
    output_dir: Path
    weights_csv: Path
    start: str | None = None
    end: str | None = None
    capital: float = 100_000_000.0
    fill_mode: str = "close"
    allow_fractional: bool = True
    close_filename: str = "qw_adj_c.parquet"
    open_filename: str = "qw_adj_o.parquet"
    tradable_filename: str | None = None
    exit_tradable_filename: str | None = None
    benchmark_weights_filename: str = "qw_bm_weights.parquet"
    fee: float = 0.0
    sell_tax: float = 0.0
    slippage: float = 0.0
    borrow_fee_annual: float = 0.0
    short_cash_collateral_ratio: float = 1.0

    def __post_init__(self) -> None:
        if self.fill_mode not in {"close", "next_open"}:
            raise ValueError("fill_mode must be 'close' or 'next_open'")
        if self.capital <= 0.0:
            raise ValueError("capital must be positive")

    @property
    def cost(self) -> CostModel:
        return CostModel(
            fee=self.fee,
            sell_tax=self.sell_tax,
            slippage=self.slippage,
            borrow_fee_annual=self.borrow_fee_annual,
            short_cash_collateral_ratio=self.short_cash_collateral_ratio,
        )

    def frame_path(self, filename: str | None) -> Path | None:
        if filename is None:
            return None
        return self.data_dir / filename

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["weights_csv"] = str(self.weights_csv)
        return payload
