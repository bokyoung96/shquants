from dataclasses import dataclass

import pandas as pd

from backtesting.catalog import DataCatalog, DatasetId, DatasetSpec
from backtesting.data.policy import expand_monthly_frame
from backtesting.data.store import ParquetStore


@dataclass(frozen=True, slots=True)
class LoadRequest:
    datasets: list[DatasetId]
    start: str
    end: str
    universe: pd.DataFrame | None = None
    benchmark: pd.Series | None = None
    universe_id: str | None = None
    price_mode: str = "adj"


@dataclass(slots=True)
class MarketData:
    frames: dict[str, pd.DataFrame]
    universe: pd.DataFrame | None
    benchmark: pd.Series | None


class DataLoader:
    FRAME_KEYS = {
        DatasetId.QW_ADJ_C: "close",
        DatasetId.QW_ADJ_O: "open",
        DatasetId.QW_ADJ_H: "high",
        DatasetId.QW_ADJ_L: "low",
        DatasetId.QW_ASSET_LFQ0: "asset",
        DatasetId.QW_BM: "benchmark",
        DatasetId.QW_C: "close_raw",
        DatasetId.QW_EPS_NFQ1: "eps_fwd_q1",
        DatasetId.QW_EPS_NFQ2: "eps_fwd_q2",
        DatasetId.QW_EPS_NFY1: "eps_fwd",
        DatasetId.QW_EQUITY_LFQ0: "equity",
        DatasetId.QW_ETF_ADJ_C: "close",
        DatasetId.QW_ETF_ADJ_O: "open",
        DatasetId.QW_ETF_ADJ_H: "high",
        DatasetId.QW_ETF_ADJ_L: "low",
        DatasetId.QW_ETF_ADJ_V: "volume",
        DatasetId.QW_ETF_AUM: "aum",
        DatasetId.QW_ETF_DIV: "dividend",
        DatasetId.QW_ETF_PDF_AVG_RET: "pdf_avg_ret",
        DatasetId.QW_ETF_PDF_V_VALUE: "pdf_v_value",
        DatasetId.QW_ETF_SPREAD: "spread",
        DatasetId.QW_ETF_TE: "tracking_error",
        DatasetId.QW_FOREIGN: "foreign_flow",
        DatasetId.QW_FOREIGN_RATIO: "foreign_ratio",
        DatasetId.QW_GP_LFQ0: "gross_profit",
        DatasetId.QW_INSTITUTION: "inst_flow",
        DatasetId.QW_MKTCAP: "market_cap",
        DatasetId.QW_K200_YN: "k200_yn",
        DatasetId.QW_KSDQ_ADJ_C: "close",
        DatasetId.QW_KSDQ_ADJ_O: "open",
        DatasetId.QW_KSDQ_ADJ_H: "high",
        DatasetId.QW_KSDQ_ADJ_L: "low",
        DatasetId.QW_KSDQ_V: "volume",
        DatasetId.QW_KSDQ_MKTCAP: "market_cap",
        DatasetId.QW_KSDQ_MKTCAP_FLT: "float_market_cap",
        DatasetId.QW_KSDQ150_YN: "universe_membership",
        DatasetId.QW_KSDQ_WICS_SEC_BIG: "sector_big",
        DatasetId.QW_LIABILITY_LFQ0: "liability",
        DatasetId.QW_MKTCAP_FLT: "float_market_cap",
        DatasetId.QW_MKT_TYP: "market_type",
        DatasetId.QW_NI_LFQ0: "net_income",
        DatasetId.QW_OCF_LFQ0: "oper_cash_flow",
        DatasetId.QW_OP_LFQ0: "op",
        DatasetId.QW_OP_NFQ1: "op_fwd_q1",
        DatasetId.QW_OP_NFQ2: "op_fwd_q2",
        DatasetId.QW_OP_NFY1: "op_fwd",
        DatasetId.QW_RETAIL: "retail_flow",
        DatasetId.QW_SHA_OUT: "shares_out",
        DatasetId.QW_TRS_BAN: "trade_ban",
        DatasetId.QW_V: "volume",
        DatasetId.QW_WICS_SEC_BIG: "sector_big",
    }

    def __init__(self, catalog: DataCatalog, store: ParquetStore) -> None:
        self.catalog = catalog
        self.store = store

    def load(self, request: LoadRequest) -> MarketData:
        if request.price_mode != "adj":
            raise ValueError(f"unsupported price_mode: {request.price_mode}")

        frames: dict[str, pd.DataFrame] = {}
        for dataset_id in request.datasets:
            spec = self.catalog.get(dataset_id)
            frame = self._load_frame(spec, request)
            key = self.FRAME_KEYS.get(dataset_id, spec.stem)
            if key in frames:
                raise ValueError(f"duplicate semantic frame key: {key}")
            frames[key] = frame
        return MarketData(
            frames=frames,
            universe=request.universe,
            benchmark=request.benchmark,
        )

    def _load_frame(self, spec: DatasetSpec, request: LoadRequest) -> pd.DataFrame:
        frame = self.store.read(spec.stem)
        if spec.validity == "daily":
            return frame.loc[request.start : request.end]
        if spec.validity == "month_only":
            start = pd.Timestamp(request.start).to_period("M").start_time
            end = pd.Timestamp(request.end).to_period("M").end_time.normalize()
            frame = frame.loc[start:end]
            calendar = pd.date_range(request.start, request.end, freq="D")
            return expand_monthly_frame(frame=frame, calendar=calendar, validity=spec.validity)
        raise ValueError(f"unsupported validity: {spec.validity}")
