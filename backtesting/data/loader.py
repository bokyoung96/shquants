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
        DatasetId.QW_BM_WEIGHTS: "bm_weights",
        DatasetId.QW_C: "close_raw",
        DatasetId.QW_DIVIDEND_CASH: "dividend_cash",
        DatasetId.QW_DIVIDEND_CASH_TTM: "dividend_cash_ttm",
        DatasetId.QW_DIVIDEND_YLD_FY0: "dividend_yld_fy0",
        DatasetId.QW_DPS_TTM: "dps_ttm",
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
        DatasetId.QW_FCF: "free_cash_flow",
        DatasetId.QW_FOREIGN: "foreign_flow",
        DatasetId.QW_FOREIGN_RATIO: "foreign_ratio",
        DatasetId.QW_GP_LFQ0: "gross_profit",
        DatasetId.QW_INSTITUTION: "inst_flow",
        DatasetId.QW_INT_BEARING_LIAB_NFQ0: "interest_bearing_liability",
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
        DatasetId.QW_OP_FWD_12M: "op_fwd_12m",
        DatasetId.QW_OP_NFQ1: "op_fwd_q1",
        DatasetId.QW_OP_NFQ2: "op_fwd_q2",
        DatasetId.QW_OP_NFY1: "op_fwd",
        DatasetId.QW_QUICK_ASSETS_NFQ0: "quick_asset",
        DatasetId.QW_RETAIL: "retail_flow",
        DatasetId.QW_SHA_OUT: "shares_out",
        DatasetId.QW_TANGIBLE_ASSETS_NFQ0: "tangible_asset",
        DatasetId.QW_TRS_BAN: "trade_ban",
        DatasetId.QW_V: "volume",
        DatasetId.QW_V_VALUE: "trading_value",
        DatasetId.QW_WI_SEC_26: "sector_big",
        DatasetId.QW_WI_SEC_26_BIG: "sector_big",
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
            start = self._lagged_start(request.start, spec.lag)
            frame = frame.loc[start : request.end]
            return self._apply_lag(frame, spec.lag).loc[request.start : request.end]
        if spec.validity == "month_only":
            load_start = self._lagged_start(request.start, spec.lag)
            start = pd.Timestamp(load_start).to_period("M").start_time
            end = pd.Timestamp(request.end).to_period("M").end_time.normalize()
            frame = frame.loc[start:end]
            load_calendar = pd.date_range(load_start, request.end, freq="D")
            expanded = expand_monthly_frame(frame=frame, calendar=load_calendar, validity=spec.validity)
            return self._apply_lag(expanded, spec.lag).loc[request.start : request.end]
        raise ValueError(f"unsupported validity: {spec.validity}")

    @staticmethod
    def _lagged_start(start: str, lag: int) -> pd.Timestamp:
        return pd.Timestamp(start) - pd.Timedelta(days=max(int(lag), 0))

    @staticmethod
    def _apply_lag(frame: pd.DataFrame, lag: int) -> pd.DataFrame:
        if lag <= 0:
            return frame
        lagged = frame.copy()
        lagged.index = pd.DatetimeIndex(lagged.index) + pd.Timedelta(days=int(lag))
        return lagged
