from dataclasses import dataclass
from enum import Enum


class DatasetId(str, Enum):
    QW_ADJ_C = "qw_adj_c"
    QW_ADJ_O = "qw_adj_o"
    QW_ADJ_H = "qw_adj_h"
    QW_ADJ_L = "qw_adj_l"
    QW_BM_WEIGHTS = "qw_bm_weights"
    QW_C = "qw_c"
    QW_DIVIDEND_YLD_FY0 = "qw_dividend_yld_fy0"
    QW_DPS_TTM = "qw_dps_ttm"
    QW_FCF = "qw_fcf"
    QW_INT_BEARING_LIAB_NFQ0 = "qw_int_bearing_liab_nfq0"
    QW_K200_YN = "qw_k200_yn"
    QW_MKTCAP = "qw_mktcap"
    QW_MKTCAP_FLT = "qw_mktcap_flt"
    QW_OP_FWD_12M = "qw_op_fwd_12m"
    QW_QUICK_ASSETS_NFQ0 = "qw_quick_assets_nfq0"
    QW_RETAIL = "qw_retail"
    QW_WI_SEC_26_BIG = "qw_wi_sec_26_big"
    QW_WICS_SEC_BIG = "qw_wics_sec_big"


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    id: DatasetId
    stem: str
    freq: str
    kind: str
    validity: str
    lag: int = 0


@dataclass(frozen=True, slots=True)
class DataCatalog:
    specs: dict[DatasetId, DatasetSpec]

    @classmethod
    def default(cls) -> "DataCatalog":
        monthly = {DatasetId.QW_FCF, DatasetId.QW_INT_BEARING_LIAB_NFQ0, DatasetId.QW_QUICK_ASSETS_NFQ0, DatasetId.QW_WICS_SEC_BIG}
        kinds = {
            DatasetId.QW_ADJ_C: "price", DatasetId.QW_ADJ_O: "price", DatasetId.QW_ADJ_H: "price", DatasetId.QW_ADJ_L: "price",
            DatasetId.QW_BM_WEIGHTS: "benchmark_weights", DatasetId.QW_C: "price", DatasetId.QW_DIVIDEND_YLD_FY0: "dividend_yld_fy0",
            DatasetId.QW_DPS_TTM: "dps_ttm", DatasetId.QW_FCF: "free_cash_flow", DatasetId.QW_INT_BEARING_LIAB_NFQ0: "interest_bearing_liability",
            DatasetId.QW_K200_YN: "flag", DatasetId.QW_MKTCAP: "market_cap", DatasetId.QW_MKTCAP_FLT: "float_market_cap",
            DatasetId.QW_OP_FWD_12M: "estimate", DatasetId.QW_QUICK_ASSETS_NFQ0: "quick_asset", DatasetId.QW_RETAIL: "flow",
            DatasetId.QW_WI_SEC_26_BIG: "sector", DatasetId.QW_WICS_SEC_BIG: "sector",
        }
        return cls({d: DatasetSpec(d, d.value, "M" if d in monthly else "D", k, "month_only" if d in monthly else "daily") for d, k in kinds.items()})

    def get(self, dataset_id: DatasetId) -> DatasetSpec:
        return self.specs[DatasetId(dataset_id)]

    def ids(self) -> tuple[DatasetId, ...]:
        return tuple(self.specs)
