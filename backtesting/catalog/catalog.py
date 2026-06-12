from dataclasses import dataclass

from .enums import DatasetGroup, DatasetId
from .groups import DatasetGroups
from .specs import DatasetSpec


FORWARD_ESTIMATE_LAG_DAYS = 0


def _spec(
    dataset_id: DatasetId,
    *,
    group: DatasetGroup,
    freq: str,
    kind: str,
    fill: str = "none",
    validity: str | None = None,
    lag: int = 0,
    dtype: str = "float64",
    axis: str = "date_symbol",
) -> DatasetSpec:
    return DatasetSpec(
        id=dataset_id,
        stem=dataset_id.value,
        group=group,
        freq=freq,
        kind=kind,
        fill=fill,
        validity=validity or ("month_only" if freq == "M" else "daily"),
        lag=lag,
        dtype=dtype,
        axis=axis,
    )


@dataclass(slots=True)
class DataCatalog:
    specs: dict[DatasetId, DatasetSpec]

    @classmethod
    def default(cls) -> "DataCatalog":
        specs = {
            DatasetId.QW_ADJ_C: _spec(DatasetId.QW_ADJ_C, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_ADJ_O: _spec(DatasetId.QW_ADJ_O, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_ADJ_H: _spec(DatasetId.QW_ADJ_H, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_ADJ_L: _spec(DatasetId.QW_ADJ_L, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_ASSET_LFQ0: _spec(DatasetId.QW_ASSET_LFQ0, group=DatasetGroup.FUNDAMENTAL, freq="M", kind="asset"),
            DatasetId.QW_BM: _spec(
                DatasetId.QW_BM,
                group=DatasetGroup.BENCHMARK,
                freq="D",
                kind="benchmark",
                axis="date_code_field",
            ),
            DatasetId.QW_BM_WEIGHTS: _spec(
                DatasetId.QW_BM_WEIGHTS,
                group=DatasetGroup.BENCHMARK,
                freq="D",
                kind="benchmark_weights",
            ),
            DatasetId.QW_C: _spec(DatasetId.QW_C, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_DIVIDEND_CASH: _spec(
                DatasetId.QW_DIVIDEND_CASH,
                group=DatasetGroup.FUNDAMENTAL,
                freq="D",
                kind="dividend_cash",
            ),
            DatasetId.QW_DIVIDEND_CASH_TTM: _spec(
                DatasetId.QW_DIVIDEND_CASH_TTM,
                group=DatasetGroup.FUNDAMENTAL,
                freq="D",
                kind="dividend_cash_ttm",
            ),
            DatasetId.QW_DIVIDEND_YLD_FY0: _spec(
                DatasetId.QW_DIVIDEND_YLD_FY0,
                group=DatasetGroup.FUNDAMENTAL,
                freq="D",
                kind="dividend_yld_fy0",
            ),
            DatasetId.QW_DPS_TTM: _spec(DatasetId.QW_DPS_TTM, group=DatasetGroup.FUNDAMENTAL, freq="D", kind="dps_ttm"),
            DatasetId.QW_EPS_NFQ1: _spec(
                DatasetId.QW_EPS_NFQ1,
                group=DatasetGroup.ESTIMATE,
                freq="M",
                kind="estimate",
                lag=FORWARD_ESTIMATE_LAG_DAYS,
            ),
            DatasetId.QW_EPS_NFQ2: _spec(
                DatasetId.QW_EPS_NFQ2,
                group=DatasetGroup.ESTIMATE,
                freq="M",
                kind="estimate",
                lag=FORWARD_ESTIMATE_LAG_DAYS,
            ),
            DatasetId.QW_EPS_NFY1: _spec(
                DatasetId.QW_EPS_NFY1,
                group=DatasetGroup.ESTIMATE,
                freq="M",
                kind="estimate",
                lag=FORWARD_ESTIMATE_LAG_DAYS,
            ),
            DatasetId.QW_EQUITY_LFQ0: _spec(DatasetId.QW_EQUITY_LFQ0, group=DatasetGroup.FUNDAMENTAL, freq="M", kind="equity"),
            DatasetId.QW_ETF_ADJ_C: _spec(DatasetId.QW_ETF_ADJ_C, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_ETF_ADJ_O: _spec(DatasetId.QW_ETF_ADJ_O, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_ETF_ADJ_H: _spec(DatasetId.QW_ETF_ADJ_H, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_ETF_ADJ_L: _spec(DatasetId.QW_ETF_ADJ_L, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_ETF_ADJ_V: _spec(DatasetId.QW_ETF_ADJ_V, group=DatasetGroup.PRICE, freq="D", kind="volume", fill="zero"),
            DatasetId.QW_ETF_AUM: _spec(DatasetId.QW_ETF_AUM, group=DatasetGroup.PRICE, freq="D", kind="aum"),
            DatasetId.QW_ETF_DIV: _spec(DatasetId.QW_ETF_DIV, group=DatasetGroup.PRICE, freq="D", kind="dividend"),
            DatasetId.QW_ETF_PDF_AVG_RET: _spec(DatasetId.QW_ETF_PDF_AVG_RET, group=DatasetGroup.PRICE, freq="D", kind="pdf_avg_ret"),
            DatasetId.QW_ETF_PDF_V_VALUE: _spec(DatasetId.QW_ETF_PDF_V_VALUE, group=DatasetGroup.PRICE, freq="D", kind="pdf_v_value"),
            DatasetId.QW_ETF_SPREAD: _spec(DatasetId.QW_ETF_SPREAD, group=DatasetGroup.PRICE, freq="D", kind="spread"),
            DatasetId.QW_ETF_TE: _spec(DatasetId.QW_ETF_TE, group=DatasetGroup.PRICE, freq="D", kind="tracking_error"),
            DatasetId.QW_FCF: _spec(DatasetId.QW_FCF, group=DatasetGroup.FUNDAMENTAL, freq="M", kind="free_cash_flow"),
            DatasetId.QW_FOREIGN: _spec(DatasetId.QW_FOREIGN, group=DatasetGroup.FLOW, freq="D", kind="flow"),
            DatasetId.QW_FOREIGN_RATIO: _spec(DatasetId.QW_FOREIGN_RATIO, group=DatasetGroup.FLOW, freq="D", kind="ratio"),
            DatasetId.QW_GP_LFQ0: _spec(DatasetId.QW_GP_LFQ0, group=DatasetGroup.FUNDAMENTAL, freq="M", kind="gross_profit"),
            DatasetId.QW_INSTITUTION: _spec(DatasetId.QW_INSTITUTION, group=DatasetGroup.FLOW, freq="D", kind="flow"),
            DatasetId.QW_INT_BEARING_LIAB_NFQ0: _spec(
                DatasetId.QW_INT_BEARING_LIAB_NFQ0,
                group=DatasetGroup.FUNDAMENTAL,
                freq="M",
                kind="interest_bearing_liability",
            ),
            DatasetId.QW_K200_YN: _spec(DatasetId.QW_K200_YN, group=DatasetGroup.FLAG, freq="D", kind="flag", fill="zero", dtype="int64"),
            DatasetId.QW_KSDQ_ADJ_C: _spec(DatasetId.QW_KSDQ_ADJ_C, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_KSDQ_ADJ_O: _spec(DatasetId.QW_KSDQ_ADJ_O, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_KSDQ_ADJ_H: _spec(DatasetId.QW_KSDQ_ADJ_H, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_KSDQ_ADJ_L: _spec(DatasetId.QW_KSDQ_ADJ_L, group=DatasetGroup.PRICE, freq="D", kind="price"),
            DatasetId.QW_KSDQ_V: _spec(DatasetId.QW_KSDQ_V, group=DatasetGroup.PRICE, freq="D", kind="volume", fill="zero"),
            DatasetId.QW_KSDQ_MKTCAP: _spec(DatasetId.QW_KSDQ_MKTCAP, group=DatasetGroup.PRICE, freq="D", kind="market_cap"),
            DatasetId.QW_KSDQ_MKTCAP_FLT: _spec(
                DatasetId.QW_KSDQ_MKTCAP_FLT,
                group=DatasetGroup.PRICE,
                freq="D",
                kind="float_market_cap",
            ),
            DatasetId.QW_KSDQ150_YN: _spec(
                DatasetId.QW_KSDQ150_YN,
                group=DatasetGroup.FLAG,
                freq="D",
                kind="flag",
                fill="zero",
                dtype="int64",
            ),
            DatasetId.QW_KSDQ_WICS_SEC_BIG: _spec(
                DatasetId.QW_KSDQ_WICS_SEC_BIG,
                group=DatasetGroup.META,
                freq="M",
                kind="sector",
                dtype="string",
            ),
            DatasetId.QW_LIABILITY_LFQ0: _spec(DatasetId.QW_LIABILITY_LFQ0, group=DatasetGroup.FUNDAMENTAL, freq="M", kind="liability"),
            DatasetId.QW_MKTCAP: _spec(DatasetId.QW_MKTCAP, group=DatasetGroup.PRICE, freq="D", kind="market_cap"),
            DatasetId.QW_MKTCAP_FLT: _spec(DatasetId.QW_MKTCAP_FLT, group=DatasetGroup.PRICE, freq="D", kind="float_market_cap"),
            DatasetId.QW_MKT_TYP: _spec(DatasetId.QW_MKT_TYP, group=DatasetGroup.META, freq="M", kind="market_type", dtype="string"),
            DatasetId.QW_NI_LFQ0: _spec(DatasetId.QW_NI_LFQ0, group=DatasetGroup.FUNDAMENTAL, freq="M", kind="net_income"),
            DatasetId.QW_OCF_LFQ0: _spec(DatasetId.QW_OCF_LFQ0, group=DatasetGroup.FUNDAMENTAL, freq="M", kind="oper_cash_flow"),
            DatasetId.QW_OP_LFQ0: _spec(DatasetId.QW_OP_LFQ0, group=DatasetGroup.FUNDAMENTAL, freq="M", kind="oper_profit"),
            DatasetId.QW_OP_FWD_12M: _spec(
                DatasetId.QW_OP_FWD_12M,
                group=DatasetGroup.ESTIMATE,
                freq="M",
                kind="estimate",
                lag=FORWARD_ESTIMATE_LAG_DAYS,
            ),
            DatasetId.QW_OP_NFQ1: _spec(
                DatasetId.QW_OP_NFQ1,
                group=DatasetGroup.ESTIMATE,
                freq="M",
                kind="estimate",
                lag=FORWARD_ESTIMATE_LAG_DAYS,
            ),
            DatasetId.QW_OP_NFQ2: _spec(
                DatasetId.QW_OP_NFQ2,
                group=DatasetGroup.ESTIMATE,
                freq="M",
                kind="estimate",
                lag=FORWARD_ESTIMATE_LAG_DAYS,
            ),
            DatasetId.QW_OP_NFY1: _spec(
                DatasetId.QW_OP_NFY1,
                group=DatasetGroup.ESTIMATE,
                freq="M",
                kind="estimate",
                lag=FORWARD_ESTIMATE_LAG_DAYS,
            ),
            DatasetId.QW_QUICK_ASSETS_NFQ0: _spec(
                DatasetId.QW_QUICK_ASSETS_NFQ0,
                group=DatasetGroup.FUNDAMENTAL,
                freq="M",
                kind="quick_asset",
            ),
            DatasetId.QW_RETAIL: _spec(DatasetId.QW_RETAIL, group=DatasetGroup.FLOW, freq="D", kind="flow"),
            DatasetId.QW_SHA_OUT: _spec(DatasetId.QW_SHA_OUT, group=DatasetGroup.FUNDAMENTAL, freq="M", kind="shares_out"),
            DatasetId.QW_TANGIBLE_ASSETS_NFQ0: _spec(
                DatasetId.QW_TANGIBLE_ASSETS_NFQ0,
                group=DatasetGroup.FUNDAMENTAL,
                freq="M",
                kind="tangible_asset",
            ),
            DatasetId.QW_TRS_BAN: _spec(DatasetId.QW_TRS_BAN, group=DatasetGroup.FLAG, freq="D", kind="flag", fill="zero", dtype="int64"),
            DatasetId.QW_V: _spec(DatasetId.QW_V, group=DatasetGroup.PRICE, freq="D", kind="volume", fill="zero"),
            DatasetId.QW_V_VALUE: _spec(DatasetId.QW_V_VALUE, group=DatasetGroup.PRICE, freq="D", kind="trading_value"),
            DatasetId.QW_WI_SEC_26: _spec(DatasetId.QW_WI_SEC_26, group=DatasetGroup.META, freq="D", kind="sector", dtype="string"),
            DatasetId.QW_WI_SEC_26_BIG: _spec(DatasetId.QW_WI_SEC_26_BIG, group=DatasetGroup.META, freq="D", kind="sector", dtype="string"),
            DatasetId.QW_WICS_SEC_BIG: _spec(DatasetId.QW_WICS_SEC_BIG, group=DatasetGroup.META, freq="M", kind="sector", dtype="string"),
        }
        return cls(specs=specs)

    def get(self, dataset_id: DatasetId) -> DatasetSpec:
        return self.specs[dataset_id]

    def ids(self, group: DatasetGroup | None = None) -> tuple[DatasetId, ...]:
        if group is None:
            return tuple(self.specs)
        return tuple(dataset_id for dataset_id, spec in self.specs.items() if spec.group is group)

    def group(self, group: DatasetGroup) -> tuple[DatasetSpec, ...]:
        return tuple(spec for spec in self.specs.values() if spec.group is group)

    def groups(self) -> DatasetGroups:
        return DatasetGroups(
            price=self.ids(DatasetGroup.PRICE),
            fundamental=self.ids(DatasetGroup.FUNDAMENTAL),
            estimate=self.ids(DatasetGroup.ESTIMATE),
            flow=self.ids(DatasetGroup.FLOW),
            flag=self.ids(DatasetGroup.FLAG),
            meta=self.ids(DatasetGroup.META),
            benchmark=self.ids(DatasetGroup.BENCHMARK),
        )
