from backtesting.catalog import DataCatalog, DatasetGroup, DatasetId


EXPECTED_RAW_STEMS = {
    "qw_adj_c",
    "qw_adj_h",
    "qw_adj_l",
    "qw_adj_o",
    "qw_asset_lfq0",
    "qw_BM",
    "qw_c",
    "qw_dividend_cash",
    "qw_dividend_cash_ttm",
    "qw_dps_ttm",
    "qw_eps_nfq1",
    "qw_eps_nfq2",
    "qw_eps_nfy1",
    "qw_op_fwd_12m",
    "qw_equity_lfq0",
    "qw_etf_adj_c",
    "qw_etf_adj_h",
    "qw_etf_adj_l",
    "qw_etf_adj_o",
    "qw_etf_adj_v",
    "qw_etf_aum",
    "qw_etf_div",
    "qw_etf_pdf_avg_ret",
    "qw_etf_pdf_v_value",
    "qw_etf_spread",
    "qw_etf_te",
    "qw_foreign",
    "qw_foreign_ratio",
    "qw_fcf",
    "qw_gp_lfq0",
    "qw_institution",
    "qw_int_bearing_liab_nfq0",
    "qw_ksdq150_yn",
    "qw_ksdq_adj_c",
    "qw_ksdq_adj_h",
    "qw_ksdq_adj_l",
    "qw_ksdq_adj_o",
    "qw_ksdq_mktcap",
    "qw_ksdq_mktcap_flt",
    "qw_ksdq_v",
    "qw_ksdq_wics_sec_big",
    "qw_k200_yn",
    "qw_liability_lfq0",
    "qw_mkt_typ",
    "qw_mktcap",
    "qw_mktcap_flt",
    "qw_ni_lfq0",
    "qw_ocf_lfq0",
    "qw_op_lfq0",
    "qw_op_nfq1",
    "qw_op_nfq2",
    "qw_op_nfy1",
    "qw_quick_assets_nfq0",
    "qw_retail",
    "qw_sha_out",
    "qw_tangible_assets_nfq0",
    "qw_trs_ban",
    "qw_v",
    "qw_v_value",
    "qw_wi_sec_26",
    "qw_wi_sec_26_big",
    "qw_wics_sec_big",
}


def test_catalog_groups_cover_known_datasets() -> None:
    catalog = DataCatalog.default()

    assert DatasetId.QW_ADJ_C in catalog.ids(DatasetGroup.PRICE)
    assert DatasetId.QW_ETF_ADJ_C in catalog.ids(DatasetGroup.PRICE)
    assert DatasetId.QW_KSDQ_ADJ_C in catalog.ids(DatasetGroup.PRICE)
    assert DatasetId.QW_KSDQ150_YN in catalog.ids(DatasetGroup.FLAG)
    assert DatasetId.QW_OP_FWD_12M in catalog.ids(DatasetGroup.ESTIMATE)
    assert DatasetId.QW_OP_NFY1 in catalog.ids(DatasetGroup.ESTIMATE)
    assert DatasetId.QW_FOREIGN in catalog.ids(DatasetGroup.FLOW)
    assert DatasetId.QW_K200_YN in catalog.ids(DatasetGroup.FLAG)
    assert DatasetId.QW_DPS_TTM in catalog.ids(DatasetGroup.FUNDAMENTAL)
    assert DatasetId.QW_DIVIDEND_CASH in catalog.ids(DatasetGroup.FUNDAMENTAL)
    assert DatasetId.QW_DIVIDEND_CASH_TTM in catalog.ids(DatasetGroup.FUNDAMENTAL)
    assert DatasetId.QW_FCF in catalog.ids(DatasetGroup.FUNDAMENTAL)
    assert DatasetId.QW_INT_BEARING_LIAB_NFQ0 in catalog.ids(DatasetGroup.FUNDAMENTAL)
    assert DatasetId.QW_QUICK_ASSETS_NFQ0 in catalog.ids(DatasetGroup.FUNDAMENTAL)
    assert DatasetId.QW_TANGIBLE_ASSETS_NFQ0 in catalog.ids(DatasetGroup.FUNDAMENTAL)
    assert DatasetId.QW_WI_SEC_26 in catalog.ids(DatasetGroup.META)
    assert DatasetId.QW_WI_SEC_26_BIG in catalog.ids(DatasetGroup.META)
    assert DatasetId.QW_WICS_SEC_BIG in catalog.ids(DatasetGroup.META)
    assert DatasetId.QW_KSDQ_WICS_SEC_BIG in catalog.ids(DatasetGroup.META)


def test_catalog_returns_grouped_specs() -> None:
    catalog = DataCatalog.default()

    specs = catalog.group(DatasetGroup.ESTIMATE)

    assert any(spec.id is DatasetId.QW_OP_NFY1 for spec in specs)
    assert all(spec.group is DatasetGroup.ESTIMATE for spec in specs)


def test_catalog_treats_forward_estimate_datasets_as_point_in_time() -> None:
    catalog = DataCatalog.default()
    estimate_ids = (
        DatasetId.QW_EPS_NFQ1,
        DatasetId.QW_EPS_NFQ2,
        DatasetId.QW_EPS_NFY1,
        DatasetId.QW_OP_NFQ1,
        DatasetId.QW_OP_NFQ2,
        DatasetId.QW_OP_NFY1,
        DatasetId.QW_OP_FWD_12M,
    )

    assert all(catalog.get(dataset_id).lag == 0 for dataset_id in estimate_ids)


def test_catalog_treats_wi26_sector_as_daily_observations() -> None:
    catalog = DataCatalog.default()

    spec = catalog.get(DatasetId.QW_WI_SEC_26)

    assert spec.freq == "D"
    assert spec.validity == "daily"
    assert spec.dtype == "string"


def test_catalog_treats_wi26_big_sector_as_daily_observations() -> None:
    catalog = DataCatalog.default()

    spec = catalog.get(DatasetId.QW_WI_SEC_26_BIG)

    assert spec.freq == "D"
    assert spec.validity == "daily"
    assert spec.dtype == "string"


def test_catalog_covers_all_stock_raw_stems() -> None:
    catalog = DataCatalog.default()

    assert {dataset_id.value for dataset_id in catalog.ids()} == EXPECTED_RAW_STEMS


def test_catalog_exposes_group_view() -> None:
    catalog = DataCatalog.default()
    groups = catalog.groups()

    assert DatasetId.QW_ADJ_C in groups.price
    assert DatasetId.QW_OP_FWD_12M in groups.estimate
    assert DatasetId.QW_OP_NFY1 in groups.estimate
    assert DatasetId.QW_DPS_TTM in groups.fundamental
    assert DatasetId.QW_DIVIDEND_CASH in groups.fundamental
    assert DatasetId.QW_DIVIDEND_CASH_TTM in groups.fundamental
    assert DatasetId.QW_EQUITY_LFQ0 in groups.fundamental
    assert DatasetId.QW_FCF in groups.fundamental
    assert DatasetId.QW_INT_BEARING_LIAB_NFQ0 in groups.fundamental
    assert DatasetId.QW_QUICK_ASSETS_NFQ0 in groups.fundamental
    assert DatasetId.QW_TANGIBLE_ASSETS_NFQ0 in groups.fundamental
    assert groups.get(DatasetGroup.FLAG) == groups.flag
