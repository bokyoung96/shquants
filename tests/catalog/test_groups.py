from backtesting.catalog import DataCatalog, DatasetGroup, DatasetId


EXPECTED_RAW_STEMS = {
    "qw_adj_c",
    "qw_adj_h",
    "qw_adj_l",
    "qw_adj_o",
    "qw_asset_lfq0",
    "qw_BM",
    "qw_c",
    "qw_eps_nfq1",
    "qw_eps_nfq2",
    "qw_eps_nfy1",
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
    "qw_gp_lfq0",
    "qw_institution",
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
    "qw_retail",
    "qw_sha_out",
    "qw_trs_ban",
    "qw_v",
    "qw_v_value",
    "qw_wics_sec_big",
}


def test_catalog_groups_cover_known_datasets() -> None:
    catalog = DataCatalog.default()

    assert DatasetId.QW_ADJ_C in catalog.ids(DatasetGroup.PRICE)
    assert DatasetId.QW_ETF_ADJ_C in catalog.ids(DatasetGroup.PRICE)
    assert DatasetId.QW_KSDQ_ADJ_C in catalog.ids(DatasetGroup.PRICE)
    assert DatasetId.QW_KSDQ150_YN in catalog.ids(DatasetGroup.FLAG)
    assert DatasetId.QW_OP_NFY1 in catalog.ids(DatasetGroup.ESTIMATE)
    assert DatasetId.QW_FOREIGN in catalog.ids(DatasetGroup.FLOW)
    assert DatasetId.QW_K200_YN in catalog.ids(DatasetGroup.FLAG)
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
    )

    assert all(catalog.get(dataset_id).lag == 0 for dataset_id in estimate_ids)


def test_catalog_covers_all_stock_raw_stems() -> None:
    catalog = DataCatalog.default()

    assert {dataset_id.value for dataset_id in catalog.ids()} == EXPECTED_RAW_STEMS


def test_catalog_exposes_group_view() -> None:
    catalog = DataCatalog.default()
    groups = catalog.groups()

    assert DatasetId.QW_ADJ_C in groups.price
    assert DatasetId.QW_OP_NFY1 in groups.estimate
    assert DatasetId.QW_EQUITY_LFQ0 in groups.fundamental
    assert groups.get(DatasetGroup.FLAG) == groups.flag
