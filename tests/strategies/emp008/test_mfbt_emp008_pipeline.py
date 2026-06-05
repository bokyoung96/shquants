import pandas as pd

from backtesting.catalog import DatasetId
from backtesting.strategies.emp008.mfbt_emp008_data import MfbtEmp008Config, required_datasets


def test_mfbt_emp008_config_defaults_to_wi26_big_sector_and_bm_weights() -> None:
    config = MfbtEmp008Config()

    assert config.sector_dataset is DatasetId.QW_WI_SEC_26_BIG
    assert config.bm_weights_dataset is DatasetId.QW_BM_WEIGHTS
    assert config.universe_dataset is DatasetId.QW_K200_YN
    assert config.float_market_cap_dataset is DatasetId.QW_MKTCAP_FLT
    assert pd.Timedelta(days=config.retail_flow_lookback_days) == pd.Timedelta(days=252)


def test_required_datasets_include_all_mfbt_emp008_inputs() -> None:
    datasets = set(required_datasets(MfbtEmp008Config()))

    assert DatasetId.QW_ADJ_C in datasets
    assert DatasetId.QW_C in datasets
    assert DatasetId.QW_BM_WEIGHTS in datasets
    assert DatasetId.QW_OP_FWD_12M in datasets
    assert DatasetId.QW_DPS_TTM in datasets
    assert DatasetId.QW_RETAIL in datasets
    assert DatasetId.QW_WI_SEC_26_BIG in datasets
    assert DatasetId.QW_MKTCAP in datasets
    assert DatasetId.QW_MKTCAP_FLT in datasets
    assert DatasetId.QW_FCF in datasets
    assert DatasetId.QW_INT_BEARING_LIAB_NFQ0 in datasets
    assert DatasetId.QW_QUICK_ASSETS_NFQ0 in datasets
    assert DatasetId.QW_K200_YN in datasets
