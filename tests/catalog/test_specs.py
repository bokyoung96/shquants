from backtesting.catalog import DataCatalog, DatasetId


def test_catalog_returns_known_spec():
    catalog = DataCatalog.default()
    spec = catalog.get(DatasetId.QW_ADJ_C)

    assert spec.id is DatasetId.QW_ADJ_C
    assert spec.stem == "qw_adj_c"
    assert spec.freq == "D"
    assert spec.kind == "price"
    assert spec.axis == "date_symbol"
