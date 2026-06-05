from backtesting.catalog import DataCatalog, DatasetId


def test_catalog_returns_known_spec():
    catalog = DataCatalog.default()
    spec = catalog.get(DatasetId.QW_ADJ_C)

    assert spec.id is DatasetId.QW_ADJ_C
    assert spec.stem == "qw_adj_c"
    assert spec.freq == "D"
    assert spec.kind == "price"
    assert spec.axis == "date_symbol"


def test_catalog_marks_quantwise_benchmark_as_date_code_field():
    catalog = DataCatalog.default()
    spec = catalog.get(DatasetId.QW_BM)

    assert spec.id is DatasetId.QW_BM
    assert spec.stem == "qw_BM"
    assert spec.kind == "benchmark"
    assert spec.axis == "date_code_field"
