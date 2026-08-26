from pathlib import Path

from data.catalog import DatasetId
from scripts.run_all_models import resolve_sector_run


def test_resolve_sector_run_uses_wi26_default_dataset_and_folder(tmp_path: Path) -> None:
    dataset, results_dir = resolve_sector_run(tmp_path, "wi26")

    assert dataset is None
    assert results_dir == tmp_path / "results" / "WI26" / "all_models"


def test_resolve_sector_run_uses_wics_dataset_and_folder(tmp_path: Path) -> None:
    dataset, results_dir = resolve_sector_run(tmp_path, "wics")

    assert dataset is DatasetId.QW_WICS_SEC_BIG
    assert results_dir == tmp_path / "results" / "WICS" / "all_models"

