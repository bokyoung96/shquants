from pathlib import Path

from root import ROOT


def test_root_exposes_repo_paths():
    checkout_root = Path(__file__).resolve().parents[2]

    assert ROOT.root == checkout_root
    assert ROOT.config_path == ROOT.root / "config" / "config.json"
    assert ROOT.kis_path == ROOT.root / "kis"
    assert ROOT.raw_path == ROOT.root / "raw"
    assert ROOT.parquet_path == ROOT.root / "parquet"
    assert ROOT.kis_path.is_dir()
