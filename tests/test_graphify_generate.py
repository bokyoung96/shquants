import importlib.util
from pathlib import Path

import pandas as pd


def _load_generate_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "graphify-out" / "generate_graphify.py"
    spec = importlib.util.spec_from_file_location("graphify_generate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_output_paths_cleans_stale_generated_files(tmp_path) -> None:
    module = _load_generate_module()

    out = tmp_path / "graphify-out"
    obsidian = out / "obsidian"
    wiki = out / "wiki"
    obsidian.mkdir(parents=True)
    wiki.mkdir(parents=True)
    (obsidian / "qw__ksdq_v.csv.md").write_text("stale", encoding="utf-8")
    (wiki / "old.md").write_text("stale", encoding="utf-8")
    (out / "graph.svg").write_text("stale", encoding="utf-8")
    (out / "graph 2.html").write_text("stale", encoding="utf-8")
    (out / "summary 2.json").write_text("stale", encoding="utf-8")

    module.prepare_output_paths(out)

    assert not (obsidian / "qw__ksdq_v.csv.md").exists()
    assert not (wiki / "old.md").exists()
    assert not (out / "graph.svg").exists()
    assert not (out / "graph 2.html").exists()
    assert not (out / "summary 2.json").exists()
    assert obsidian.exists()
    assert wiki.exists()


def test_materialize_raw_reference_docs_writes_map_and_gics_outputs(tmp_path) -> None:
    module = _load_generate_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    with pd.ExcelWriter(raw_dir / "map.xlsx") as writer:
        pd.DataFrame({"Code": ["G10"], "Name": ["에너지"]}).to_excel(writer, sheet_name="sector_map", index=False)
        pd.DataFrame({"Ticker": ["A091990"], "Name": ["셀트리온헬스케어"]}).to_excel(writer, sheet_name="Sheet3", index=False)

    pd.DataFrame(
        {
            "DATE": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
            "TICKER": ["A091990", "A091990"],
            "GICS_SECTOR_LV1_NAME": ["Health Care", "Information Technology"],
            "GICS_SECTOR_LV2_NAME": ["Biotechnology", "Health Care Equipment & Services"],
        }
    ).to_excel(raw_dir / "snp_ksdq_gics_sector_big.xlsx", index=False)

    generated = module.materialize_raw_reference_docs(raw_dir)

    generated_names = {path.name for path in generated}
    assert "map_sector_codes.md" in generated_names
    assert "map_ticker_name_index.md" in generated_names
    assert "snp_ksdq_gics_sector_big_pivot.csv" in generated_names
    assert "snp_ksdq_gics_sector_latest.md" in generated_names
    assert "snp_ksdq_gics_sector_membership.md" in generated_names

    latest_doc = (raw_dir / "snp_ksdq_gics_sector_latest.md").read_text(encoding="utf-8")
    membership_doc = (raw_dir / "snp_ksdq_gics_sector_membership.md").read_text(encoding="utf-8")
    pivot = pd.read_csv(raw_dir / "snp_ksdq_gics_sector_big_pivot.csv")

    assert "KOSDAQ GICS Sector Latest Mapping" in latest_doc
    assert "셀트리온헬스케어" in latest_doc
    assert "Information Technology" in latest_doc
    assert "## Information Technology" in membership_doc
    assert list(pivot.columns) == ["date", "A091990"]


def test_materialize_raw_reference_docs_writes_lv1_and_lv2_gics_outputs_from_single_workbook(tmp_path) -> None:
    module = _load_generate_module()

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    with pd.ExcelWriter(raw_dir / "map.xlsx") as writer:
        pd.DataFrame({"Ticker": ["A091990"], "Name": ["셀트리온헬스케어"]}).to_excel(writer, sheet_name="Sheet3", index=False)

    pd.DataFrame(
        {
            "DATE": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")],
            "TICKER": ["A091990", "A091990"],
            "GICS_SECTOR_LV1_NAME": ["Health Care", "Information Technology"],
            "GICS_SECTOR_LV2_NAME": ["Biotechnology", "Healthcare Equipment"],
        }
    ).to_excel(raw_dir / "snp_ksdq_gics_sector_big.xlsx", index=False)

    generated = module.materialize_raw_reference_docs(raw_dir)

    generated_names = {path.name for path in generated}
    assert "snp_ksdq_gics_sector_big_lv1_pivot.csv" in generated_names
    assert "snp_ksdq_gics_sector_big_lv2_pivot.csv" in generated_names
    assert "snp_ksdq_gics_sector_latest_lv1.md" in generated_names
    assert "snp_ksdq_gics_sector_latest_lv2.md" in generated_names
    assert "snp_ksdq_gics_sector_membership_lv1.md" in generated_names
    assert "snp_ksdq_gics_sector_membership_lv2.md" in generated_names
    assert "snp_ksdq_gics_sector_big_pivot.csv" in generated_names
    assert "snp_ksdq_gics_sector_latest.md" in generated_names
    assert "snp_ksdq_gics_sector_membership.md" in generated_names

    latest_lv1 = (raw_dir / "snp_ksdq_gics_sector_latest_lv1.md").read_text(encoding="utf-8")
    latest_lv2 = (raw_dir / "snp_ksdq_gics_sector_latest_lv2.md").read_text(encoding="utf-8")
    pivot_lv1 = pd.read_csv(raw_dir / "snp_ksdq_gics_sector_big_lv1_pivot.csv")
    pivot_lv2 = pd.read_csv(raw_dir / "snp_ksdq_gics_sector_big_lv2_pivot.csv")
    compat_latest = (raw_dir / "snp_ksdq_gics_sector_latest.md").read_text(encoding="utf-8")

    assert "Information Technology" in latest_lv1
    assert "Healthcare Equipment" in latest_lv2
    assert list(pivot_lv1.columns) == ["date", "A091990"]
    assert list(pivot_lv2.columns) == ["date", "A091990"]
    assert "Information Technology" in compat_latest
