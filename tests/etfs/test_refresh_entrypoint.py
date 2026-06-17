import shutil
from types import SimpleNamespace

from etfs.refresh import refresh


def test_refresh_entrypoint_defaults_to_etfs_workbooks_and_filtered_tickers() -> None:
    args = refresh.build_parser().parse_args([])

    assert args.ticker_workbook == "etfs/refresh/ticker.xlsx"
    assert args.template == "etfs/refresh/pdf.xlsx"
    assert args.output_dir == "etfs/output/files"
    assert args.work_dir == "etfs/refresh/work"
    assert args.manifest == ""
    assert args.limit is None
    assert args.chunk_size == 25
    assert args.all_rows is False
    assert args.force is False
    assert args.keep_work is False
    assert args.mode == "batch"


def test_remove_default_work_dir_retries_transient_excel_lock(tmp_path, monkeypatch) -> None:
    refresh_dir = tmp_path / "refresh"
    work_dir = refresh_dir / "work"
    work_dir.mkdir(parents=True)
    calls: list[object] = []
    real_rmtree = shutil.rmtree

    def fake_rmtree(path):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("locked workbook")
        real_rmtree(path)

    monkeypatch.setattr(refresh.paths, "REFRESH_DIR", refresh_dir)
    monkeypatch.setattr(refresh.shutil, "rmtree", fake_rmtree)
    monkeypatch.setattr(refresh, "time", SimpleNamespace(sleep=lambda _seconds: None), raising=False)

    refresh._remove_default_work_dir(work_dir)

    assert calls == [work_dir.resolve(), work_dir.resolve()]
    assert not work_dir.exists()
