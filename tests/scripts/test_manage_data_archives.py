import zipfile

import pytest

from scripts.manage_data_archives import archive_path, pack, restore, verify


def seed(root, name="raw/prices.csv", content=b"date,price\n2026-01-01,100\n"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_remove_restore_round_trip(tmp_path):
    source = seed(tmp_path)
    original = source.read_bytes()
    pack(tmp_path, "raw", remove_originals=True)
    assert not source.exists()
    assert verify(tmp_path, "raw")["raw/prices.csv"]["size"] == len(original)
    restore(tmp_path, "raw")
    assert source.read_bytes() == original
    restore(tmp_path, "raw")  # Identical destinations are safe.


def test_pack_retains_originals_by_default(tmp_path):
    source = seed(tmp_path)
    pack(tmp_path, "raw")
    assert source.exists()


def test_repack_preserves_archived_files_and_updates_restored_subset(tmp_path):
    seed(tmp_path, "raw/a.csv", b"old")
    seed(tmp_path, "raw/b.csv", b"retained")
    pack(tmp_path, "raw", remove_originals=True)
    seed(tmp_path, "raw/a.csv", b"new")
    pack(tmp_path, "raw", remove_originals=True)
    restore(tmp_path, "raw")
    assert (tmp_path / "raw/a.csv").read_bytes() == b"new"
    assert (tmp_path / "raw/b.csv").read_bytes() == b"retained"


def test_restore_refuses_changed_files(tmp_path):
    source = seed(tmp_path)
    pack(tmp_path, "raw")
    source.write_bytes(b"local edits")
    with pytest.raises(FileExistsError):
        restore(tmp_path, "raw")
    assert source.read_bytes() == b"local edits"


def test_corrupt_archive_does_not_remove_sources(tmp_path):
    source = seed(tmp_path)
    pack(tmp_path, "raw")
    with zipfile.ZipFile(archive_path(tmp_path, "raw"), "a") as archive:
        archive.writestr("raw/unlisted.csv", "corrupt")
    with pytest.raises(ValueError):
        pack(tmp_path, "raw", remove_originals=True)
    assert source.exists()


def test_restore_rejects_symlink_parent(tmp_path):
    source = seed(tmp_path)
    pack(tmp_path, "raw", remove_originals=True)
    source.parent.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    source.parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        restore(tmp_path, "raw")
    assert not list(outside.iterdir())
