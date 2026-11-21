"""Pack local datasets into verified ZIP64 archives; restore before research runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("raw", "parquet", "results", "etfs/output", "etc/data")
MANIFEST = "__manifest__.json"
CHUNK = 1024 * 1024


def digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(CHUNK), b""):
        result.update(chunk)
    return result.hexdigest()


def archive_path(root, group):
    return root / "data_archives" / (group.replace("/", "__") + ".zip")


def read_manifest(archive, group):
    manifest = json.loads(archive.read(MANIFEST))
    if manifest.get("version") != 1 or manifest.get("group") != group:
        raise ValueError("Invalid archive manifest")
    entries = manifest["files"]
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != set(entries) | {MANIFEST}:
        raise ValueError("Archive members differ from manifest")
    for name in entries:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or not name.startswith(group + "/"):
            raise ValueError(f"Unsafe archive member: {name}")
    return entries


def verify(root, group, path=None):
    path = path or archive_path(root, group)
    with zipfile.ZipFile(path) as archive:
        entries = read_manifest(archive, group)
        for name, record in entries.items():
            if archive.getinfo(name).file_size != record["size"]:
                raise ValueError(f"Size mismatch: {name}")
            with archive.open(name) as stream:
                if digest(stream) != record["sha256"]:
                    raise ValueError(f"Checksum mismatch: {name}")
    return entries


def pack(root, group, remove_originals=False):
    folder = root / group
    sources = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.exists() else []
    if folder.is_symlink() or any(p.is_symlink() for p in folder.rglob("*")):
        raise ValueError(f"Symlinks are not supported: {group}")
    destination = archive_path(root, group)
    if not sources:
        if destination.exists():
            entries = verify(root, group)
            print(f"{group}: existing archive verified ({len(entries)} files)", flush=True)
        else:
            print(f"{group}: no files", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = verify(root, group) if destination.exists() else {}
    source_names = {p.relative_to(root).as_posix() for p in sources}
    fd, temporary = tempfile.mkstemp(suffix=".zip.partial", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary)
    entries = {}
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            # Keep previously archived files when only a subset has been restored.
            if previous:
                with zipfile.ZipFile(destination) as old:
                    for name, record in previous.items():
                        if name not in source_names:
                            with old.open(name) as src, archive.open(name, "w", force_zip64=True) as dst:
                                shutil.copyfileobj(src, dst, CHUNK)
                            entries[name] = record
            for source in sources:
                name = source.relative_to(root).as_posix()
                before = source.stat()
                checksum = hashlib.sha256()
                with source.open("rb") as src, archive.open(name, "w", force_zip64=True) as dst:
                    for chunk in iter(lambda: src.read(CHUNK), b""):
                        dst.write(chunk)
                        checksum.update(chunk)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Source changed while packing: {source}")
                entries[name] = {"size": after.st_size, "sha256": checksum.hexdigest(), "mtime_ns": after.st_mtime_ns}
            archive.writestr(MANIFEST, json.dumps({"version": 1, "group": group, "files": entries}, ensure_ascii=False, indent=2))
        verify(root, group, temporary)
        os.replace(temporary, destination)
        if remove_originals:
            # Recheck current contents, so concurrent edits are never discarded.
            for source in sources:
                name = source.relative_to(root).as_posix()
                with source.open("rb") as stream:
                    if digest(stream) != entries[name]["sha256"]:
                        raise RuntimeError(f"Source changed; retained original: {source}")
                source.unlink()
            for directory in sorted(folder.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()
        total = sum(record["size"] for record in entries.values())
        print(f"{group}: verified {len(entries)} files, {total:,} -> {destination.stat().st_size:,} bytes; originals {'removed' if remove_originals else 'retained'}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def restore(root, group):
    entries = verify(root, group)
    # Preflight all destinations before writing any file.
    for name, record in entries.items():
        target = root / name
        if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root.parent):
            raise ValueError(f"Symlink destination: {name}")
        if target.exists():
            if not target.is_file():
                raise FileExistsError(target)
            with target.open("rb") as stream:
                if digest(stream) != record["sha256"]:
                    raise FileExistsError(f"Refusing to overwrite changed file: {name}")
    with zipfile.ZipFile(archive_path(root, group)) as archive:
        for name, record in entries.items():
            target = root / name
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".restore.partial")
            try:
                with os.fdopen(fd, "wb") as dst, archive.open(name) as src:
                    shutil.copyfileobj(src, dst, CHUNK)
                # Exclusive creation: never overwrite a concurrently created file.
                os.link(temporary, target)
                os.utime(target, ns=(record["mtime_ns"], record["mtime_ns"]))
            finally:
                Path(temporary).unlink(missing_ok=True)
    print(f"{group}: restored {len(entries)} files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "verify", "restore"))
    parser.add_argument("groups", nargs="*", help="Dataset groups; default: all")
    parser.add_argument("--remove-originals", action="store_true", help="Remove source files only after ZIP/SHA256 verification")
    args = parser.parse_args()
    groups = args.groups or GROUPS
    if any(group not in GROUPS for group in groups):
        parser.error(f"Allowed groups: {', '.join(GROUPS)}")
    if args.remove_originals and args.command != "pack":
        parser.error("--remove-originals requires pack")
    for group in groups:
        if args.command == "pack":
            pack(ROOT, group, args.remove_originals)
        elif args.command == "restore":
            restore(ROOT, group)
        else:
            entries = verify(ROOT, group)
            print(f"{group}: verified {len(entries)} files", flush=True)


if __name__ == "__main__":
    main()
