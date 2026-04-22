from pathlib import Path

import pandas as pd


def find_raw_path(raw_dir: Path, stem: str) -> Path:
    # Prefer CSV when both raw exports exist for the same dataset stem.
    for suffix in (".csv", ".xlsx"):
        direct = raw_dir / f"{stem}{suffix}"
        if direct.exists():
            return direct

    matches: list[Path] = []
    for suffix in (".csv", ".xlsx"):
        matches.extend(sorted(raw_dir.rglob(f"{stem}{suffix}")))
    if matches:
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"ambiguous raw dataset: {stem} -> {', '.join(str(path) for path in matches)}")

    raise FileNotFoundError(f"missing raw dataset: {stem}")


def read_raw_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        encodings = ("utf-8", "utf-8-sig", "cp949")
        last_error: UnicodeDecodeError | None = None
        for encoding in encodings:
            try:
                return pd.read_csv(path, encoding=encoding, low_memory=False)
            except UnicodeDecodeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise UnicodeDecodeError("utf-8", b"", 0, 1, f"unable to decode csv file: {path}")
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"unsupported raw dataset format: {path.suffix}")
