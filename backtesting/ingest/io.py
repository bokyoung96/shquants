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
        return pd.read_csv(path)
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"unsupported raw dataset format: {path.suffix}")
