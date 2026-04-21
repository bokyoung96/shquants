import pandas as pd


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "date" not in normalized.columns:
        first = normalized.columns[0]
        if first == "" or str(first).startswith("Unnamed:"):
            normalized = normalized.rename(columns={first: "date"})
        else:
            raise KeyError("missing date column in raw dataset")
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized = normalized.sort_values("date")

    if normalized["date"].duplicated().any():
        raise ValueError("duplicate date values in raw dataset")

    return normalized.set_index("date")
