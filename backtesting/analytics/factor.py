from __future__ import annotations

import numpy as np
import pandas as pd


def quantile_returns(
    signal: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    q: int = 5,
) -> pd.DataFrame:
    cols = [f"q{i}" for i in range(1, q + 1)]
    rows: list[pd.Series] = []
    idx: list[pd.Timestamp | object] = []
    common_idx = [ts for ts in signal.index if ts in fwd_returns.index]
    common_cols = [col for col in signal.columns if col in fwd_returns.columns]

    if not common_cols:
        return pd.DataFrame(columns=cols, dtype=float)

    for ts in common_idx:
        sig = signal.loc[ts, common_cols]
        fut = fwd_returns.loc[ts, common_cols]
        valid = sig.notna() & fut.notna()
        row = pd.Series(np.nan, index=cols, dtype=float)
        if valid.any():
            frame = pd.DataFrame({"signal": sig[valid], "fwd": fut[valid]})
            if len(frame) == 1 or frame["signal"].nunique(dropna=True) == 1:
                row["q1"] = float(frame["fwd"].mean())
            else:
                try:
                    bins = pd.qcut(
                        frame["signal"],
                        q=min(q, len(frame)),
                        labels=False,
                        duplicates="drop",
                    )
                except ValueError:
                    bins = pd.Series(0, index=frame.index)

                if isinstance(bins, pd.Series):
                    bins = bins.dropna().astype(int)
                    if bins.empty:
                        row["q1"] = float(frame["fwd"].mean())
                    else:
                        grp = frame.loc[bins.index].groupby(bins)["fwd"].mean()
                        for bucket, value in grp.items():
                            col = f"q{int(bucket) + 1}"
                            if col in row.index:
                                row[col] = float(value)
        rows.append(row)
        idx.append(ts)

    if not rows:
        return pd.DataFrame(columns=cols, dtype=float)

    return pd.DataFrame(rows, index=idx, columns=cols, dtype=float)


def rank_ic(signal: pd.DataFrame, fwd_returns: pd.DataFrame) -> pd.Series:
    vals: list[float] = []
    idx: list[pd.Timestamp | object] = []
    common_idx = [ts for ts in signal.index if ts in fwd_returns.index]
    common_cols = [col for col in signal.columns if col in fwd_returns.columns]

    if not common_cols:
        return pd.Series(dtype=float)

    for ts in common_idx:
        sig = signal.loc[ts, common_cols]
        fut = fwd_returns.loc[ts, common_cols]
        valid = sig.notna() & fut.notna()
        if valid.sum() < 2:
            vals.append(np.nan)
        else:
            pair = pd.DataFrame({"signal": sig[valid], "fwd": fut[valid]})
            if pair["signal"].nunique(dropna=True) < 2 or pair["fwd"].nunique(dropna=True) < 2:
                vals.append(np.nan)
            else:
                vals.append(float(pair["signal"].corr(pair["fwd"], method="spearman")))
        idx.append(ts)

    return pd.Series(vals, index=idx, dtype=float)
