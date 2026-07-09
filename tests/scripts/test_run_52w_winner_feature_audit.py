from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_52w_winner_feature_audit import label_winner_buckets, summarize_numeric_features


def test_label_winner_buckets_marks_top_and_bottom_groups() -> None:
    frame = pd.DataFrame({"ret20": [0.10, 0.08, 0.04, 0.01, -0.02, -0.05]})

    labeled = label_winner_buckets(frame, target_column="ret20", top_fraction=0.33, bottom_fraction=0.50)

    assert labeled["is_top_winner"].tolist() == [True, True, False, False, False, False]
    assert labeled["is_bottom_group"].tolist() == [False, False, False, True, True, True]


def test_summarize_numeric_features_reports_directional_differences() -> None:
    frame = pd.DataFrame(
        {
            "feature": [10.0, 8.0, 1.0, 0.0],
            "ret20": [0.20, 0.15, -0.05, -0.10],
            "is_top_winner": [True, True, False, False],
            "is_bottom_group": [False, False, True, True],
        }
    )

    summary = summarize_numeric_features(frame, ["feature"], target_column="ret20")

    row = summary.iloc[0]
    assert row["feature"] == "feature"
    assert row["top_winner_mean"] == pytest.approx(9.0)
    assert row["rest_mean"] == pytest.approx(0.5)
    assert row["bottom_group_mean"] == pytest.approx(0.5)
    assert row["top_minus_rest"] == pytest.approx(8.5)
    assert row["spearman_corr"] > 0.0
