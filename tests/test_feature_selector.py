"""Tests for Phase 1 feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from soc_ready_ids.data.feature_selector import (
    sample_for_feature_selection,
    save_feature_selection,
    select_features,
)


def _feature_frame(rows: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    """Create a small deterministic classification matrix."""
    rng = np.random.default_rng(42)
    y = np.array([0, 1] * (rows // 2))
    X = pd.DataFrame(
        {
            "packet_rate": y + rng.normal(0.0, 0.05, size=rows),
            "byte_rate": y * 2 + rng.normal(0.0, 0.05, size=rows),
            "duration": rng.normal(1.0, 0.1, size=rows),
            "flag_count": rng.integers(0, 3, size=rows),
        }
    )
    return X, y


def test_sample_for_feature_selection_caps_rows() -> None:
    """Large training sets should be sampled deterministically."""
    X, y = _feature_frame(rows=40)

    sampled_X, sampled_y = sample_for_feature_selection(
        X, y, max_rows=20, random_state=42
    )

    assert sampled_X.shape[0] == 20
    assert sampled_y.shape[0] == 20
    assert set(sampled_y) == {0, 1}


def test_select_features_combines_mi_and_rfe() -> None:
    """Feature ranking should mark exactly top_k selected features."""
    X, y = _feature_frame(rows=40)

    ranking = select_features(
        X,
        y,
        top_k=2,
        random_state=42,
        n_estimators=20,
        rfe_step=0.5,
    )

    assert ranking["selected"].sum() == 2
    assert {"feature", "mutual_information", "rfe_rank", "combined_rank"}.issubset(
        ranking.columns
    )
    assert set(ranking.loc[ranking["selected"], "feature"]).issubset(X.columns)


def test_save_feature_selection_writes_artifacts(tmp_path) -> None:
    """Selected feature outputs should be written for downstream phases."""
    X, y = _feature_frame(rows=40)
    ranking = select_features(
        X,
        y,
        top_k=2,
        random_state=42,
        n_estimators=20,
        rfe_step=0.5,
    )

    paths = save_feature_selection(
        ranking,
        tmp_path / "figures",
        tmp_path / "processed",
        X_train=X,
        X_test=X.head(5),
        top_k=2,
    )

    for path in paths.values():
        assert path.exists()
    selected = pd.read_csv(paths["X_train_selected"])
    assert selected.shape[1] == 2
