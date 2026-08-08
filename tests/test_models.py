"""Tests for Phase 2 model training and common artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from soc_ready_ids.models.autoencoder_ids import (
    AutoencoderIDSModel,
    train_autoencoder,
)
from soc_ready_ids.models.common import (
    evaluate_predictions,
    load_best_model,
    predict_proba_safe,
)
from soc_ready_ids.models.random_forest import (
    maybe_apply_smote,
    train_random_forest,
)
from soc_ready_ids.models.xgboost_ids import train_xgboost


def _classification_data(
    rows: int = 48,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Create deterministic three-class train/test data."""
    rng = np.random.default_rng(42)
    y = np.array([0, 1, 2] * (rows // 3))
    X = pd.DataFrame(
        {
            "f1": y + rng.normal(0, 0.1, size=rows),
            "f2": (y == 1).astype(float) + rng.normal(0, 0.1, size=rows),
            "f3": (y == 2).astype(float) + rng.normal(0, 0.1, size=rows),
            "f4": rng.normal(0, 1, size=rows),
        }
    )
    return X.iloc[:36], y[:36], X.iloc[36:], y[36:]


def _config_values() -> dict:
    """Return fast model settings for tests."""
    return {
        "project": {"random_state": 42},
        "models": {
            "random_forest": {
                "n_estimators": 20,
                "max_depth": 5,
                "min_samples_leaf": 1,
                "n_jobs": 1,
                "use_smote": True,
            },
            "xgboost": {
                "optuna_trials": 1,
                "timeout_seconds": 20,
                "n_estimators": 40,
                "max_depth": 3,
            },
            "autoencoder": {
                "hidden_dim": 8,
                "latent_dim": 3,
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.01,
                "threshold_percentile": 90,
            },
        },
    }


def test_evaluate_predictions_saves_metrics_and_figures(tmp_path: Path) -> None:
    """Metric evaluation should persist tables, predictions, PNG, and PDF."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.4, 0.5, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.1, 0.8],
            [0.6, 0.1, 0.3],
        ]
    )

    metrics = evaluate_predictions(
        y_true,
        y_pred,
        ["BENIGN", "DDoS", "PortScan"],
        tmp_path,
        "toy",
        probabilities,
    )

    assert metrics["f1_macro"] is not None
    assert metrics["binary_attack_f1_macro"] is not None
    assert (tmp_path / "toy_metrics.json").exists()
    assert (tmp_path / "toy_predictions.csv").exists()
    assert (tmp_path / "toy_confusion_matrix.pdf").exists()
    assert (tmp_path / "toy_detection_fpr_curve.pdf").exists()


def test_train_random_forest_and_load_best(tmp_path: Path) -> None:
    """Random forest artifacts should load through the best-model manifest."""
    X_train, y_train, X_test, y_test = _classification_data()
    model_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    model, metrics = train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        ["BENIGN", "DDoS", "PortScan"],
        list(X_train.columns),
        _config_values(),
        str(model_dir),
        str(metrics_dir),
    )
    manifest = {
        "artifact_path": str(model_dir / "random_forest.joblib")
    }
    (model_dir / "best_model.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    loaded = load_best_model(model_dir)

    assert metrics["accuracy"] is not None
    assert predict_proba_safe(model, X_test).shape == (12, 3)
    assert loaded["model_name"] == "random_forest"


def test_smote_skip_and_autoencoder_interface(tmp_path: Path) -> None:
    """SMOTE edge cases and autoencoder inference should remain loadable."""
    X = pd.DataFrame({"f1": [0.0, 1.0], "f2": [1.0, 2.0]})
    y = np.array([0, 1])
    unchanged_X, unchanged_y = maybe_apply_smote(X, y, 42, True)
    assert unchanged_X.equals(X)
    assert np.array_equal(unchanged_y, y)

    X_train, y_train, X_test, y_test = _classification_data()
    result = train_autoencoder(
        X_train,
        y_train,
        X_test,
        y_test,
        ["BENIGN", "DDoS", "PortScan"],
        list(X_train.columns),
        _config_values(),
        str(tmp_path / "models"),
        str(tmp_path / "metrics"),
    )
    probability = result.model.predict_proba(X_test)

    assert isinstance(result.model, AutoencoderIDSModel)
    assert probability.shape == (12, 2)
    assert result.model.predict(X_test).shape == (12,)
    assert result.model.deep_explainer_module() is not None


def test_train_xgboost_fast(tmp_path: Path) -> None:
    """One Optuna trial should produce a complete XGBoost artifact."""
    X_train, y_train, X_test, y_test = _classification_data()
    model, metrics = train_xgboost(
        X_train,
        y_train,
        X_test,
        y_test,
        ["BENIGN", "DDoS", "PortScan"],
        list(X_train.columns),
        _config_values(),
        str(tmp_path / "models"),
        str(tmp_path / "metrics"),
    )

    assert model.predict(X_test).shape == (12,)
    assert metrics["f1_macro"] is not None
    assert (tmp_path / "models" / "xgboost_ids.joblib").exists()
