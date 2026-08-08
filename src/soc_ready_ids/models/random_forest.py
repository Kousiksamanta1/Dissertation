"""Random forest IDS classifier with optional SMOTE balancing."""

from __future__ import annotations

import argparse
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS, normalize_dataset_name
from soc_ready_ids.data.preprocessor import load_processed_arrays
from soc_ready_ids.models.common import evaluate_predictions, predict_proba_safe, save_model_artifact
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def maybe_apply_smote(X: pd.DataFrame, y: np.ndarray, random_state: int, enabled: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """Balance minority classes with SMOTE when feasible."""
    if not enabled:
        return X, y
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2 or counts.min() < 2:
        LOGGER.warning("Skipping SMOTE because at least one class has fewer than two samples.")
        return X, y
    try:
        from imblearn.over_sampling import SMOTE

        k_neighbors = min(5, int(counts.min()) - 1)
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        LOGGER.info("Applied SMOTE: %s -> %s", X.shape, X_resampled.shape)
        return pd.DataFrame(X_resampled, columns=X.columns), np.asarray(y_resampled)
    except Exception as exc:  # pragma: no cover - optional dependency / data dependent
        LOGGER.warning("SMOTE unavailable or failed; continuing without it: %s", exc)
        return X, y


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
    feature_columns: list[str],
    config_values: dict[str, Any],
    model_dir: str,
    metrics_dir: str,
) -> tuple[RandomForestClassifier, dict]:
    """Train, evaluate, and save a random forest IDS model."""
    random_state = int(config_values["project"].get("random_state", 42))
    rf_cfg = config_values["models"]["random_forest"]
    X_fit, y_fit = maybe_apply_smote(X_train, y_train, random_state, bool(rf_cfg.get("use_smote", True)))

    model = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 300)),
        max_depth=rf_cfg.get("max_depth"),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
        n_jobs=int(rf_cfg.get("n_jobs", -1)),
        random_state=random_state,
        class_weight="balanced_subsample",
    )
    model.fit(X_fit, y_fit)
    y_pred = model.predict(X_test)
    y_proba = predict_proba_safe(model, X_test)
    metrics = evaluate_predictions(y_test, y_pred, class_names, metrics_dir, "random_forest", y_proba)
    save_model_artifact(
        model,
        "random_forest",
        model_dir,
        class_names,
        feature_columns,
        metrics,
        {
            "explainability_type": "tree",
            "background_data": X_train.head(200).to_numpy(dtype=float),
        },
    )
    return model, metrics


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for random forest training."""
    parser = argparse.ArgumentParser(description="Train RandomForest IDS model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    dataset = normalize_dataset_name(
        args.dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(dataset)
    X_train, X_test, y_train, y_test, label_encoder, feature_columns = load_processed_arrays(config.path("paths.processed_data_dir"))
    train_random_forest(
        X_train,
        y_train,
        X_test,
        y_test,
        list(label_encoder.classes_),
        feature_columns,
        config.values,
        str(config.path("paths.model_dir")),
        str(config.path("paths.metrics_dir")),
    )


if __name__ == "__main__":
    main()
