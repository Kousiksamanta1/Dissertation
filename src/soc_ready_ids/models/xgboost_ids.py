"""XGBoost IDS classifier tuned with Optuna."""

from __future__ import annotations

import argparse
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS, normalize_dataset_name
from soc_ready_ids.data.preprocessor import load_processed_arrays
from soc_ready_ids.models.common import evaluate_predictions, predict_proba_safe, save_model_artifact
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
    feature_columns: list[str],
    config_values: dict[str, Any],
    model_dir: str,
    metrics_dir: str,
) -> tuple[object, dict]:
    """Tune, train, evaluate, and save an XGBoost IDS model."""
    try:
        import optuna
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install optuna and xgboost to train the XGBoost model.") from exc

    random_state = int(config_values["project"].get("random_state", 42))
    xgb_cfg = config_values["models"]["xgboost"]
    n_classes = len(class_names)
    class_counts = np.bincount(y_train)
    stratify = y_train if class_counts.size > 1 and class_counts.min() >= 2 else None
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
    )

    objective_name = "multi:softprob" if n_classes > 2 else "binary:logistic"
    eval_metric = "mlogloss" if n_classes > 2 else "logloss"
    maximum_estimators = max(40, int(xgb_cfg.get("n_estimators", 400)))
    minimum_estimators = min(100, maximum_estimators)

    def objective(trial: optuna.Trial) -> float:
        """Return validation macro-F1 for one Optuna trial."""
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators", minimum_estimators, maximum_estimators
            ),
            "max_depth": trial.suggest_int("max_depth", 3, int(xgb_cfg.get("max_depth", 8))),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "objective": objective_name,
            "eval_metric": eval_metric,
            "random_state": random_state,
            "n_jobs": -1,
        }
        if n_classes > 2:
            params["num_class"] = n_classes
        model = XGBClassifier(**params)
        model.fit(X_fit, y_fit, verbose=False)
        predicted = model.predict(X_valid)
        return float(f1_score(y_valid, predicted, average="macro", zero_division=0))

    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_ids_f1_macro",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    timeout = xgb_cfg.get("timeout_seconds")
    study.optimize(objective, n_trials=int(xgb_cfg.get("optuna_trials", 50)), timeout=timeout)
    LOGGER.info("Best XGBoost params: %s", study.best_params)

    final_params = {
        **study.best_params,
        "objective": objective_name,
        "eval_metric": eval_metric,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if n_classes > 2:
        final_params["num_class"] = n_classes
    model = XGBClassifier(**final_params)
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    y_proba = predict_proba_safe(model, X_test)
    metrics = evaluate_predictions(y_test, y_pred, class_names, metrics_dir, "xgboost_ids", y_proba)
    save_model_artifact(
        model,
        "xgboost_ids",
        model_dir,
        class_names,
        feature_columns,
        metrics,
        {
            "explainability_type": "tree",
            "optuna_best_params": study.best_params,
            "background_data": X_train.head(200).to_numpy(dtype=float),
        },
    )
    return model, metrics


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for XGBoost training."""
    parser = argparse.ArgumentParser(description="Train Optuna-tuned XGBoost IDS model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    dataset = normalize_dataset_name(
        args.dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(dataset)
    X_train, X_test, y_train, y_test, label_encoder, feature_columns = load_processed_arrays(config.path("paths.processed_data_dir"))
    train_xgboost(
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
