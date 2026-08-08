"""Train all IDS models and select the best downstream artifact."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Callable, Iterable

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS, normalize_dataset_name
from soc_ready_ids.data.preprocessor import load_processed_arrays
from soc_ready_ids.models.autoencoder_ids import train_autoencoder
from soc_ready_ids.models.random_forest import train_random_forest
from soc_ready_ids.models.xgboost_ids import train_xgboost
from soc_ready_ids.utils.io import save_json
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)

REQUIRED_MODELS: tuple[str, ...] = (
    "random_forest",
    "xgboost_ids",
    "autoencoder_ids",
)


def _smoke_config(values: dict[str, Any]) -> dict[str, Any]:
    """Return a fast training configuration for local mock validation."""
    adjusted = copy.deepcopy(values)
    adjusted["models"]["random_forest"]["n_estimators"] = 80
    adjusted["models"]["xgboost"]["optuna_trials"] = 5
    adjusted["models"]["xgboost"]["n_estimators"] = 120
    adjusted["models"]["xgboost"]["timeout_seconds"] = 60
    adjusted["models"]["autoencoder"]["epochs"] = 5
    adjusted["models"]["autoencoder"]["batch_size"] = 64
    return adjusted


def train_all(
    config_path: str = "config.yaml",
    requested_models: list[str] | None = None,
    *,
    strict: bool = True,
    smoke: bool = False,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Train requested models, compare by macro F1, and save a manifest."""
    config = load_config(config_path)
    selected_dataset = normalize_dataset_name(
        dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(selected_dataset)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        feature_columns,
    ) = load_processed_arrays(config.path("paths.processed_data_dir"))
    class_names = list(label_encoder.classes_)
    model_dir = str(config.path("paths.model_dir"))
    metrics_dir = str(config.path("paths.metrics_dir"))
    requested = requested_models or list(REQUIRED_MODELS)
    config_values = _smoke_config(config.values) if smoke else config.values
    results: list[dict[str, Any]] = []

    trainers: dict[str, Callable[[], dict[str, Any]]] = {
        "random_forest": lambda: train_random_forest(
            X_train,
            y_train,
            X_test,
            y_test,
            class_names,
            feature_columns,
            config_values,
            model_dir,
            metrics_dir,
        )[1],
        "xgboost_ids": lambda: train_xgboost(
            X_train,
            y_train,
            X_test,
            y_test,
            class_names,
            feature_columns,
            config_values,
            model_dir,
            metrics_dir,
        )[1],
        "autoencoder_ids": lambda: train_autoencoder(
            X_train,
            y_train,
            X_test,
            y_test,
            class_names,
            feature_columns,
            config_values,
            model_dir,
            metrics_dir,
        ).metrics,
    }

    failures: list[str] = []
    for model_name in requested:
        try:
            metrics = trainers[model_name]()
            results.append(
                {
                    "model": model_name,
                    "metrics": metrics,
                    "artifact_path": str(Path(model_dir) / f"{model_name}.joblib"),
                }
            )
        except Exception as exc:
            LOGGER.exception("Training failed for %s: %s", model_name, exc)
            failures.append(f"{model_name}: {exc}")
            results.append(
                {
                    "model": model_name,
                    "error": str(exc),
                    "artifact_path": str(Path(model_dir) / f"{model_name}.joblib"),
                }
            )

    if strict and failures:
        save_json({"all_results": results}, Path(model_dir) / "training_failures.json")
        raise RuntimeError("Required model training failed: " + "; ".join(failures))

    metric_name = str(config.get("models.best_model_metric", "f1_macro"))
    successful = [
        item
        for item in results
        if "metrics" in item and item["metrics"].get(metric_name) is not None
    ]
    if not successful:
        raise RuntimeError(
            "No model trained successfully; check dependencies and processed data."
        )
    best = max(successful, key=lambda item: float(item["metrics"][metric_name]))
    manifest = {
        "best_model": best["model"],
        "selection_metric": metric_name,
        "selection_score": best["metrics"][metric_name],
        "artifact_path": best["artifact_path"],
        "all_results": results,
        "smoke_mode": smoke,
    }
    save_json(manifest, Path(model_dir) / "best_model.json")
    LOGGER.info(
        "Selected best model: %s (%s=%s)",
        manifest["best_model"],
        metric_name,
        manifest["selection_score"],
    )
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    """Train and compare all IDS models."""
    parser = argparse.ArgumentParser(description="Train and compare IDS models.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS))
    parser.add_argument(
        "--models",
        nargs="*",
        choices=list(REQUIRED_MODELS),
        help="Optional subset of models",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail the command if one model fails.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use fast hyperparameters for a local mock run.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    train_all(
        args.config,
        args.models,
        strict=not args.allow_partial,
        smoke=args.smoke,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
