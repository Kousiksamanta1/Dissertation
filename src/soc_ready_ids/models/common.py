"""Shared model evaluation and artifact utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from soc_ready_ids.utils.io import (
    ensure_dir,
    load_joblib,
    load_json,
    save_joblib,
    save_json,
)
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _safe_float(value: float | np.floating[Any]) -> float | None:
    """Convert metric values to JSON-safe floats."""
    number = float(value)
    return None if np.isnan(number) or np.isinf(number) else number


def predict_proba_safe(
    model: Any, X: pd.DataFrame | np.ndarray
) -> np.ndarray | None:
    """Return probability output when an estimator supports it."""
    if hasattr(model, "predict_proba"):
        try:
            return np.asarray(model.predict_proba(X), dtype=float)
        except Exception as exc:  # pragma: no cover - estimator dependent
            LOGGER.warning("Could not compute probabilities: %s", exc)
    return None


def _save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    y_proba: np.ndarray | None,
) -> Path:
    """Persist row-level predictions for reproducible downstream evaluation."""
    frame = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [
                class_names[int(index)] if int(index) < len(class_names) else str(index)
                for index in y_true
            ],
            "predicted_label": [
                class_names[int(index)] if int(index) < len(class_names) else str(index)
                for index in y_pred
            ],
        }
    )
    if y_proba is not None and y_proba.ndim == 2:
        for index, class_name in enumerate(class_names):
            if index < y_proba.shape[1]:
                safe_name = (
                    str(class_name)
                    .strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                )
                frame[f"probability_{safe_name}"] = y_proba[:, index]
    frame.to_csv(output_path, index=False)
    return output_path


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_dir: str | Path,
    model_name: str,
    y_proba: np.ndarray | None = None,
    task: str = "multiclass",
) -> dict[str, Any]:
    """Compute and persist standard IDS evaluation metrics and figures."""
    labels = list(range(len(class_names)))
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    directory = ensure_dir(output_dir)

    report = classification_report(
        y_true_array,
        y_pred_array,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "model": model_name,
        "task": task,
        "samples": int(len(y_true_array)),
        "accuracy": _safe_float(accuracy_score(y_true_array, y_pred_array)),
        "precision_macro": _safe_float(
            precision_score(
                y_true_array, y_pred_array, average="macro", zero_division=0
            )
        ),
        "recall_macro": _safe_float(
            recall_score(
                y_true_array, y_pred_array, average="macro", zero_division=0
            )
        ),
        "f1_macro": _safe_float(
            f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)
        ),
        "precision_weighted": _safe_float(
            precision_score(
                y_true_array, y_pred_array, average="weighted", zero_division=0
            )
        ),
        "recall_weighted": _safe_float(
            recall_score(
                y_true_array, y_pred_array, average="weighted", zero_division=0
            )
        ),
        "f1_weighted": _safe_float(
            f1_score(
                y_true_array, y_pred_array, average="weighted", zero_division=0
            )
        ),
        "classification_report": report,
        "roc_auc_per_class": {},
        "roc_auc_macro_ovr": None,
    }

    benign_indices = [
        index
        for index, name in enumerate(class_names)
        if str(name).upper() in {"BENIGN", "NORMAL"}
    ]
    benign_index = benign_indices[0] if benign_indices else 0
    y_true_binary = (y_true_array != benign_index).astype(int)
    y_pred_binary = (y_pred_array != benign_index).astype(int)
    metrics["binary_attack_precision_macro"] = _safe_float(
        precision_score(
            y_true_binary, y_pred_binary, average="macro", zero_division=0
        )
    )
    metrics["binary_attack_recall_macro"] = _safe_float(
        recall_score(
            y_true_binary, y_pred_binary, average="macro", zero_division=0
        )
    )
    metrics["binary_attack_f1_macro"] = _safe_float(
        f1_score(
            y_true_binary, y_pred_binary, average="macro", zero_division=0
        )
    )
    metrics["binary_attack_roc_auc"] = None

    if (
        y_proba is not None
        and y_proba.ndim == 2
        and y_proba.shape[1] == len(labels)
    ):
        y_bin = label_binarize(y_true_array, classes=labels)
        if len(labels) == 2 and y_bin.shape[1] == 1:
            y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])
        for index, class_name in enumerate(class_names):
            try:
                metrics["roc_auc_per_class"][class_name] = _safe_float(
                    roc_auc_score(y_bin[:, index], y_proba[:, index])
                )
            except ValueError:
                metrics["roc_auc_per_class"][class_name] = None
        try:
            metrics["roc_auc_macro_ovr"] = _safe_float(
                roc_auc_score(
                    y_bin, y_proba, average="macro", multi_class="ovr"
                )
            )
        except ValueError:
            metrics["roc_auc_macro_ovr"] = None
        try:
            attack_probability = 1.0 - y_proba[:, benign_index]
            metrics["binary_attack_roc_auc"] = _safe_float(
                roc_auc_score(y_true_binary, attack_probability)
            )
        except ValueError:
            metrics["binary_attack_roc_auc"] = None

    save_json(metrics, directory / f"{model_name}_metrics.json")
    _save_predictions(
        y_true_array,
        y_pred_array,
        class_names,
        directory / f"{model_name}_predictions.csv",
        y_proba,
    )
    save_confusion_matrix(
        y_true_array,
        y_pred_array,
        class_names,
        directory / f"{model_name}_confusion_matrix.png",
    )
    save_detection_fpr_curve(
        y_true_array,
        y_pred_array,
        class_names,
        directory / f"{model_name}_detection_fpr_curve.png",
        y_proba,
    )
    LOGGER.info(
        "%s f1_macro=%s accuracy=%s",
        model_name,
        metrics["f1_macro"],
        metrics["accuracy"],
    )
    return metrics


def _save_png_and_pdf(figure: plt.Figure, output_path: str | Path) -> Path:
    """Save a publication-ready figure as PNG and PDF."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    return path


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Save a confusion-matrix heatmap as PNG and PDF."""
    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )
    figure, axis = plt.subplots(
        figsize=(
            max(7, len(class_names) * 0.8),
            max(5, len(class_names) * 0.6),
        )
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    return _save_png_and_pdf(figure, output_path)


def save_detection_fpr_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    y_proba: np.ndarray | None = None,
) -> Path:
    """Save detection-rate versus false-positive-rate curves."""
    labels = list(range(len(class_names)))
    figure, axis = plt.subplots(figsize=(8, 6))
    if (
        y_proba is not None
        and y_proba.ndim == 2
        and y_proba.shape[1] == len(labels)
    ):
        y_bin = label_binarize(y_true, classes=labels)
        if len(labels) == 2 and y_bin.shape[1] == 1:
            y_bin = np.column_stack([1 - y_bin[:, 0], y_bin[:, 0]])
        for index, class_name in enumerate(class_names):
            try:
                false_positive_rate, true_positive_rate, _ = roc_curve(
                    y_bin[:, index], y_proba[:, index]
                )
                axis.plot(
                    false_positive_rate,
                    true_positive_rate,
                    label=class_name,
                )
            except ValueError:
                continue
    else:
        rows: list[dict[str, float | str]] = []
        for label, class_name in zip(labels, class_names):
            actual_positive = y_true == label
            predicted_positive = y_pred == label
            true_positive = int(
                np.logical_and(actual_positive, predicted_positive).sum()
            )
            false_positive = int(
                np.logical_and(~actual_positive, predicted_positive).sum()
            )
            false_negative = int(
                np.logical_and(actual_positive, ~predicted_positive).sum()
            )
            true_negative = int(
                np.logical_and(~actual_positive, ~predicted_positive).sum()
            )
            detection_rate = (
                true_positive / (true_positive + false_negative)
                if (true_positive + false_negative)
                else 0.0
            )
            false_positive_rate = (
                false_positive / (false_positive + true_negative)
                if (false_positive + true_negative)
                else 0.0
            )
            rows.append(
                {
                    "class": class_name,
                    "fpr": false_positive_rate,
                    "detection_rate": detection_rate,
                }
            )
        frame = pd.DataFrame(rows)
        sns.scatterplot(
            data=frame,
            x="fpr",
            y="detection_rate",
            hue="class",
            s=90,
            ax=axis,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", alpha=0.5)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("Detection Rate / Recall")
    axis.set_title("Detection Rate vs False Positive Rate")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    return _save_png_and_pdf(figure, output_path)


def save_model_artifact(
    model: Any,
    model_name: str,
    model_dir: str | Path,
    class_names: Sequence[str],
    feature_columns: Sequence[str],
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a model and metadata for explainability and triage."""
    artifact = {
        "model": model,
        "model_name": model_name,
        "class_names": list(class_names),
        "feature_columns": list(feature_columns),
        "metrics": metrics,
    }
    if extra:
        artifact.update(extra)
    path = Path(model_dir) / f"{model_name}.joblib"
    save_joblib(artifact, path)
    LOGGER.info("Saved %s model artifact to %s", model_name, path)
    return path


def load_best_model(model_dir: str | Path) -> dict[str, Any]:
    """Load the artifact selected by the training orchestrator."""
    directory = Path(model_dir)
    best_path = directory / "best_model.json"
    if not best_path.exists():
        raise FileNotFoundError(f"Best-model manifest not found: {best_path}")
    manifest = load_json(best_path)
    artifact_path = Path(str(manifest["artifact_path"]))
    if not artifact_path.is_absolute():
        artifact_path = directory / artifact_path.name
    return load_joblib(artifact_path)
