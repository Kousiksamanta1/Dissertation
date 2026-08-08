"""SHAP explainers for global and local IDS model interpretability."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS, normalize_dataset_name
from soc_ready_ids.data.preprocessor import load_processed_arrays
from soc_ready_ids.explainability.explanation_generator import (
    generate_explanation_payload,
    save_explanation,
)
from soc_ready_ids.models.common import load_best_model, predict_proba_safe
from soc_ready_ids.utils.io import ensure_dir, load_joblib
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _mean_abs_shap(values: Any) -> np.ndarray:
    """Return mean absolute SHAP values per feature."""
    if isinstance(values, list):
        arrays = [np.asarray(item) for item in values]
        stacked = np.stack(arrays, axis=-1)
        return np.abs(stacked).mean(axis=(0, 2))
    array = np.asarray(values)
    if array.ndim == 3:
        return np.abs(array).mean(axis=(0, 2))
    return np.abs(array).mean(axis=0)


def _class_shap_values(
    values: Any, class_index: int, class_count: int
) -> np.ndarray:
    """Extract a two-dimensional SHAP array for one output class."""
    if isinstance(values, list):
        selected_index = min(class_index, len(values) - 1)
        selected = np.asarray(values[selected_index])
    else:
        selected = np.asarray(values)
        if selected.ndim == 3:
            output_count = selected.shape[-1]
            selected = selected[:, :, min(class_index, output_count - 1)]
    if selected.ndim == 1:
        selected = selected.reshape(1, -1)
    if class_count == 2 and class_index == 0 and (
        not isinstance(values, list) or len(values) == 1
    ):
        selected = -selected
    return selected


def _class_base_value(
    expected_value: Any, class_index: int, class_count: int
) -> float:
    """Extract one numeric SHAP base value."""
    values = np.asarray(expected_value).reshape(-1)
    if values.size == 0:
        return 0.0
    if values.size == 1:
        value = float(values[0])
        return -value if class_count == 2 and class_index == 0 else value
    return float(values[min(class_index, values.size - 1)])


class ShapIDSExplainer:
    """Generate global and local SHAP artifacts for saved IDS models."""

    def __init__(
        self,
        artifact: dict[str, Any],
        output_dir: str | Path,
        background: pd.DataFrame | None = None,
    ) -> None:
        """Create a tree or deep explainer from a saved model artifact."""
        self.artifact = artifact
        self.model = artifact.get("model")
        self.model_name = str(artifact.get("model_name", "model"))
        self.class_names = list(artifact.get("class_names", []))
        self.feature_columns = list(artifact.get("feature_columns", []))
        self.explainability_type = str(
            artifact.get("explainability_type", "tree")
        )
        self.output_dir = ensure_dir(output_dir)
        if background is not None:
            self.background = background.reindex(
                columns=self.feature_columns, fill_value=0.0
            )
        else:
            saved_background = np.asarray(
                artifact.get("background_data", []), dtype=float
            )
            self.background = (
                pd.DataFrame(saved_background, columns=self.feature_columns)
                if saved_background.size
                else pd.DataFrame(columns=self.feature_columns)
            )

    def _build_explainer(self) -> Any:
        """Build TreeExplainer or DeepExplainer for the current artifact."""
        try:
            import shap
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install shap to generate SHAP explanations."
            ) from exc
        if self.model is None:
            raise ValueError("The model artifact does not contain a model.")
        if self.explainability_type == "deep":
            import torch

            background = self.background.head(200)
            if background.empty:
                raise ValueError(
                    "Deep SHAP requires background data in the artifact."
                )
            module = self.model.deep_explainer_module()
            tensor = torch.tensor(
                background.to_numpy(dtype=np.float32), dtype=torch.float32
            )
            return shap.DeepExplainer(module, tensor)
        return shap.TreeExplainer(self.model)

    def _shap_values(
        self, frame: pd.DataFrame
    ) -> tuple[Any, Any]:
        """Compute SHAP values and expected values for a frame."""
        explainer = self._build_explainer()
        if self.explainability_type == "deep":
            import torch

            tensor = torch.tensor(
                frame.to_numpy(dtype=np.float32), dtype=torch.float32
            )
            return explainer.shap_values(tensor), explainer.expected_value
        try:
            values = explainer.shap_values(
                frame,
                check_additivity=False,
                approximate=True,
            )
        except TypeError:
            values = explainer.shap_values(frame)
        return values, explainer.expected_value

    def save_global_plots(
        self, X: pd.DataFrame, max_samples: int = 500
    ) -> dict[str, Path]:
        """Save SHAP summary and feature-importance plots as PNG and PDF."""
        try:
            import shap
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Install shap to generate SHAP plots."
            ) from exc

        sample = X.reindex(
            columns=self.feature_columns, fill_value=0.0
        ).head(max_samples)
        values, _ = self._shap_values(sample)
        summary_values: Any = values
        if self.explainability_type == "deep":
            summary_values = _class_shap_values(
                values, 1, max(2, len(self.class_names))
            )

        summary_path = (
            self.output_dir / f"{self.model_name}_shap_summary.png"
        )
        plt.figure()
        shap.summary_plot(
            summary_values,
            sample,
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches="tight")
        plt.savefig(summary_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close()

        importance = pd.DataFrame(
            {
                "feature": sample.columns,
                "mean_abs_shap": _mean_abs_shap(values),
            }
        ).sort_values("mean_abs_shap", ascending=False).head(20)
        importance.to_csv(
            self.output_dir
            / f"{self.model_name}_shap_feature_importance.csv",
            index=False,
        )
        bar_path = (
            self.output_dir
            / f"{self.model_name}_shap_feature_importance.png"
        )
        figure, axis = plt.subplots(figsize=(10, 7))
        axis.barh(
            importance["feature"][::-1],
            importance["mean_abs_shap"][::-1],
            color="#457b9d",
        )
        axis.set_xlabel("Mean |SHAP value|")
        axis.set_title("Global SHAP Feature Importance")
        figure.tight_layout()
        figure.savefig(bar_path, dpi=300, bbox_inches="tight")
        figure.savefig(bar_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        LOGGER.info("Saved SHAP global plots to %s", self.output_dir)
        return {"summary": summary_path, "bar": bar_path}

    def local_explanation(
        self,
        row: pd.Series | pd.DataFrame,
        alert_id: str,
        predicted_class_index: int | None = None,
        output_json_dir: str | Path | None = None,
        risk_score: float = 0.0,
    ) -> dict[str, Any]:
        """Generate local values, a SHAP waterfall, and analyst-facing JSON."""
        frame = row.to_frame().T if isinstance(row, pd.Series) else row.copy()
        frame = frame.reindex(columns=self.feature_columns, fill_value=0.0)
        probability = predict_proba_safe(self.model, frame)
        if predicted_class_index is None:
            if probability is not None:
                predicted_class_index = int(np.argmax(probability[0]))
            else:
                predicted_class_index = int(self.model.predict(frame)[0])
        confidence = (
            float(probability[0, predicted_class_index])
            if probability is not None
            else 0.0
        )
        attack_type = (
            self.class_names[predicted_class_index]
            if self.class_names
            else str(predicted_class_index)
        )

        try:
            values, expected_value = self._shap_values(frame)
            selected = _class_shap_values(
                values, predicted_class_index, len(self.class_names)
            )[0]
            base_value = _class_base_value(
                expected_value, predicted_class_index, len(self.class_names)
            )
        except Exception as exc:
            LOGGER.warning(
                "Falling back to feature-magnitude explanation: %s", exc
            )
            selected = frame.iloc[0].to_numpy(dtype=float)
            base_value = 0.0

        top_indices = np.argsort(np.abs(selected))[::-1][:10]
        top_features: list[dict[str, Any]] = []
        reference = self.background if not self.background.empty else frame
        for index in top_indices:
            feature_name = frame.columns[index]
            value = float(frame.iloc[0, index])
            percentile = float(
                (reference[feature_name].astype(float) <= value).mean() * 100.0
            )
            top_features.append(
                {
                    "feature": feature_name,
                    "value": round(value, 4),
                    "shap_value": round(float(selected[index]), 6),
                    "percentile": round(percentile, 2),
                }
            )

        plot_path = (
            self.output_dir / f"{alert_id}_shap_waterfall.png"
        )
        self._save_local_plot(
            frame.iloc[0], selected, base_value, plot_path
        )
        payload = generate_explanation_payload(
            alert_id,
            attack_type,
            confidence,
            risk_score,
            top_features[:3],
        )
        payload["shap_plot_path"] = str(plot_path)
        if output_json_dir:
            save_explanation(payload, output_json_dir)
        return payload

    def _save_local_plot(
        self,
        row: pd.Series,
        shap_values: np.ndarray,
        base_value: float,
        output_path: str | Path,
    ) -> Path:
        """Save an actual SHAP waterfall with a robust bar-chart fallback."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import shap

            explanation = shap.Explanation(
                values=np.asarray(shap_values, dtype=float),
                base_values=float(base_value),
                data=row.to_numpy(dtype=float),
                feature_names=list(row.index),
            )
            shap.plots.waterfall(explanation, max_display=10, show=False)
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches="tight")
            plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
            plt.close()
            return path
        except Exception as exc:  # pragma: no cover - plotting backend dependent
            LOGGER.warning("SHAP waterfall rendering failed: %s", exc)

        order = np.argsort(np.abs(shap_values))[::-1][:10]
        colors = [
            "#e76f51" if shap_values[index] >= 0 else "#2a9d8f"
            for index in order
        ]
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.barh(
            [row.index[index] for index in order][::-1],
            [shap_values[index] for index in order][::-1],
            color=colors[::-1],
        )
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_title(f"Local SHAP Contributions (base={base_value:.3f})")
        axis.set_xlabel("SHAP contribution")
        figure.tight_layout()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        return path


def build_deep_explainer_for_autoencoder(
    artifact_path: str | Path, background: pd.DataFrame
) -> Any:
    """Build a SHAP DeepExplainer from a saved autoencoder artifact."""
    try:
        import shap
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Install shap and torch to explain the autoencoder."
        ) from exc
    artifact = load_joblib(artifact_path)
    model = artifact["model"]
    tensor = torch.tensor(
        background[artifact["feature_columns"]].to_numpy(dtype=np.float32)
    )
    return shap.DeepExplainer(model.deep_explainer_module(), tensor)


def main(argv: Iterable[str] | None = None) -> None:
    """Generate global and example local SHAP explanations."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for the best IDS model."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    dataset = normalize_dataset_name(
        args.dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(dataset)
    artifact = load_best_model(config.path("paths.model_dir"))
    X_train, X_test, _, _, _, _ = load_processed_arrays(
        config.path("paths.processed_data_dir")
    )
    explainer = ShapIDSExplainer(
        artifact, config.path("paths.shap_dir"), background=X_train
    )
    explainer.save_global_plots(
        X_train,
        int(config.get("explainability.max_explanation_samples", 500)),
    )
    explainer.local_explanation(
        X_test.iloc[0],
        "example-alert-001",
        output_json_dir=config.path("paths.explanation_dir"),
    )


if __name__ == "__main__":
    main()
