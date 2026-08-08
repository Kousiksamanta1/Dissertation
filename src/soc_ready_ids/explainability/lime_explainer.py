"""LIME tabular explanations used to cross-check SHAP explanations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS, normalize_dataset_name
from soc_ready_ids.data.preprocessor import load_processed_arrays
from soc_ready_ids.models.common import load_best_model
from soc_ready_ids.utils.io import ensure_dir
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


class LimeIDSExplainer:
    """Generate LIME HTML reports for IDS alerts."""

    def __init__(self, artifact: dict[str, Any], training_data: pd.DataFrame) -> None:
        """Create a LIME tabular explainer from training features."""
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install lime to generate LIME explanations.") from exc
        self.artifact = artifact
        self.model = artifact["model"]
        self.class_names = list(artifact["class_names"])
        self.feature_columns = list(artifact["feature_columns"])
        self.explainer = LimeTabularExplainer(
            training_data[self.feature_columns].to_numpy(),
            feature_names=self.feature_columns,
            class_names=self.class_names,
            mode="classification",
            discretize_continuous=True,
        )

    def explain_row(self, row: pd.Series, output_path: str | Path, num_features: int = 10) -> Path:
        """Save a LIME explanation HTML file for one alert."""
        explanation = self.explainer.explain_instance(
            row[self.feature_columns].to_numpy(),
            self.model.predict_proba,
            num_features=num_features,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        explanation.save_to_file(str(path))
        LOGGER.info("Saved LIME explanation to %s", path)
        return path


def generate_lime_reports_by_alert_type(
    artifact: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: list[int],
    output_dir: str | Path,
    num_features: int = 10,
) -> list[Path]:
    """Create one representative LIME HTML report per alert type."""
    explainer = LimeIDSExplainer(artifact, X_train)
    directory = ensure_dir(output_dir)
    paths: list[Path] = []
    for class_index, class_name in enumerate(artifact["class_names"]):
        matches = [idx for idx, label in enumerate(y_test) if int(label) == class_index]
        if not matches:
            continue
        row_index = matches[0]
        filename = f"lime_{class_name.replace(' ', '_').replace('/', '_')}.html"
        paths.append(explainer.explain_row(X_test.iloc[row_index], directory / filename, num_features=num_features))
    return paths


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for LIME reports."""
    parser = argparse.ArgumentParser(description="Generate LIME reports for the best IDS model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    dataset = normalize_dataset_name(
        args.dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(dataset)
    artifact = load_best_model(config.path("paths.model_dir"))
    X_train, X_test, _, y_test, _, _ = load_processed_arrays(config.path("paths.processed_data_dir"))
    generate_lime_reports_by_alert_type(
        artifact,
        X_train,
        X_test,
        list(y_test),
        config.path("paths.lime_dir"),
        int(config.get("explainability.lime_num_features", 10)),
    )


if __name__ == "__main__":
    main()
