"""Run all explainability and ground-truth evaluation phases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS, normalize_dataset_name
from soc_ready_ids.data.preprocessor import load_processed_arrays
from soc_ready_ids.evaluation.ablation_study import (
    run_ablation,
    save_ablation_results,
)
from soc_ready_ids.evaluation.explanation_quality import (
    evaluate_explanations,
    save_explanation_quality,
)
from soc_ready_ids.evaluation.ids_metrics import (
    collect_model_metrics,
    save_metric_table,
)
from soc_ready_ids.evaluation.pipeline_evaluator import (
    evaluate_triage_pipeline,
)
from soc_ready_ids.evaluation.results_plotter import generate_all_plots
from soc_ready_ids.explainability.lime_explainer import (
    generate_lime_reports_by_alert_type,
)
from soc_ready_ids.explainability.shap_explainer import ShapIDSExplainer
from soc_ready_ids.models.common import load_best_model
from soc_ready_ids.utils.io import load_json, save_json
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _source_metrics_section(
    dataset: str,
    processed_dir: Path,
    metrics_dir: Path,
    best_model_name: str,
) -> str:
    """Return and persist held-out metrics grouped by source dataset."""
    if dataset != "combined":
        return ""
    metadata_path = processed_dir / "metadata_test.csv"
    predictions_path = (
        metrics_dir / f"{best_model_name}_predictions.csv"
    )
    if not metadata_path.exists() or not predictions_path.exists():
        return ""
    metadata = pd.read_csv(metadata_path)
    predictions = pd.read_csv(predictions_path)
    if (
        "source_dataset" not in metadata.columns
        or len(metadata) != len(predictions)
    ):
        return ""
    rows: list[dict[str, Any]] = []
    for source_dataset, positions in metadata.groupby(
        "source_dataset"
    ).indices.items():
        source_predictions = predictions.iloc[positions]
        rows.append(
            {
                "source_dataset": source_dataset,
                "test_rows": len(source_predictions),
                "accuracy": accuracy_score(
                    source_predictions["y_true"],
                    source_predictions["y_pred"],
                ),
                "f1_macro": f1_score(
                    source_predictions["y_true"],
                    source_predictions["y_pred"],
                    average="macro",
                    zero_division=0,
                ),
            }
        )
    source_metrics = pd.DataFrame(rows).sort_values("source_dataset")
    source_metrics.to_csv(
        metrics_dir / "source_dataset_metrics.csv", index=False
    )
    markdown = source_metrics.to_markdown(index=False)
    (metrics_dir / "source_dataset_metrics.md").write_text(
        markdown + "\n", encoding="utf-8"
    )
    return (
        "\n## Unified Model by Source Dataset\n\n"
        "The same selected model was evaluated separately on held-out rows "
        "from each original dataset.\n\n"
        f"{markdown}\n"
    )


def _labels_for_artifact(
    y_test: np.ndarray, original_classes: list[str], model_classes: list[str]
) -> list[int]:
    """Align original multiclass labels with a binary anomaly artifact."""
    if [name.upper() for name in model_classes] == ["BENIGN", "ATTACK"]:
        benign_index = next(
            (
                index
                for index, name in enumerate(original_classes)
                if name.upper() in {"BENIGN", "NORMAL"}
            ),
            0,
        )
        return np.where(y_test == benign_index, 0, 1).astype(int).tolist()
    return y_test.astype(int).tolist()


def write_results_summary(
    config_path: str = "config.yaml", dataset: str | None = None
) -> Path:
    """Generate the dissertation-facing Markdown results summary."""
    config = load_config(config_path)
    selected_dataset = normalize_dataset_name(
        dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(selected_dataset)
    metrics_dir = config.path("paths.metrics_dir")
    report_path = config.path("paths.report_dir") / "results_summary.md"
    preprocessing = load_json(
        config.path("paths.processed_data_dir")
        / "preprocessing_manifest.json"
    )
    best_model = load_json(
        config.path("paths.model_dir") / "best_model.json"
    )
    triage = load_json(metrics_dir / "triage_metrics.json")
    quality = load_json(
        metrics_dir / "explanation_quality_summary.json"
    )
    comparison_path = metrics_dir / "ids_model_comparison.md"
    comparison = (
        comparison_path.read_text(encoding="utf-8")
        if comparison_path.exists()
        else "Model comparison was not generated."
    )
    ablation_path = metrics_dir / "ablation_study.md"
    ablation = (
        ablation_path.read_text(encoding="utf-8")
        if ablation_path.exists()
        else "Ablation table was not generated."
    )
    data_notice = (
        "> Results use the configured real dataset sample. Record the row cap,\n"
        "> sampling procedure, and class distribution when reporting findings."
    )
    source_metrics = _source_metrics_section(
        selected_dataset,
        config.path("paths.processed_data_dir"),
        metrics_dir,
        str(best_model.get("best_model", "")),
    )
    content = f"""# Results Summary

This file is generated from the current pipeline artifacts. The active dataset
is **{preprocessing.get('dataset')}** with {preprocessing.get('retained_rows')}
retained rows and {preprocessing.get('feature_count')} encoded features.

{data_notice}

## IDS Model Comparison

{comparison}

The selected downstream model is **{best_model.get('best_model')}**, chosen by
`{best_model.get('selection_metric')}` with score
**{float(best_model.get('selection_score', 0.0)):.4f}**.

## Triage Ground-Truth Evaluation

- Alerts before triage: {triage.get('alerts_before')}
- Alerts after triage: {triage.get('alerts_after')}
- Alert reduction rate: {triage.get('alert_reduction_rate')}%
- True-positive preservation rate: {triage.get('true_positive_preservation_rate')}%
- False-negative rate for unrepresented attacks: {triage.get('false_negative_rate_suppressed_real_attacks')}%
- Mean explanation length: {triage.get('mean_explanation_length')} words
- Feature coverage: {triage.get('feature_coverage')}

Evaluation design: {triage.get('evaluation_design')}.

## Explanation Quality

- Completeness: {quality.get('completeness')}
- Actionability: {quality.get('actionability')}
- Conciseness: {quality.get('conciseness')}
- Overall SCS: {quality.get('scs_overall')}
- Evaluated explanations: {quality.get('sample_count')}

{source_metrics}

## Ablation Study

{ablation}

## Generated Evidence

- Model metrics and prediction tables: `results/{selected_dataset}/metrics/`
- SHAP summary, importance, and waterfall plots: `results/{selected_dataset}/shap/`
- LIME representative HTML reports: `results/{selected_dataset}/lime/`
- Publication PNG/PDF figures: `results/{selected_dataset}/figures/`
- Wazuh integration and OpenSearch guide: `wazuh/`
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    return report_path


def run_all_evaluations(
    config_path: str = "config.yaml", dataset: str | None = None
) -> dict[str, Any]:
    """Run model aggregation, XAI, triage, ablation, quality, and plots."""
    config = load_config(config_path)
    selected_dataset = normalize_dataset_name(
        dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(selected_dataset)
    metrics_dir = config.path("paths.metrics_dir")
    model_metrics = collect_model_metrics(metrics_dir)
    metric_paths = save_metric_table(model_metrics, metrics_dir)

    artifact = load_best_model(config.path("paths.model_dir"))
    X_train, X_test, _, y_test, label_encoder, _ = load_processed_arrays(
        config.path("paths.processed_data_dir")
    )
    shap_explainer = ShapIDSExplainer(
        artifact,
        config.path("paths.shap_dir"),
        background=X_train.head(
            int(config.get("explainability.max_background_samples", 200))
        ),
    )
    shap_paths = shap_explainer.save_global_plots(
        X_train,
        int(config.get("explainability.max_explanation_samples", 500)),
    )
    local_payload = shap_explainer.local_explanation(
        X_test.iloc[0],
        "evaluation-example-alert",
        output_json_dir=config.path("paths.explanation_dir"),
    )

    aligned_labels = _labels_for_artifact(
        y_test,
        list(label_encoder.classes_),
        list(artifact["class_names"]),
    )
    lime_paths = generate_lime_reports_by_alert_type(
        artifact,
        X_train,
        X_test,
        aligned_labels,
        config.path("paths.lime_dir"),
        int(config.get("explainability.lime_num_features", 10)),
    )

    before, after, triage_metrics = evaluate_triage_pipeline(config)
    quality_scores = evaluate_explanations(after)
    quality_paths = save_explanation_quality(quality_scores, metrics_dir)
    ablation = run_ablation(
        before,
        list(artifact["feature_columns"]),
        config.get("triage", {}),
    )
    ablation_paths = save_ablation_results(ablation, metrics_dir)
    plot_paths = generate_all_plots(config_path, selected_dataset)
    report_path = write_results_summary(config_path, selected_dataset)
    manifest = {
        "model_metrics": {key: str(value) for key, value in metric_paths.items()},
        "shap": {key: str(value) for key, value in shap_paths.items()},
        "local_explanation": local_payload,
        "lime_reports": [str(path) for path in lime_paths],
        "triage_metrics": triage_metrics,
        "quality": {key: str(value) for key, value in quality_paths.items()},
        "ablation": {key: str(value) for key, value in ablation_paths.items()},
        "plots": plot_paths,
        "report": str(report_path),
    }
    save_json(manifest, metrics_dir / "evaluation_manifest.json")
    LOGGER.info("Completed all evaluation phases")
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    """Run every evaluation phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS))
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = run_all_evaluations(args.config, args.dataset)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
