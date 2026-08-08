"""Generate a combined report for all completed real-dataset runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS
from soc_ready_ids.utils.io import load_json


def collect_dataset_results(
    config_path: str = "config.yaml",
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Collect comparable metrics and metadata from each dataset run."""
    base_config = load_config(config_path)
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    for dataset in SUPPORTED_DATASETS:
        config = base_config.for_dataset(dataset)
        processed = config.path("paths.processed_data_dir")
        model_dir = config.path("paths.model_dir")
        metrics_dir = config.path("paths.metrics_dir")
        required = [
            processed / "preprocessing_manifest.json",
            model_dir / "best_model.json",
            metrics_dir / "triage_metrics.json",
            metrics_dir / "explanation_quality_summary.json",
        ]
        if not all(path.exists() for path in required):
            continue
        preprocessing = load_json(required[0])
        best_model = load_json(required[1])
        triage = load_json(required[2])
        quality = load_json(required[3])
        selected_metrics = next(
            (
                item.get("metrics", {})
                for item in best_model.get("all_results", [])
                if item.get("model") == best_model.get("best_model")
            ),
            {},
        )
        details[dataset] = {
            "preprocessing": preprocessing,
            "best_model": best_model,
            "triage": triage,
            "quality": quality,
        }
        rows.append(
            {
                "dataset": dataset,
                "retained_rows": preprocessing.get("retained_rows"),
                "classes": len(preprocessing.get("classes", [])),
                "best_model": best_model.get("best_model"),
                "accuracy": selected_metrics.get("accuracy"),
                "f1_macro": selected_metrics.get("f1_macro"),
                "binary_attack_f1_macro": selected_metrics.get(
                    "binary_attack_f1_macro"
                ),
                "alert_reduction_rate": triage.get("alert_reduction_rate"),
                "tp_preservation_rate": triage.get(
                    "true_positive_preservation_rate"
                ),
                "explanation_scs": quality.get("scs_overall"),
            }
        )
    return pd.DataFrame(rows), details


def write_combined_summary(config_path: str = "config.yaml") -> Path:
    """Write the top-level comparison report."""
    table, details = collect_dataset_results(config_path)
    if table.empty:
        raise RuntimeError("No completed dataset runs were found.")
    output = load_config(config_path).root / "reports" / "results_summary.md"
    sections = [
        "# Combined Real-Dataset Results",
        "",
        "CICIDS2017 and BoT-IoT were evaluated independently, and the combined "
        "run trained one unified model on both source schemas with a common "
        "label taxonomy. "
        "Artifacts are isolated by run.",
        "",
        table.to_markdown(index=False),
        "",
        "> Compare results with care: the configured samples preserve each "
        "dataset's observed class distribution, and the supplied BoT-IoT files "
        "contain very few benign rows. The combined run uses a broad common "
        "label taxonomy and equal row allocation from each source.",
    ]
    for dataset, payload in details.items():
        preprocessing = payload["preprocessing"]
        dataset_section = [
            "",
            f"## {dataset}",
            "",
            f"- Dataset report: `reports/{dataset}/results_summary.md`",
            f"- Processed data: `data/processed/{dataset}/`",
            f"- Models: `models/saved/{dataset}/`",
            f"- Results: `results/{dataset}/`",
            f"- Classes: {', '.join(preprocessing.get('classes', []))}",
            "- Class distribution: "
            + ", ".join(
                f"{name}={count}"
                for name, count in preprocessing.get(
                    "class_distribution", {}
                ).items()
            ),
        ]
        source_distribution = preprocessing.get(
            "source_distribution", {}
        )
        if source_distribution:
            dataset_section.append(
                "- Source distribution: "
                + ", ".join(
                    f"{name}={count}"
                    for name, count in source_distribution.items()
                )
            )
        sections.extend(dataset_section)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return output


def main(argv: Iterable[str] | None = None) -> None:
    """Generate the combined report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(write_combined_summary(args.config))


if __name__ == "__main__":
    main()
