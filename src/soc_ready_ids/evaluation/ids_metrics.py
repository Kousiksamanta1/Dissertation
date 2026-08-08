"""IDS model metric aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from soc_ready_ids.config import load_config
from soc_ready_ids.utils.io import load_json
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def collect_model_metrics(metrics_dir: str | Path) -> pd.DataFrame:
    """Collect saved metrics JSON files into a comparison table."""
    rows: list[dict] = []
    for path in sorted(Path(metrics_dir).glob("*_metrics.json")):
        metrics = load_json(path)
        if not isinstance(metrics, dict) or not metrics.get("model"):
            LOGGER.debug("Skipping non-model metrics file: %s", path)
            continue
        rows.append(
            {
                "model": metrics["model"],
                "accuracy": metrics.get("accuracy"),
                "precision_macro": metrics.get("precision_macro"),
                "recall_macro": metrics.get("recall_macro"),
                "f1_macro": metrics.get("f1_macro"),
                "roc_auc_macro_ovr": metrics.get("roc_auc_macro_ovr"),
                "binary_attack_f1_macro": metrics.get(
                    "binary_attack_f1_macro"
                ),
                "binary_attack_roc_auc": metrics.get(
                    "binary_attack_roc_auc"
                ),
            }
        )
    if not rows:
        LOGGER.warning("No model metric JSON files found in %s", metrics_dir)
        return pd.DataFrame(
            columns=[
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "roc_auc_macro_ovr",
                "binary_attack_f1_macro",
                "binary_attack_roc_auc",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        "f1_macro", ascending=False, na_position="last"
    )


def save_metric_table(table: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Save IDS comparison metrics as CSV and Markdown."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "ids_model_comparison.csv"
    md_path = directory / "ids_model_comparison.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(table.to_markdown(index=False), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for aggregating IDS metrics."""
    parser = argparse.ArgumentParser(description="Aggregate IDS model metrics.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    table = collect_model_metrics(config.path("paths.metrics_dir"))
    save_metric_table(table, config.path("paths.metrics_dir"))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
