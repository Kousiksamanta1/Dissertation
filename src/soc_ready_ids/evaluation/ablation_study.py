"""Ablation study for triage-pipeline components."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from soc_ready_ids.config import load_config
from soc_ready_ids.evaluation.triage_metrics import compute_triage_metrics
from soc_ready_ids.triage.alert_clusterer import cluster_alerts
from soc_ready_ids.triage.deduplicator import deduplicate_alerts
from soc_ready_ids.utils.io import ensure_dir


def run_ablation(
    alerts: pd.DataFrame,
    feature_columns: list[str],
    triage_config: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate clustering, deduplication, and risk-scoring removals."""
    scenarios = [
        {
            "name": "full_pipeline",
            "clustering": True,
            "deduplication": True,
            "risk_scoring": True,
        },
        {
            "name": "no_clustering",
            "clustering": False,
            "deduplication": True,
            "risk_scoring": True,
        },
        {
            "name": "no_deduplication",
            "clustering": True,
            "deduplication": False,
            "risk_scoring": True,
        },
        {
            "name": "no_risk_scoring",
            "clustering": True,
            "deduplication": True,
            "risk_scoring": False,
        },
    ]
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        working = alerts.copy()
        if not scenario["risk_scoring"] and "risk_score" in working.columns:
            working["risk_score"] = 50.0
        if scenario["clustering"]:
            clustered = cluster_alerts(
                working, feature_columns, triage_config
            )
            visible = clustered[
                ~clustered["is_suppressed"].astype(bool)
            ].copy()
        else:
            visible = working.copy()
            visible["merged_alert_ids"] = visible["alert_id"].astype(str)
        if scenario["deduplication"]:
            after = deduplicate_alerts(
                visible,
                int(triage_config.get("duplicate_window_seconds", 60)),
                int(triage_config.get("duplicate_threshold", 3)),
            )
        else:
            after = visible
        metrics = compute_triage_metrics(alerts, after)
        rows.append(
            {
                "scenario": scenario["name"],
                **metrics,
                "mean_visible_risk": round(
                    float(after["risk_score"].mean())
                    if "risk_score" in after and not after.empty
                    else 0.0,
                    3,
                ),
            }
        )
    return pd.DataFrame(rows)


def save_ablation_results(
    results: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    """Save ablation tables as CSV and Markdown."""
    directory = ensure_dir(output_dir)
    csv_path = directory / "ablation_study.csv"
    markdown_path = directory / "ablation_study.md"
    results.to_csv(csv_path, index=False)
    markdown_path.write_text(
        results.to_markdown(index=False), encoding="utf-8"
    )
    return {"csv": csv_path, "markdown": markdown_path}


def main(argv: Iterable[str] | None = None) -> None:
    """Run the ablation study from a saved alert table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--alerts", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    alerts = pd.read_csv(args.alerts)
    feature_columns = [
        column
        for column in alerts.columns
        if alerts[column].dtype.kind in "if"
        and column
        not in {
            "confidence",
            "risk_score",
            "cluster_id",
            "cluster_x",
            "cluster_y",
        }
    ]
    results = run_ablation(
        alerts, feature_columns, config.get("triage", {})
    )
    save_ablation_results(results, config.path("paths.metrics_dir"))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
