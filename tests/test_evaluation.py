"""Tests for Phase 6 ground-truth evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from soc_ready_ids.config import ProjectConfig
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
from soc_ready_ids.evaluation.results_plotter import (
    plot_ablation,
    plot_alert_volume,
    plot_explanation_quality,
    plot_model_comparison,
    plot_triage_kpis,
)
from soc_ready_ids.evaluation.triage_metrics import (
    compute_triage_metrics,
    represented_alert_ids,
)


def _alerts() -> pd.DataFrame:
    """Create repeated attack and benign alerts with numeric features."""
    rows = []
    for incident, ground_truth in enumerate(["DDoS", "BENIGN", "PortScan"]):
        for duplicate in range(4):
            alert_id = f"a-{incident}-{duplicate}"
            rows.append(
                {
                    "alert_id": alert_id,
                    "incident_id": f"incident-{incident}",
                    "timestamp": pd.Timestamp("2026-01-01")
                    + pd.Timedelta(minutes=incident, seconds=duplicate * 5),
                    "src_ip": f"10.0.0.{incident + 1}",
                    "dst_port": 80,
                    "attack_type": ground_truth,
                    "ground_truth": ground_truth,
                    "risk_score": 60 + incident,
                    "f1": float(incident),
                    "f2": float(duplicate),
                    "top_features": [
                        {"feature": "f1", "value": incident, "shap_value": 1}
                    ],
                    "explanation_text": (
                        f"This alert was classified as {ground_truth} because f1 "
                        "pushed risk up. Network attack context is available. "
                        "Recommended first response: Review the source and isolate the host."
                    ),
                    "recommended_action": "Review the source and isolate the host.",
                }
            )
    return pd.DataFrame(rows)


def test_ground_truth_metrics_preserve_merged_attack_ids() -> None:
    """Merged duplicates should preserve all represented attack evidence."""
    before = _alerts()
    after = before.iloc[[0, 4, 8]].copy()
    after["merged_alert_ids"] = [
        "a-0-0,a-0-1,a-0-2,a-0-3",
        "a-1-0,a-1-1,a-1-2,a-1-3",
        "a-2-0,a-2-1,a-2-2,a-2-3",
    ]
    metrics = compute_triage_metrics(before, after)

    assert len(represented_alert_ids(after)) == 12
    assert metrics["alert_reduction_rate"] == 75.0
    assert metrics["true_positive_preservation_rate"] == 100.0
    assert metrics["false_negative_rate_suppressed_real_attacks"] == 0.0


def test_ground_truth_metrics_fall_back_from_missing_merged_ids() -> None:
    """NaN merged metadata should preserve the visible alert itself."""
    before = _alerts()
    after = before.iloc[[0, 8]].copy()
    after["merged_alert_ids"] = pd.NA

    represented = represented_alert_ids(after)

    assert represented == {"a-0-0", "a-2-0"}


def test_explanation_quality_and_ablation_outputs(tmp_path: Path) -> None:
    """SCS and ablation outputs should be reproducible files."""
    alerts = _alerts()
    scores = evaluate_explanations(alerts)
    quality_paths = save_explanation_quality(scores, tmp_path)
    ablation = run_ablation(
        alerts,
        ["f1", "f2"],
        {
            "hdbscan_min_cluster_size": 2,
            "cluster_window_minutes": 5,
            "duplicate_window_seconds": 60,
            "duplicate_threshold": 3,
        },
    )
    ablation_paths = save_ablation_results(ablation, tmp_path)

    assert scores["scs_overall"].mean() > 0.6
    assert quality_paths["summary"].exists()
    assert set(ablation["scenario"]) == {
        "full_pipeline",
        "no_clustering",
        "no_deduplication",
        "no_risk_scoring",
    }
    assert ablation_paths["markdown"].exists()


def test_metric_aggregation_and_result_plots(tmp_path: Path) -> None:
    """Saved metric inputs should produce all publication formats."""
    metrics_dir = tmp_path / "metrics"
    figures = tmp_path / "figures"
    metrics_dir.mkdir()
    for name, score in [("rf", 0.8), ("xgb", 0.9)]:
        (metrics_dir / f"{name}_metrics.json").write_text(
            json.dumps(
                {
                    "model": name,
                    "accuracy": score,
                    "precision_macro": score,
                    "recall_macro": score,
                    "f1_macro": score,
                    "roc_auc_macro_ovr": score,
                    "binary_attack_f1_macro": score,
                    "binary_attack_roc_auc": score,
                }
            ),
            encoding="utf-8",
        )
    (metrics_dir / "triage_metrics.json").write_text(
        json.dumps({"alert_reduction_rate": 75}),
        encoding="utf-8",
    )
    table = collect_model_metrics(metrics_dir)
    save_metric_table(table, metrics_dir)
    assert set(table["model"]) == {"rf", "xgb"}

    ablation = pd.DataFrame(
        {
            "scenario": ["full_pipeline", "no_clustering"],
            "alert_reduction_rate": [75, 50],
            "true_positive_preservation_rate": [100, 100],
        }
    )
    ablation.to_csv(metrics_dir / "ablation_study.csv", index=False)
    (metrics_dir / "triage_metrics.json").write_text(
        json.dumps(
            {
                "alert_reduction_rate": 75,
                "true_positive_preservation_rate": 100,
                "false_negative_rate_suppressed_real_attacks": 0,
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "explanation_quality_summary.json").write_text(
        json.dumps(
            {
                "completeness": 0.9,
                "actionability": 0.8,
                "conciseness": 1.0,
                "scs_overall": 0.9,
            }
        ),
        encoding="utf-8",
    )
    _alerts().to_csv(metrics_dir / "alerts.csv", index=False)

    outputs = [
        plot_model_comparison(table, figures),
        plot_ablation(metrics_dir / "ablation_study.csv", figures),
        plot_triage_kpis(metrics_dir / "triage_metrics.json", figures),
        plot_explanation_quality(
            metrics_dir / "explanation_quality_summary.json", figures
        ),
        plot_alert_volume(metrics_dir / "alerts.csv", figures),
    ]
    assert all(output and output["pdf"].exists() for output in outputs)


def test_pipeline_evaluator_with_mock_alert_builder(
    tmp_path: Path, monkeypatch
) -> None:
    """Pipeline evaluator should save before/after tables and metrics."""
    import soc_ready_ids.evaluation.pipeline_evaluator as evaluator

    alerts = _alerts()
    monkeypatch.setattr(
        evaluator, "build_evaluation_alerts", lambda config: (alerts, ["f1", "f2"])
    )
    config = ProjectConfig(
        values={
            "paths": {
                "metrics_dir": str(tmp_path / "metrics"),
                "figure_dir": str(tmp_path / "figures"),
            },
            "triage": {
                "hdbscan_min_cluster_size": 2,
                "cluster_window_minutes": 5,
                "duplicate_window_seconds": 60,
                "duplicate_threshold": 3,
            },
        },
        root=tmp_path,
    )
    before, after, metrics = evaluate_triage_pipeline(config)

    assert len(before) == 12
    assert len(after) < len(before)
    assert metrics["true_positive_preservation_rate"] == 100.0
    assert (tmp_path / "metrics" / "triage_metrics.json").exists()
