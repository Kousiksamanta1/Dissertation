"""Dataset-ground-truth evaluation of alert triage behavior."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from soc_ready_ids.config import ProjectConfig
from soc_ready_ids.data.preprocessor import load_processed_arrays
from soc_ready_ids.evaluation.triage_metrics import compute_triage_metrics
from soc_ready_ids.explainability.explanation_generator import (
    generate_explanation_payload,
)
from soc_ready_ids.models.common import load_best_model, predict_proba_safe
from soc_ready_ids.triage.alert_clusterer import (
    cluster_alerts,
    save_cluster_map,
)
from soc_ready_ids.triage.deduplicator import deduplicate_alerts
from soc_ready_ids.triage.risk_scorer import compute_risk_score
from soc_ready_ids.utils.io import ensure_dir, save_json


def _metadata_value(
    metadata: pd.DataFrame, index: int, aliases: list[str], default: Any
) -> Any:
    """Return the first available metadata alias for one row."""
    lookup = {column.lower(): column for column in metadata.columns}
    for alias in aliases:
        column = lookup.get(alias.lower())
        if column is not None and index < len(metadata):
            value = metadata.iloc[index][column]
            if pd.notna(value):
                return value
    return default


def _top_features(row: pd.Series, count: int = 3) -> list[dict[str, Any]]:
    """Create deterministic local feature evidence for quality evaluation."""
    top = row.abs().sort_values(ascending=False).head(count)
    return [
        {
            "feature": feature,
            "value": round(float(row[feature]), 4),
            "shap_value": round(float(row[feature]), 6),
        }
        for feature in top.index
    ]


def build_evaluation_alerts(
    config: ProjectConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a controlled duplicate-burst alert set from held-out data."""
    processed = config.path("paths.processed_data_dir")
    X_train, X_test, _, y_test, label_encoder, feature_columns = (
        load_processed_arrays(processed)
    )
    artifact = load_best_model(config.path("paths.model_dir"))
    model = artifact["model"]
    model_classes = list(artifact["class_names"])
    original_classes = list(label_encoder.classes_)
    maximum = min(
        len(X_test), int(config.get("evaluation.max_base_alerts", 100))
    )
    repetitions = max(
        1, int(config.get("evaluation.duplicate_repetitions", 4))
    )
    predictions = np.asarray(model.predict(X_test.iloc[:maximum]), dtype=int)
    probabilities = predict_proba_safe(model, X_test.iloc[:maximum])
    metadata_path = processed / "metadata_test.csv"
    metadata = (
        pd.read_csv(metadata_path)
        if metadata_path.exists()
        else pd.DataFrame(index=range(maximum))
    )
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    records: list[dict[str, Any]] = []

    for index in range(maximum):
        predicted_index = int(predictions[index])
        attack_type = model_classes[predicted_index]
        confidence = (
            float(probabilities[index, predicted_index])
            if probabilities is not None
            else 0.0
        )
        ground_truth = original_classes[int(y_test[index])]
        source_ip = _metadata_value(
            metadata,
            index,
            ["Src IP", "Source IP", "saddr"],
            f"10.0.{index // 250}.{index % 250 + 1}",
        )
        destination_ip = _metadata_value(
            metadata,
            index,
            ["Dst IP", "Destination IP", "daddr"],
            f"192.168.10.{index % 50 + 1}",
        )
        destination_port = 80 if index % 2 == 0 else 443
        top_features = _top_features(X_test.iloc[index])
        for repetition in range(repetitions):
            timestamp = start + timedelta(
                seconds=(index * 70) + (repetition * 10)
            )
            risk = compute_risk_score(
                confidence,
                attack_type,
                asset_criticality=50 + (index % 6) * 10,
                timestamp=timestamp.replace(tzinfo=None),
                thresholds=config.get("triage", {}),
            )
            alert_id = f"eval-{index:04d}-{repetition}"
            explanation = generate_explanation_payload(
                alert_id,
                attack_type,
                confidence,
                risk.risk_score,
                top_features,
            )
            record: dict[str, Any] = {
                "alert_id": alert_id,
                "incident_id": f"incident-{index:04d}",
                "timestamp": timestamp.isoformat(),
                "src_ip": str(source_ip),
                "dst_ip": str(destination_ip),
                "dst_port": destination_port,
                "attack_type": attack_type,
                "ground_truth": ground_truth,
                "confidence": confidence,
                "risk_score": risk.risk_score,
                "risk_tier": risk.risk_tier,
                "top_features": explanation["top_features"],
                "explanation_text": explanation["explanation_text"],
                "recommended_action": explanation["recommended_action"],
                "is_suppressed": False,
            }
            record.update(X_test.iloc[index].to_dict())
            records.append(record)
    return pd.DataFrame(records), feature_columns


def evaluate_triage_pipeline(
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run clustering and deduplication on the held-out stress scenario."""
    before, feature_columns = build_evaluation_alerts(config)
    clustered = cluster_alerts(
        before, feature_columns, config.get("triage", {})
    )
    visible = clustered[
        ~clustered["is_suppressed"].astype(bool)
    ].copy()
    after = deduplicate_alerts(
        visible,
        int(config.get("triage.duplicate_window_seconds", 60)),
        int(config.get("triage.duplicate_threshold", 3)),
    )
    metrics = compute_triage_metrics(before, after)
    metrics["evaluation_design"] = (
        "Held-out dataset flows repeated in controlled four-alert bursts"
    )
    metrics["base_test_flows"] = int(
        before["incident_id"].nunique() if not before.empty else 0
    )

    metrics_dir = ensure_dir(config.path("paths.metrics_dir"))
    figures_dir = ensure_dir(config.path("paths.figure_dir"))
    serializable_before = before.copy()
    serializable_after = after.copy()
    for frame in (serializable_before, serializable_after):
        if "top_features" in frame.columns:
            frame["top_features"] = frame["top_features"].map(
                lambda value: json.dumps(value, default=str)
            )
    serializable_before.to_csv(
        metrics_dir / "triage_alerts_before.csv", index=False
    )
    serializable_after.to_csv(
        metrics_dir / "triage_alerts_after.csv", index=False
    )
    save_json(metrics, metrics_dir / "triage_metrics.json")
    save_cluster_map(
        clustered, figures_dir / "alert_cluster_scatter.png"
    )
    return before, after, metrics
