"""Ground-truth triage evaluation metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def alert_reduction_rate(alerts_before: int, alerts_after: int) -> float:
    """Compute alert reduction percentage."""
    return round(
        (
            (alerts_before - alerts_after) / alerts_before * 100.0
            if alerts_before
            else 0.0
        ),
        2,
    )


def _is_attack(values: pd.Series) -> pd.Series:
    """Return a mask for non-benign ground-truth values."""
    return ~values.fillna("BENIGN").astype(str).str.upper().isin(
        {"BENIGN", "NORMAL", "0", "FALSE"}
    )


def represented_alert_ids(after: pd.DataFrame) -> set[str]:
    """Return original alert IDs represented by visible/merged rows."""
    represented: set[str] = set()
    if after.empty:
        return represented
    for _, row in after.iterrows():
        merged_value = row.get("merged_alert_ids")
        if merged_value is None or pd.isna(merged_value) or not str(merged_value):
            merged_value = row.get("alert_id", "")
        merged = str(merged_value)
        represented.update(value for value in merged.split(",") if value)
    return represented


def true_positive_preservation_rate(
    before: pd.DataFrame,
    after: pd.DataFrame,
    label_column: str = "ground_truth",
) -> float:
    """Measure real attack alerts represented after triage reduction."""
    if before.empty or label_column not in before.columns:
        return 0.0
    attacks = before[_is_attack(before[label_column])]
    if attacks.empty:
        return 100.0
    if "alert_id" not in attacks.columns:
        if label_column not in after.columns:
            return 0.0
        visible_attacks = after[_is_attack(after[label_column])]
        return round(min(len(visible_attacks) / len(attacks), 1.0) * 100.0, 2)
    represented = represented_alert_ids(after)
    preserved = attacks["alert_id"].astype(str).isin(represented).sum()
    return round(float(preserved) / len(attacks) * 100.0, 2)


def false_negative_rate_suppressed_real_attacks(
    before: pd.DataFrame,
    after: pd.DataFrame,
    label_column: str = "ground_truth",
) -> float:
    """Compute real attacks no longer represented after triage."""
    preservation = true_positive_preservation_rate(
        before, after, label_column=label_column
    )
    return round(max(0.0, 100.0 - preservation), 2)


def mean_explanation_length(
    alerts: pd.DataFrame, text_column: str = "explanation_text"
) -> float:
    """Compute mean explanation length in words."""
    if alerts.empty or text_column not in alerts.columns:
        return 0.0
    lengths = (
        alerts[text_column]
        .fillna("")
        .astype(str)
        .map(lambda text: len(text.split()))
    )
    return round(float(lengths.mean()), 2)


def feature_coverage(
    alerts: pd.DataFrame,
    top_feature_column: str = "top_features",
    text_column: str = "explanation_text",
) -> float:
    """Measure how many named top features appear in explanation text."""
    if (
        alerts.empty
        or text_column not in alerts.columns
        or top_feature_column not in alerts.columns
    ):
        return 0.0
    coverages: list[float] = []
    for _, row in alerts.iterrows():
        text = str(row.get(text_column, "")).lower()
        features = row.get(top_feature_column, [])
        if isinstance(features, str):
            try:
                import json

                features = json.loads(features)
            except (json.JSONDecodeError, TypeError):
                features = [
                    item.strip() for item in features.split(",") if item.strip()
                ]
        names = [
            (
                str(item.get("feature", item)).lower()
                if isinstance(item, dict)
                else str(item).lower()
            )
            for item in features[:10]
        ]
        if names:
            coverages.append(
                sum(1 for name in names if name in text) / len(names)
            )
    return round(
        float(pd.Series(coverages).mean()) if coverages else 0.0, 3
    )


def compute_triage_metrics(
    before: pd.DataFrame, after: pd.DataFrame
) -> dict[str, Any]:
    """Compute all dissertation triage metrics."""
    return {
        "alerts_before": len(before),
        "alerts_after": len(after),
        "alert_reduction_rate": alert_reduction_rate(
            len(before), len(after)
        ),
        "true_positive_preservation_rate": true_positive_preservation_rate(
            before, after
        ),
        "false_negative_rate_suppressed_real_attacks": (
            false_negative_rate_suppressed_real_attacks(before, after)
        ),
        "mean_explanation_length": mean_explanation_length(after),
        "feature_coverage": feature_coverage(after),
    }
