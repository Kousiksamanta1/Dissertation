"""Time-window alert deduplication and alert-storm detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass
class StormSummary:
    """Alert storm summary result."""

    is_storm: bool
    peak_alerts_per_minute: int
    storm_windows: list[str]
    summary_text: str


def normalize_alert_columns(alerts: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected deduplication columns exist."""
    frame = alerts.copy()
    if "timestamp" not in frame.columns:
        frame["timestamp"] = pd.Timestamp.utcnow()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce").fillna(pd.Timestamp.utcnow())
    for column, default in {"src_ip": "unknown", "attack_type": "UNKNOWN", "dst_port": -1}.items():
        if column not in frame.columns:
            frame[column] = default
    if "risk_score" not in frame.columns:
        frame["risk_score"] = 0.0
    return frame


def deduplicate_alerts(alerts: pd.DataFrame, window_seconds: int = 60, threshold: int = 3) -> pd.DataFrame:
    """Merge repeated alerts with the same src_ip, attack_type, and dst_port."""
    if alerts.empty:
        return alerts.copy()
    frame = normalize_alert_columns(alerts).sort_values("timestamp").reset_index(drop=True)
    output_rows: list[dict] = []
    group_columns = ["src_ip", "attack_type", "dst_port"]
    window = timedelta(seconds=window_seconds)

    for _, group in frame.groupby(group_columns, dropna=False):
        records = group.sort_values("timestamp").to_dict("records")
        current: list[dict] = []
        window_start = None
        for record in records:
            timestamp = record["timestamp"]
            if window_start is None or timestamp - window_start <= window:
                current.append(record)
                window_start = window_start or timestamp
            else:
                output_rows.append(_merge_window(current, threshold))
                current = [record]
                window_start = timestamp
        if current:
            output_rows.append(_merge_window(current, threshold))

    deduped = pd.DataFrame(output_rows).sort_values("timestamp", ascending=False).reset_index(drop=True)
    return deduped


def _merge_window(records: list[dict], threshold: int) -> dict:
    """Merge one duplicate window into a representative row."""
    representative = max(records, key=lambda item: float(item.get("risk_score", 0.0))).copy()
    representative["duplicate_count"] = len(records)
    representative["deduplicated"] = len(records) > threshold
    representative["is_suppressed"] = bool(representative.get("is_suppressed", False))
    merged_ids: list[str] = []
    for item in records:
        existing_value = item.get("merged_alert_ids")
        if existing_value is None or pd.isna(existing_value):
            existing_value = item.get("alert_id", "")
        existing = str(existing_value)
        merged_ids.extend(
            value for value in existing.split(",") if value and value not in merged_ids
        )
    representative["merged_alert_ids"] = ",".join(merged_ids)
    return representative


def detect_alert_storm(alerts: pd.DataFrame, threshold_per_minute: int = 50) -> StormSummary:
    """Detect whether alert volume crosses the configured storm threshold."""
    if alerts.empty:
        return StormSummary(False, 0, [], "No alerts were present.")
    frame = normalize_alert_columns(alerts)
    counts = frame.set_index("timestamp").sort_index().resample("1min").size()
    storm_counts = counts[counts > threshold_per_minute]
    peak = int(counts.max()) if not counts.empty else 0
    if storm_counts.empty:
        return StormSummary(False, peak, [], f"No alert storm detected; peak was {peak} alerts/minute.")
    windows = [timestamp.isoformat() for timestamp in storm_counts.index]
    top_types = frame["attack_type"].value_counts().head(3).to_dict()
    summary = f"Alert storm detected with peak {peak} alerts/minute. Dominant alert types: {top_types}."
    return StormSummary(True, peak, windows, summary)
