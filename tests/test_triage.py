"""Tests for triage scoring and deduplication."""

from __future__ import annotations

import pandas as pd

from soc_ready_ids.triage.deduplicator import deduplicate_alerts, detect_alert_storm
from soc_ready_ids.triage.risk_scorer import compute_risk_score, lookup_attack_severity


def test_risk_scoring_tiers_high_attack() -> None:
    """High-confidence critical attacks should produce high risk."""
    result = compute_risk_score(0.95, "Botnet", asset_criticality=90)
    assert result.attack_severity == "Critical"
    assert result.risk_score >= 70
    assert result.risk_tier in {"High", "Critical"}


def test_lookup_attack_severity_variants() -> None:
    """Known CICIDS labels should map to expected severities."""
    assert lookup_attack_severity("PortScan") == "Medium"
    assert lookup_attack_severity("FTP-Patator") == "High"


def test_deduplicate_alerts_merges_repeated_window() -> None:
    """More than three same-key alerts in sixty seconds should merge."""
    alerts = pd.DataFrame(
        {
            "alert_id": [f"a{i}" for i in range(4)],
            "timestamp": pd.date_range("2026-01-01 00:00:00", periods=4, freq="10s"),
            "src_ip": ["10.0.0.1"] * 4,
            "attack_type": ["DDoS"] * 4,
            "dst_port": [80] * 4,
            "risk_score": [50, 60, 70, 80],
        }
    )
    deduped = deduplicate_alerts(alerts, window_seconds=60, threshold=3)
    assert len(deduped) == 1
    assert int(deduped.iloc[0]["duplicate_count"]) == 4
    assert bool(deduped.iloc[0]["deduplicated"])


def test_detect_alert_storm() -> None:
    """Alert storm detection should flag high per-minute volume."""
    alerts = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 00:00:00", periods=55, freq="s"),
            "attack_type": ["DDoS"] * 55,
        }
    )
    summary = detect_alert_storm(alerts, threshold_per_minute=50)
    assert summary.is_storm
    assert summary.peak_alerts_per_minute == 55
