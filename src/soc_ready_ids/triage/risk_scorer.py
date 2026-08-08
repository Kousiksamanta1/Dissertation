"""Composite SOC risk scoring for IDS alerts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from soc_ready_ids.explainability.explanation_generator import canonical_attack_key
from soc_ready_ids.utils.constants import ATTACK_SEVERITY, SEVERITY_TO_NUMERIC


@dataclass
class RiskResult:
    """Risk scoring output."""

    risk_score: float
    risk_tier: str
    attack_severity: str
    components: dict[str, float]


def lookup_attack_severity(attack_type: str) -> str:
    """Return severity label for an attack type."""
    label = attack_type.upper().strip()
    canonical = canonical_attack_key(attack_type)
    return ATTACK_SEVERITY.get(label, ATTACK_SEVERITY.get(canonical, "Medium"))


def time_of_day_factor(timestamp: datetime | None = None) -> float:
    """Return higher risk for off-hours alerts."""
    ts = timestamp or datetime.utcnow()
    return 100.0 if ts.hour < 7 or ts.hour >= 18 else 40.0


def tier_from_score(score: float, thresholds: dict[str, Any] | None = None) -> str:
    """Convert a numeric risk score into a SOC tier."""
    thresholds = thresholds or {}
    critical = float(thresholds.get("critical_threshold", 85))
    high = float(thresholds.get("high_threshold", 65))
    medium = float(thresholds.get("medium_threshold", 40))
    if score >= critical:
        return "Critical"
    if score >= high:
        return "High"
    if score >= medium:
        return "Medium"
    return "Low"


def compute_risk_score(
    confidence: float,
    attack_type: str,
    asset_criticality: float = 50.0,
    timestamp: datetime | None = None,
    thresholds: dict[str, Any] | None = None,
) -> RiskResult:
    """Compute composite risk score from confidence, severity, asset, and time."""
    confidence_score = max(0.0, min(float(confidence) * 100.0, 100.0))
    severity = lookup_attack_severity(attack_type)
    severity_score = float(SEVERITY_TO_NUMERIC.get(severity, 50))
    asset_score = max(0.0, min(float(asset_criticality), 100.0))
    time_score = time_of_day_factor(timestamp)
    score = (0.40 * confidence_score) + (0.30 * severity_score) + (0.20 * asset_score) + (0.10 * time_score)
    score = round(max(0.0, min(score, 100.0)), 2)
    return RiskResult(
        risk_score=score,
        risk_tier=tier_from_score(score, thresholds),
        attack_severity=severity,
        components={
            "confidence": confidence_score,
            "attack_severity": severity_score,
            "asset_criticality": asset_score,
            "time_of_day": time_score,
        },
    )
