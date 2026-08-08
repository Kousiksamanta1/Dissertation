"""Template-based natural-language explanation generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from soc_ready_ids.utils.constants import ATTACK_CONTEXT
from soc_ready_ids.utils.io import save_json


def canonical_attack_key(attack_type: str) -> str:
    """Map detailed dataset labels to operational explanation categories."""
    label = attack_type.upper().strip()
    if label in {"BENIGN", "NORMAL"}:
        return "BENIGN"
    if "DDOS" in label:
        return "DDOS"
    if "DOS" in label or "HEARTBLEED" in label:
        return "DOS"
    if "PORT" in label and "SCAN" in label:
        return "PORTSCAN"
    if "PATATOR" in label or "BRUTE" in label:
        return "BRUTE FORCE"
    if "BOT" in label:
        return "BOTNET"
    if "WEB" in label or "SQL" in label or "XSS" in label:
        return "WEB ATTACK"
    if "INFILTRATION" in label:
        return "INFILTRATION"
    if "RECON" in label or "SCAN" in label:
        return "RECONNAISSANCE"
    if "THEFT" in label or "KEYLOG" in label or "EXFIL" in label:
        return "THEFT"
    return label


def feature_phrase(feature: dict[str, Any]) -> str:
    """Convert one feature attribution into compact analyst-facing prose."""
    name = str(feature.get("feature", "unknown feature"))
    direction = "increased" if float(feature.get("shap_value", 0.0)) >= 0 else "reduced"
    direction_word = "up" if direction == "increased" else "down"
    percentile = feature.get("percentile")
    value = feature.get("value")
    if percentile is not None:
        return (
            f"{name} is at the {float(percentile):.0f}th percentile "
            f"and pushed risk {direction_word}"
        )
    if value is not None:
        return f"{name}={value} pushed the score {direction_word}"
    return f"{name} pushed the score {direction_word}"


def generate_explanation_text(attack_type: str, confidence: float, top_features: list[dict[str, Any]]) -> tuple[str, str]:
    """Create a three-sentence explanation and recommended action."""
    key = canonical_attack_key(attack_type)
    context = ATTACK_CONTEXT.get(
        key,
        {
            "context": f"{attack_type} is an attack category detected from flow-level network behavior.",
            "action": "Review the source, destination asset, and nearby correlated alerts before containment.",
        },
    )
    reasons = ", ".join(feature_phrase(feature) for feature in top_features[:3]) or "the model found no dominant feature"
    sentence_one = f"This alert was classified as {attack_type} with {confidence:.1%} confidence because {reasons}."
    sentence_two = context["context"]
    sentence_three = f"Recommended first response: {context['action']}"
    return " ".join([sentence_one, sentence_two, sentence_three]), context["action"]


def generate_explanation_payload(
    alert_id: str,
    attack_type: str,
    confidence: float,
    risk_score: float,
    top_features: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the required JSON-style explanation payload."""
    explanation_text, recommended_action = generate_explanation_text(attack_type, confidence, top_features)
    return {
        "alert_id": alert_id,
        "attack_type": attack_type,
        "confidence": round(float(confidence), 4),
        "risk_score": round(float(risk_score), 2),
        "top_features": top_features,
        "explanation_text": explanation_text,
        "recommended_action": recommended_action,
    }


def save_explanation(payload: dict[str, Any], output_dir: str | Path) -> Path:
    """Save one explanation payload as JSON."""
    return save_json(payload, Path(output_dir) / f"{payload['alert_id']}.json")
