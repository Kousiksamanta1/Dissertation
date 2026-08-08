"""SCS explanation-quality scoring without human participants."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from soc_ready_ids.config import load_config
from soc_ready_ids.utils.io import ensure_dir, save_json

ACTION_VERBS: tuple[str, ...] = (
    "check",
    "review",
    "inspect",
    "isolate",
    "block",
    "validate",
    "escalate",
    "collect",
    "apply",
    "enforce",
)


def _feature_names(value: Any) -> list[str]:
    """Extract feature names from JSON, dictionaries, or strings."""
    features = value
    if isinstance(features, str):
        try:
            features = json.loads(features)
        except json.JSONDecodeError:
            features = [
                item.strip() for item in features.split(",") if item.strip()
            ]
    if not isinstance(features, list):
        return []
    return [
        str(item.get("feature", item))
        if isinstance(item, dict)
        else str(item)
        for item in features
    ]


def score_explanation(row: pd.Series) -> dict[str, float]:
    """Score completeness, actionability, and conciseness from 0 to 1."""
    text = str(row.get("explanation_text", "")).strip()
    lower_text = text.lower()
    recommended_action = str(row.get("recommended_action", "")).strip()
    features = _feature_names(row.get("top_features", []))[:3]

    required_values = [
        row.get("attack_type"),
        row.get("confidence"),
        row.get("risk_score"),
    ]
    required_score = sum(
        value not in (None, "") for value in required_values
    ) / len(required_values)
    feature_score = (
        sum(name.lower() in lower_text for name in features) / len(features)
        if features
        else 0.0
    )
    context_score = float(
        any(
            token in lower_text
            for token in (
                "traffic",
                "activity",
                "attack",
                "network",
                "service",
            )
        )
    )
    completeness = min(
        1.0, (0.4 * required_score) + (0.4 * feature_score) + (0.2 * context_score)
    )

    action_text = f"{recommended_action} {text}".lower()
    has_action = float(bool(recommended_action))
    has_verb = float(any(verb in action_text for verb in ACTION_VERBS))
    specific_object = float(len(recommended_action.split()) >= 5)
    actionability = (
        0.4 * has_action + 0.4 * has_verb + 0.2 * specific_object
    )

    word_count = len(text.split())
    sentence_count = len(
        [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
            if sentence.strip()
        ]
    )
    if 30 <= word_count <= 90:
        length_score = 1.0
    elif 20 <= word_count < 30 or 90 < word_count <= 110:
        length_score = 0.7
    elif word_count:
        length_score = 0.4
    else:
        length_score = 0.0
    sentence_score = 1.0 if sentence_count == 3 else max(
        0.0, 1.0 - 0.25 * abs(sentence_count - 3)
    )
    conciseness = 0.7 * length_score + 0.3 * sentence_score

    overall = (completeness + actionability + conciseness) / 3.0
    return {
        "completeness": round(completeness, 3),
        "actionability": round(actionability, 3),
        "conciseness": round(conciseness, 3),
        "scs_overall": round(overall, 3),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
    }


def evaluate_explanations(alerts: pd.DataFrame) -> pd.DataFrame:
    """Return row-level SCS scores for generated explanations."""
    rows: list[dict[str, Any]] = []
    for index, alert in alerts.iterrows():
        rows.append(
            {
                "alert_id": alert.get("alert_id", str(index)),
                "attack_type": alert.get("attack_type"),
                **score_explanation(alert),
            }
        )
    return pd.DataFrame(rows)


def save_explanation_quality(
    scores: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    """Save row-level and summary explanation-quality artifacts."""
    directory = ensure_dir(output_dir)
    detailed_path = directory / "explanation_quality.csv"
    summary_path = directory / "explanation_quality_summary.json"
    scores.to_csv(detailed_path, index=False)
    metric_columns = [
        "completeness",
        "actionability",
        "conciseness",
        "scs_overall",
        "word_count",
        "sentence_count",
    ]
    summary = {
        "sample_count": len(scores),
        **{
            metric: round(float(scores[metric].mean()), 3)
            if metric in scores and not scores.empty
            else 0.0
            for metric in metric_columns
        },
    }
    save_json(summary, summary_path)
    return {"detailed": detailed_path, "summary": summary_path}


def main(argv: Iterable[str] | None = None) -> None:
    """Score explanations from an alerts CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--alerts", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    alerts = pd.read_csv(args.alerts)
    scores = evaluate_explanations(alerts)
    save_explanation_quality(scores, config.path("paths.metrics_dir"))
    print(scores.describe().to_string())


if __name__ == "__main__":
    main()
