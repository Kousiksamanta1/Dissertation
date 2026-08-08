"""Tests for SHAP, LIME, and analyst-facing explanations."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from soc_ready_ids.explainability.explanation_generator import (
    canonical_attack_key,
    generate_explanation_payload,
    save_explanation,
)
from soc_ready_ids.explainability.lime_explainer import (
    LimeIDSExplainer,
    generate_lime_reports_by_alert_type,
)
from soc_ready_ids.explainability.shap_explainer import ShapIDSExplainer


def _artifact() -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Build a tiny fitted tree artifact."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        rng.normal(size=(30, 4)), columns=["duration", "rate", "syn", "bytes"]
    )
    y = (X["rate"] + X["syn"] > 0).astype(int).to_numpy()
    model = RandomForestClassifier(n_estimators=15, random_state=42).fit(X, y)
    return (
        {
            "model": model,
            "model_name": "test_tree",
            "class_names": ["BENIGN", "DDoS"],
            "feature_columns": list(X.columns),
            "explainability_type": "tree",
            "background_data": X.head(10).to_numpy(),
        },
        X,
        y,
    )


def test_explanation_payload_is_three_sentences(tmp_path: Path) -> None:
    """Generated explanations should contain context and a first action."""
    payload = generate_explanation_payload(
        "a1",
        "DDoS",
        0.92,
        88,
        [{"feature": "rate", "value": 10, "shap_value": 1.2}],
    )
    path = save_explanation(payload, tmp_path)

    assert canonical_attack_key("Reconnaissance") == "RECONNAISSANCE"
    assert canonical_attack_key("data theft") == "THEFT"
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])", payload["explanation_text"]
    )
    assert len(sentences) == 3
    assert "Recommended first response" in payload["explanation_text"]
    assert path.exists()


def test_tree_shap_global_and_local_outputs(tmp_path: Path) -> None:
    """Tree SHAP should create summary, importance, and waterfall files."""
    artifact, X, _ = _artifact()
    explainer = ShapIDSExplainer(
        artifact, tmp_path / "shap", background=X.head(10)
    )
    global_paths = explainer.save_global_plots(X, max_samples=12)
    local = explainer.local_explanation(
        X.iloc[0],
        "tree-alert",
        output_json_dir=tmp_path / "json",
        risk_score=70,
    )

    assert global_paths["summary"].exists()
    assert global_paths["bar"].with_suffix(".pdf").exists()
    assert Path(local["shap_plot_path"]).exists()
    assert len(local["top_features"]) == 3


def test_lime_report_generation(tmp_path: Path) -> None:
    """LIME should save one representative HTML report per class."""
    artifact, X, y = _artifact()
    explainer = LimeIDSExplainer(artifact, X)
    one = explainer.explain_row(X.iloc[0], tmp_path / "one.html", 3)
    reports = generate_lime_reports_by_alert_type(
        artifact,
        X.iloc[:20],
        X.iloc[20:].reset_index(drop=True),
        y[20:].tolist(),
        tmp_path / "reports",
        num_features=3,
    )

    assert one.exists()
    assert reports
    assert all(path.exists() for path in reports)
