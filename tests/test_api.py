"""Tests for Flask API endpoints."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from soc_ready_ids.api.app import create_app
from soc_ready_ids.config import load_config


def test_predict_alert_feedback_flow(tmp_path: Path) -> None:
    """Prediction should create an alert that can be listed, detailed, and reviewed."""
    project_config = load_config("config.yaml")
    values = copy.deepcopy(project_config.values)
    values["paths"]["processed_data_dir"] = str(project_config.path("paths.processed_data_dir"))
    values["paths"]["model_dir"] = str(project_config.path("paths.model_dir"))
    values["paths"]["shap_dir"] = str(tmp_path / "shap")
    values["paths"]["explanation_dir"] = str(tmp_path / "explanations")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")

    app = create_app(config_path, db_path=tmp_path / "alerts.db")
    client = app.test_client()

    index = client.get("/")
    assert index.status_code == 200
    assert index.get_json()["dataset"] == "cicids2017"
    assert index.get_json()["endpoints"]["predict"] == "POST /predict"

    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json() == {
        "status": "ok",
        "dataset": "cicids2017",
    }

    response = client.post(
        "/predict",
        json={
            "src_ip": "10.0.0.5",
            "dst_ip": "192.168.10.50",
            "Destination Port": 80,
            "Flow Packets/s": 120,
            "SYN Flag Count": 5,
            "asset_criticality": 80,
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["alert_id"]
    assert "explanation_text" in payload

    feed = client.get("/alerts").get_json()
    assert feed["alerts"]

    detail = client.get(f"/alert/{payload['alert_id']}")
    assert detail.status_code == 200
    assert detail.get_json()["alert_id"] == payload["alert_id"]

    feedback = client.post("/feedback", json={"alert_id": payload["alert_id"], "feedback": "true_positive"})
    assert feedback.status_code == 200

    stats = client.get("/stats")
    assert stats.status_code == 200
    assert "total_alerts" in stats.get_json()

    invalid_batch = client.post("/predict/batch", json={"flows": "invalid"})
    assert invalid_batch.status_code == 400

    batch = client.post(
        "/predict/batch",
        json={
            "flows": [
                {
                    "alert_id": f"batch-{index}",
                    "timestamp": f"2026-01-01T00:00:{index * 10:02d}",
                    "src_ip": "10.0.0.9",
                    "dst_ip": "192.168.1.2",
                    "Destination Port": 80,
                    "Flow Packets/s": 120,
                    "SYN Flag Count": 5,
                }
                for index in range(4)
            ]
        },
    )
    assert batch.status_code == 200
    assert batch.get_json()["alerts_before"] == 4
