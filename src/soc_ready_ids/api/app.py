"""Flask API exposing prediction, alert feed, feedback, and KPI endpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS, normalize_dataset_name
from soc_ready_ids.triage.triage_pipeline import TriagePipeline
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def create_app(
    config_path: str | Path = "config.yaml",
    db_path: str | Path | None = None,
    dataset: str | None = None,
) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    config = load_config(config_path)
    selected_dataset = normalize_dataset_name(
        dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(selected_dataset)
    pipeline = TriagePipeline(config=config, db_path=db_path)

    @app.get("/")
    def index() -> tuple[Any, int]:
        """Return API status and the available endpoint map."""
        return jsonify(
            {
                "service": "SOC-Ready Explainable IDS Triage API",
                "status": "running",
                "dataset": selected_dataset,
                "endpoints": {
                    "health": "GET /health",
                    "predict": "POST /predict",
                    "predict_batch": "POST /predict/batch",
                    "alerts": "GET /alerts",
                    "alert_detail": "GET /alert/<alert_id>",
                    "feedback": "POST /feedback",
                    "stats": "GET /stats",
                },
            }
        ), 200

    @app.get("/health")
    def health() -> tuple[Any, int]:
        """Return a lightweight health response."""
        return jsonify(
            {
                "status": "ok",
                "dataset": selected_dataset,
            }
        ), 200

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        """
        Submit raw flow features and receive prediction plus explanation.
        ---
        tags:
          - prediction
        requestBody:
          required: true
          content:
            application/json:
              schema:
                type: object
        responses:
          200:
            description: IDS prediction, risk, and explanation JSON.
        """
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        try:
            result = pipeline.process_flow(payload)
            return jsonify(result), 200
        except Exception as exc:
            LOGGER.exception("Prediction failed: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.get("/alerts")
    def alerts() -> tuple[Any, int]:
        """
        Return paginated alert feed with optional risk-tier filter.
        ---
        tags:
          - alerts
        parameters:
          - in: query
            name: page
            schema: {type: integer, default: 1}
          - in: query
            name: page_size
            schema: {type: integer, default: 50}
          - in: query
            name: risk_tier
            schema: {type: string}
        responses:
          200:
            description: Alert feed page.
        """
        page = max(int(request.args.get("page", "1")), 1)
        page_size = min(max(int(request.args.get("page_size", str(config.get("api.page_size", 50)))), 1), 250)
        risk_tier = request.args.get("risk_tier")
        offset = (page - 1) * page_size
        return jsonify({"page": page, "page_size": page_size, "alerts": pipeline.get_alerts(page_size, offset, risk_tier)}), 200

    @app.post("/predict/batch")
    def predict_batch() -> tuple[Any, int]:
        """Submit a list of flows for clustering and deduplication."""
        payload = request.get_json(silent=True)
        flows = payload.get("flows") if isinstance(payload, dict) else payload
        if not isinstance(flows, list) or not all(
            isinstance(item, dict) for item in flows
        ):
            return jsonify(
                {"error": "Provide a JSON list of flow objects or {'flows': [...]}."}
            ), 400
        try:
            return jsonify(pipeline.process_batch(flows)), 200
        except Exception as exc:
            LOGGER.exception("Batch prediction failed: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.get("/alert/<alert_id>")
    def alert_detail(alert_id: str) -> tuple[Any, int]:
        """
        Return full alert detail including SHAP/top-feature values.
        ---
        tags:
          - alerts
        parameters:
          - in: path
            name: alert_id
            required: true
            schema: {type: string}
        responses:
          200:
            description: Full alert detail.
          404:
            description: Alert not found.
        """
        alert = pipeline.get_alert(alert_id)
        if alert is None:
            return jsonify({"error": "Alert not found."}), 404
        return jsonify(alert), 200

    @app.post("/feedback")
    def feedback() -> tuple[Any, int]:
        """
        Submit analyst true-positive/false-positive/needs-review feedback.
        ---
        tags:
          - feedback
        requestBody:
          required: true
          content:
            application/json:
              schema:
                type: object
                required: [alert_id, feedback]
        responses:
          200:
            description: Feedback stored.
          404:
            description: Alert not found.
        """
        payload = request.get_json(silent=True) or {}
        alert_id = str(payload.get("alert_id", ""))
        feedback_value = str(payload.get("feedback", ""))
        allowed = {"true_positive", "false_positive", "needs_review"}
        if not alert_id or feedback_value not in allowed:
            return jsonify({"error": f"Provide alert_id and feedback in {sorted(allowed)}."}), 400
        updated = pipeline.submit_feedback(alert_id, feedback_value)
        if not updated:
            return jsonify({"error": "Alert not found."}), 404
        return jsonify({"status": "ok", "alert_id": alert_id, "feedback": feedback_value}), 200

    @app.get("/stats")
    def stats() -> tuple[Any, int]:
        """
        Return triage KPI statistics.
        ---
        tags:
          - stats
        responses:
          200:
            description: Triage KPI object.
        """
        return jsonify(pipeline.stats()), 200

    return app


def main() -> None:
    """Run the Flask development server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS))
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = normalize_dataset_name(
        args.dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(dataset)
    app = create_app(args.config, dataset=dataset)
    app.run(host=str(config.get("api.host", "127.0.0.1")), port=int(config.get("api.port", 5000)), debug=False)


if __name__ == "__main__":
    main()
