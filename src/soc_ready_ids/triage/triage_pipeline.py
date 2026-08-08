"""End-to-end IDS prediction, explanation, risk scoring, and persistence."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from soc_ready_ids.config import ProjectConfig, load_config
from soc_ready_ids.data.harmonizer import harmonize_flow
from soc_ready_ids.explainability.explanation_generator import (
    generate_explanation_payload,
    save_explanation,
)
from soc_ready_ids.explainability.shap_explainer import ShapIDSExplainer
from soc_ready_ids.models.common import load_best_model, predict_proba_safe
from soc_ready_ids.triage.alert_clusterer import cluster_alerts
from soc_ready_ids.triage.deduplicator import (
    deduplicate_alerts,
    detect_alert_storm,
)
from soc_ready_ids.triage.risk_scorer import compute_risk_score
from soc_ready_ids.utils.io import load_joblib, load_json
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def utc_now_iso() -> str:
    """Return the current UTC timestamp as ISO text."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class TriagePipeline:
    """SOC-ready alert triage pipeline."""

    def __init__(
        self,
        config: ProjectConfig | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        """Load model/preprocessing artifacts and initialize SQLite."""
        self.config = config or load_config()
        self.db_path = (
            Path(db_path)
            if db_path
            else self.config.path("paths.sqlite_db")
        )
        self.model_artifact = self._load_model_artifact()
        self.feature_columns = self._model_feature_columns()
        self.input_feature_columns = self._input_feature_columns()
        self.scaler = self._load_scaler()
        self._shap_explainer: ShapIDSExplainer | None = None
        self.init_db()

    def _load_model_artifact(self) -> dict[str, Any] | None:
        """Load the selected model or enable deterministic demo fallback."""
        try:
            return load_best_model(self.config.path("paths.model_dir"))
        except Exception as exc:
            LOGGER.warning(
                "No trained best model available; using demo predictions: %s",
                exc,
            )
            return None

    def _model_feature_columns(self) -> list[str]:
        """Return the features expected by the selected model."""
        if self.model_artifact and self.model_artifact.get("feature_columns"):
            return list(self.model_artifact["feature_columns"])
        selected_path = (
            self.config.path("paths.processed_data_dir")
            / "selected_feature_columns.json"
        )
        if selected_path.exists():
            return list(load_json(selected_path).get("feature_columns", []))
        return self._input_feature_columns()

    def _input_feature_columns(self) -> list[str]:
        """Return the full encoded columns expected by the fitted scaler."""
        feature_path = (
            self.config.path("paths.processed_data_dir")
            / "feature_columns.json"
        )
        if feature_path.exists():
            return list(load_json(feature_path).get("feature_columns", []))
        return [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Flow Bytes/s",
            "Flow Packets/s",
            "SYN Flag Count",
            "ACK Flag Count",
            "Protocol",
        ]

    def _load_scaler(self) -> Any | None:
        """Load the training-fitted StandardScaler when available."""
        scaler_path = (
            self.config.path("paths.processed_data_dir") / "scaler.joblib"
        )
        if not scaler_path.exists():
            return None
        try:
            return load_joblib(scaler_path)
        except Exception as exc:
            LOGGER.warning("Could not load scaler: %s", exc)
            return None

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def init_db(self) -> None:
        """Create and migrate SQLite alert tables."""
        with self.connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    src_ip TEXT,
                    dst_ip TEXT,
                    attack_type TEXT,
                    confidence REAL,
                    risk_score REAL,
                    risk_tier TEXT,
                    cluster_id INTEGER,
                    cluster_x REAL,
                    cluster_y REAL,
                    explanation_text TEXT,
                    is_suppressed INTEGER,
                    duplicate_count INTEGER,
                    triage_duration_ms REAL,
                    ground_truth TEXT,
                    analyst_feedback TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_details (
                    alert_id TEXT PRIMARY KEY,
                    dst_port TEXT,
                    recommended_action TEXT,
                    shap_plot_path TEXT,
                    shap_values_json TEXT,
                    raw_flow_json TEXT,
                    FOREIGN KEY(alert_id) REFERENCES alerts(alert_id)
                )
                """
            )
            self._migrate_alert_columns(connection)

    @staticmethod
    def _migrate_alert_columns(connection: sqlite3.Connection) -> None:
        """Add new alert columns when opening an older database."""
        existing = {
            row[1] for row in connection.execute("PRAGMA table_info(alerts)")
        }
        migrations = {
            "cluster_x": "REAL",
            "cluster_y": "REAL",
            "duplicate_count": "INTEGER DEFAULT 1",
            "triage_duration_ms": "REAL DEFAULT 0",
            "ground_truth": "TEXT",
        }
        for column, definition in migrations.items():
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE alerts ADD COLUMN {column} {definition}"
                )

    def vectorize_flow(self, raw_flow: dict[str, Any]) -> pd.DataFrame:
        """Align, encode, scale, and select one raw flow for inference."""
        aligned_flow = raw_flow
        if str(self.config.get("data.dataset", "")) == "combined":
            aligned_flow = {
                **raw_flow,
                **harmonize_flow(
                    raw_flow,
                    source_dataset=str(
                        raw_flow.get("source_dataset", "")
                    ),
                ),
            }
        values: dict[str, Any] = {
            column: 0.0 for column in self.input_feature_columns
        }
        for column in self.input_feature_columns:
            if column in aligned_flow:
                values[column] = aligned_flow[column]
                continue
            if "=" in column:
                base, expected = column.split("=", 1)
                raw_value = str(aligned_flow.get(base, "missing"))
                values[column] = 1.0 if raw_value == expected else 0.0
        full_row = pd.DataFrame([values])
        full_row = full_row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        scaled = full_row
        if self.scaler is not None:
            try:
                scaled = pd.DataFrame(
                    self.scaler.transform(full_row),
                    columns=self.input_feature_columns,
                )
            except Exception as exc:
                LOGGER.warning(
                    "Could not scale incoming flow; using aligned values: %s",
                    exc,
                )
        return scaled.reindex(
            columns=self.feature_columns, fill_value=0.0
        )

    def predict(
        self, feature_row: pd.DataFrame
    ) -> tuple[str, float, int]:
        """Predict attack type, confidence, and encoded class."""
        if self.model_artifact and self.model_artifact.get("model") is not None:
            model = self.model_artifact["model"]
            class_names = list(
                self.model_artifact.get("class_names", [])
            )
            probabilities = predict_proba_safe(model, feature_row)
            if probabilities is not None:
                class_index = int(np.argmax(probabilities[0]))
                label = (
                    class_names[class_index]
                    if class_index < len(class_names)
                    else str(class_index)
                )
                return label, float(probabilities[0, class_index]), class_index
            class_index = int(model.predict(feature_row)[0])
            label = (
                class_names[class_index]
                if class_index < len(class_names)
                else str(class_index)
            )
            return label, 0.0, class_index
        return self.heuristic_predict(feature_row)

    def heuristic_predict(
        self, feature_row: pd.DataFrame
    ) -> tuple[str, float, int]:
        """Return a deterministic pre-training demo prediction."""
        row = feature_row.iloc[0]
        packets_per_second = abs(float(row.get("Flow Packets/s", 0.0)))
        syn_count = abs(float(row.get("SYN Flag Count", 0.0)))
        destination_port = int(
            abs(float(row.get("Destination Port", 0.0)))
        )
        if packets_per_second > 50 or syn_count > 3:
            return "DDoS", 0.78, 1
        if destination_port not in {0, 53, 80, 443} and syn_count > 0:
            return "PortScan", 0.68, 1
        return "BENIGN", 0.64, 0

    def explain(
        self,
        feature_row: pd.DataFrame,
        alert_id: str,
        class_index: int,
        risk_score: float,
        attack_type: str,
        confidence: float,
    ) -> dict[str, Any]:
        """Generate SHAP and natural-language explanation artifacts."""
        if self.model_artifact and self.model_artifact.get("model") is not None:
            try:
                if self._shap_explainer is None:
                    self._shap_explainer = ShapIDSExplainer(
                        self.model_artifact,
                        self.config.path("paths.shap_dir"),
                    )
                return self._shap_explainer.local_explanation(
                    feature_row.iloc[0],
                    alert_id,
                    predicted_class_index=class_index,
                    output_json_dir=self.config.path(
                        "paths.explanation_dir"
                    ),
                    risk_score=risk_score,
                )
            except Exception as exc:
                LOGGER.warning(
                    "SHAP local explanation failed; using fallback: %s", exc
                )
        top_features = []
        row = feature_row.iloc[0]
        for feature in row.abs().sort_values(ascending=False).head(3).index:
            top_features.append(
                {
                    "feature": feature,
                    "value": round(float(row[feature]), 4),
                    "shap_value": round(float(row[feature]), 6),
                }
            )
        payload = generate_explanation_payload(
            alert_id,
            attack_type,
            confidence,
            risk_score,
            top_features,
        )
        save_explanation(payload, self.config.path("paths.explanation_dir"))
        return payload

    def process_flow(self, raw_flow: dict[str, Any]) -> dict[str, Any]:
        """Run one network flow through IDS, risk, explanation, and storage."""
        started = time.perf_counter()
        alert_id = str(raw_flow.get("alert_id") or uuid.uuid4())
        timestamp_text = str(raw_flow.get("timestamp") or utc_now_iso())
        try:
            timestamp = datetime.fromisoformat(
                timestamp_text.replace("Z", "+00:00")
            ).replace(tzinfo=None)
        except ValueError:
            timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
            timestamp_text = timestamp.isoformat()

        feature_row = self.vectorize_flow(raw_flow)
        attack_type, confidence, class_index = self.predict(feature_row)
        risk = compute_risk_score(
            confidence,
            attack_type,
            asset_criticality=float(
                raw_flow.get("asset_criticality", 50.0)
            ),
            timestamp=timestamp,
            thresholds=self.config.get("triage", {}),
        )
        explanation = self.explain(
            feature_row,
            alert_id,
            class_index,
            risk.risk_score,
            attack_type,
            confidence,
        )
        duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
        alert = {
            "alert_id": alert_id,
            "timestamp": timestamp_text,
            "src_ip": str(
                raw_flow.get("src_ip")
                or raw_flow.get("Src IP")
                or raw_flow.get("saddr")
                or "unknown"
            ),
            "dst_ip": str(
                raw_flow.get("dst_ip")
                or raw_flow.get("Dst IP")
                or raw_flow.get("daddr")
                or "unknown"
            ),
            "dst_port": str(
                raw_flow.get("dst_port")
                or raw_flow.get("Destination Port")
                or raw_flow.get("dport")
                or ""
            ),
            "attack_type": attack_type,
            "confidence": round(float(confidence), 6),
            "risk_score": risk.risk_score,
            "risk_tier": risk.risk_tier,
            "risk_components": risk.components,
            "cluster_id": int(raw_flow.get("cluster_id", -1)),
            "cluster_x": 0.0,
            "cluster_y": 0.0,
            "explanation_text": explanation["explanation_text"],
            "recommended_action": explanation["recommended_action"],
            "top_features": explanation["top_features"],
            "shap_plot_path": explanation.get("shap_plot_path"),
            "is_suppressed": False,
            "duplicate_count": 1,
            "triage_duration_ms": duration_ms,
            "ground_truth": raw_flow.get("ground_truth")
            or raw_flow.get("Label")
            or raw_flow.get("category"),
            "analyst_feedback": None,
        }
        self.store_alert(alert, raw_flow)
        return alert

    def process_batch(
        self, raw_flows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Process, cluster, deduplicate, suppress, and persist an alert batch."""
        alerts: list[dict[str, Any]] = []
        vectors: list[pd.Series] = []
        for flow in raw_flows:
            alerts.append(self.process_flow(flow))
            vectors.append(self.vectorize_flow(flow).iloc[0])
        frame = pd.DataFrame(alerts)
        if frame.empty:
            storm = detect_alert_storm(frame)
            return {
                "alerts_before": 0,
                "alerts_after": 0,
                "alert_reduction_rate": 0.0,
                "storm": storm.__dict__,
                "alerts": [],
            }

        vector_frame = pd.DataFrame(vectors).reset_index(drop=True)
        cluster_input = pd.concat(
            [frame.reset_index(drop=True), vector_frame], axis=1
        )
        clustered = cluster_alerts(
            cluster_input,
            self.feature_columns,
            self.config.get("triage", {}),
        )
        cluster_visible = clustered[
            ~clustered["is_suppressed"].astype(bool)
        ].copy()
        deduplicated = deduplicate_alerts(
            cluster_visible,
            int(self.config.get("triage.duplicate_window_seconds", 60)),
            int(self.config.get("triage.duplicate_threshold", 3)),
        )
        visible_ids = set(deduplicated.get("alert_id", pd.Series(dtype=str)))
        clustered["is_suppressed"] = (
            clustered["is_suppressed"].astype(bool)
            | ~clustered["alert_id"].isin(visible_ids)
        )
        duplicate_counts = {
            str(row["alert_id"]): int(row.get("duplicate_count", 1))
            for _, row in deduplicated.iterrows()
        }
        clustered["duplicate_count"] = clustered["alert_id"].map(
            duplicate_counts
        ).fillna(1)
        storm = detect_alert_storm(
            clustered,
            int(self.config.get("triage.storm_alerts_per_minute", 50)),
        )
        self.update_batch_metadata(clustered)

        public_columns = [
            column
            for column in deduplicated.columns
            if column not in self.feature_columns
        ]
        visible = deduplicated[public_columns].copy()
        return {
            "alerts_before": len(frame),
            "alerts_after": len(visible),
            "alert_reduction_rate": round(
                ((len(frame) - len(visible)) / len(frame) * 100.0)
                if len(frame)
                else 0.0,
                2,
            ),
            "storm": storm.__dict__,
            "alerts": visible.to_dict("records"),
        }

    def store_alert(
        self, alert: dict[str, Any], raw_flow: dict[str, Any]
    ) -> None:
        """Insert or replace an alert and its detailed evidence."""
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO alerts (
                    alert_id, timestamp, src_ip, dst_ip, attack_type, confidence,
                    risk_score, risk_tier, cluster_id, cluster_x, cluster_y,
                    explanation_text, is_suppressed, duplicate_count,
                    triage_duration_ms, ground_truth, analyst_feedback
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert["alert_id"],
                    alert["timestamp"],
                    alert["src_ip"],
                    alert["dst_ip"],
                    alert["attack_type"],
                    float(alert["confidence"]),
                    float(alert["risk_score"]),
                    alert["risk_tier"],
                    int(alert.get("cluster_id", -1)),
                    float(alert.get("cluster_x", 0.0)),
                    float(alert.get("cluster_y", 0.0)),
                    alert["explanation_text"],
                    int(bool(alert.get("is_suppressed", False))),
                    int(alert.get("duplicate_count", 1)),
                    float(alert.get("triage_duration_ms", 0.0)),
                    alert.get("ground_truth"),
                    alert.get("analyst_feedback"),
                ),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO alert_details (
                    alert_id, dst_port, recommended_action, shap_plot_path,
                    shap_values_json, raw_flow_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    alert["alert_id"],
                    alert.get("dst_port", ""),
                    alert.get("recommended_action", ""),
                    alert.get("shap_plot_path", ""),
                    json.dumps(alert.get("top_features", []), default=str),
                    json.dumps(raw_flow, default=str),
                ),
            )

    def update_batch_metadata(self, alerts: pd.DataFrame) -> None:
        """Persist clustering, suppression, and duplicate metadata."""
        if alerts.empty:
            return
        with self.connect() as connection:
            for _, row in alerts.iterrows():
                connection.execute(
                    """
                    UPDATE alerts
                    SET cluster_id = ?, cluster_x = ?, cluster_y = ?,
                        is_suppressed = ?, duplicate_count = ?
                    WHERE alert_id = ?
                    """,
                    (
                        int(row.get("cluster_id", -1)),
                        float(row.get("cluster_x", 0.0)),
                        float(row.get("cluster_y", 0.0)),
                        int(bool(row.get("is_suppressed", False))),
                        int(row.get("duplicate_count", 1)),
                        row["alert_id"],
                    ),
                )

    def get_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        risk_tier: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return risk-sorted paginated alerts."""
        query = "SELECT * FROM alerts"
        parameters: list[Any] = []
        if risk_tier:
            query += " WHERE risk_tier = ?"
            parameters.append(risk_tier)
        query += " ORDER BY risk_score DESC, timestamp DESC LIMIT ? OFFSET ?"
        parameters.extend([limit, offset])
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            return [
                dict(row)
                for row in connection.execute(
                    query, parameters
                ).fetchall()
            ]

    def get_alert(self, alert_id: str) -> dict[str, Any] | None:
        """Return one alert with raw flow and feature attributions."""
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)
            ).fetchone()
            if row is None:
                return None
            detail = connection.execute(
                "SELECT * FROM alert_details WHERE alert_id = ?", (alert_id,)
            ).fetchone()
        payload = dict(row)
        if detail:
            detail_dict = dict(detail)
            payload.update(detail_dict)
            payload["shap_values"] = json.loads(
                detail_dict.get("shap_values_json") or "[]"
            )
            payload["raw_flow"] = json.loads(
                detail_dict.get("raw_flow_json") or "{}"
            )
        return payload

    def submit_feedback(self, alert_id: str, feedback: str) -> bool:
        """Store analyst feedback for an alert."""
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE alerts SET analyst_feedback = ? WHERE alert_id = ?",
                (feedback, alert_id),
            )
            return cursor.rowcount > 0

    def stats(self) -> dict[str, Any]:
        """Compute Wazuh/OpenSearch KPI values from SQLite."""
        with self.connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM alerts"
            ).fetchone()[0]
            suppressed = connection.execute(
                "SELECT COUNT(*) FROM alerts WHERE is_suppressed = 1"
            ).fetchone()[0]
            true_positive = connection.execute(
                """
                SELECT COUNT(*) FROM alerts
                WHERE analyst_feedback = 'true_positive'
                """
            ).fetchone()[0]
            false_positive = connection.execute(
                """
                SELECT COUNT(*) FROM alerts
                WHERE analyst_feedback = 'false_positive'
                """
            ).fetchone()[0]
            mean_triage = connection.execute(
                "SELECT AVG(triage_duration_ms) FROM alerts"
            ).fetchone()[0]
        reviewed = true_positive + false_positive
        return {
            "total_alerts": total,
            "suppressed_percent": round(
                (suppressed / total * 100.0) if total else 0.0, 2
            ),
            "true_positive_rate": round(
                (true_positive / reviewed * 100.0) if reviewed else 0.0,
                2,
            ),
            "mean_time_to_triage_ms": round(float(mean_triage or 0.0), 3),
        }


def main() -> None:
    """Run one demonstration flow through the triage pipeline."""
    pipeline = TriagePipeline()
    result = pipeline.process_flow(
        {
            "src_ip": "10.0.0.5",
            "dst_ip": "192.168.10.10",
            "Destination Port": 80,
            "Flow Packets/s": 120,
            "SYN Flag Count": 5,
            "asset_criticality": 80,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
