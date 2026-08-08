"""Alert embedding, clustering, and representative selection."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def embed_alerts(alerts: pd.DataFrame, feature_columns: list[str], method: str = "numeric_pca") -> np.ndarray:
    """Embed alerts for clustering using sentence-transformers or PCA features."""
    if alerts.empty:
        return np.empty((0, 2))
    if method == "sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = alerts.apply(lambda row: " ".join(f"{col}={row.get(col)}" for col in feature_columns[:30]), axis=1).tolist()
            return np.asarray(model.encode(texts, show_progress_bar=False))
        except Exception as exc:  # pragma: no cover - optional dependency/network model cache
            LOGGER.warning("Sentence-transformer embedding failed; falling back to numeric PCA: %s", exc)

    numeric = alerts.reindex(columns=feature_columns, fill_value=0.0)
    numeric = numeric.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if numeric.shape[1] == 0:
        return np.zeros((len(alerts), 2))
    scaled = StandardScaler().fit_transform(numeric)
    if np.allclose(np.var(scaled, axis=0), 0.0):
        return np.zeros((len(alerts), 2))
    n_components = min(10, scaled.shape[1], max(1, scaled.shape[0] - 1))
    if n_components < 2:
        return np.column_stack([scaled[:, 0], np.zeros(len(alerts))]) if scaled.size else np.zeros((len(alerts), 2))
    return PCA(n_components=n_components, random_state=42).fit_transform(scaled)


def cluster_embeddings(embeddings: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """Cluster alert embeddings with HDBSCAN, falling back to DBSCAN."""
    if embeddings.shape[0] == 0:
        return np.array([], dtype=int)
    if embeddings.shape[0] < min_cluster_size:
        return np.full(embeddings.shape[0], -1, dtype=int)
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        return clusterer.fit_predict(embeddings)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("HDBSCAN unavailable or failed; falling back to DBSCAN: %s", exc)
        from sklearn.cluster import DBSCAN

        return DBSCAN(eps=1.5, min_samples=max(2, min_cluster_size // 2)).fit_predict(embeddings)


def mark_cluster_representatives(
    alerts: pd.DataFrame,
    cluster_window_minutes: int = 5,
    risk_column: str = "risk_score",
) -> pd.DataFrame:
    """Keep the top-risk representative per cluster within a sliding time window."""
    if alerts.empty:
        return alerts.copy()
    frame = alerts.copy()
    frame["timestamp"] = pd.to_datetime(frame.get("timestamp", pd.Timestamp.utcnow()), errors="coerce").fillna(pd.Timestamp.utcnow())
    if "cluster_id" not in frame.columns:
        frame["cluster_id"] = -1
    frame["is_representative"] = True
    frame["is_suppressed"] = False
    frame["cluster_size"] = frame.groupby("cluster_id")["cluster_id"].transform("size")

    for cluster_id, group in frame.groupby("cluster_id"):
        if cluster_id == -1:
            continue
        sorted_group = group.sort_values("timestamp")
        used_indices: set[int] = set()
        for idx, row in sorted_group.iterrows():
            if idx in used_indices:
                continue
            window_end = row["timestamp"] + timedelta(minutes=cluster_window_minutes)
            window = sorted_group[(sorted_group["timestamp"] >= row["timestamp"]) & (sorted_group["timestamp"] <= window_end)]
            representative_idx = window[risk_column].astype(float).idxmax()
            suppressed = set(window.index) - {representative_idx}
            member_ids = ",".join(
                str(value) for value in window.get("alert_id", window.index)
            )
            frame.loc[representative_idx, "merged_alert_ids"] = member_ids
            frame.loc[representative_idx, "cluster_size"] = len(window)
            frame.loc[list(suppressed), "is_representative"] = False
            frame.loc[list(suppressed), "is_suppressed"] = True
            used_indices.update(window.index)
    return frame


def cluster_alerts(alerts: pd.DataFrame, feature_columns: list[str], triage_config: dict[str, Any]) -> pd.DataFrame:
    """Embed, cluster, and suppress duplicate-cluster alerts."""
    frame = alerts.copy()
    method = str(triage_config.get("embedding_method", "numeric_pca"))
    embeddings = embed_alerts(frame, feature_columns, method=method)
    labels = cluster_embeddings(embeddings, int(triage_config.get("hdbscan_min_cluster_size", 5)))
    frame["cluster_id"] = labels
    frame["cluster_x"] = embeddings[:, 0] if embeddings.shape[1] >= 1 else 0.0
    frame["cluster_y"] = embeddings[:, 1] if embeddings.shape[1] >= 2 else 0.0
    return mark_cluster_representatives(frame, int(triage_config.get("cluster_window_minutes", 5)))


def save_cluster_map(alerts: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a 2D cluster scatter plot for reports."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 7))
    if "cluster_x" in alerts.columns and "cluster_y" in alerts.columns:
        sns.scatterplot(data=alerts, x="cluster_x", y="cluster_y", hue="attack_type", style="is_suppressed", alpha=0.8)
    plt.title("Alert Cluster Map")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.savefig(path.with_suffix(".pdf"), dpi=300)
    plt.close()
    return path
