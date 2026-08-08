"""Focused coverage for defensive branches and fallback paths."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from soc_ready_ids.config import ProjectConfig


class FailingProbabilityModel:
    """Estimator whose probability path is unavailable."""

    def predict_proba(self, X):
        raise RuntimeError("probability unavailable")


class PredictOnlyModel:
    """Estimator with only hard-label prediction."""

    def predict(self, X):
        return np.array([2] * len(X))


class ArtifactModel:
    """Small picklable model for artifact round-trips."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _config(tmp_path: Path, dataset: str = "cicids2017") -> ProjectConfig:
    """Return a minimal config for path-driven tests."""
    return ProjectConfig(
        values={
            "project": {"random_state": 42},
            "paths": {
                "raw_data_dir": str(tmp_path / "raw"),
                "cicids2017_raw_dir": str(tmp_path / "raw" / "cicids"),
                "bot_iot_raw_dir": str(tmp_path / "raw" / "bot"),
                "processed_data_dir": "processed/{dataset}",
                "model_dir": "models/{dataset}",
                "metrics_dir": "metrics/{dataset}",
                "figure_dir": "figures/{dataset}",
                "shap_dir": "shap/{dataset}",
                "explanation_dir": "explanations/{dataset}",
                "sqlite_db": "runtime/{dataset}/alerts.db",
            },
            "data": {
                "dataset": dataset,
                "test_size": 0.25,
                "metadata_columns": [],
                "drop_columns": {},
                "drop_invalid_rows": True,
            },
            "triage": {
                "duplicate_window_seconds": 60,
                "duplicate_threshold": 3,
                "hdbscan_min_cluster_size": 2,
                "storm_alerts_per_minute": 50,
            },
        },
        root=tmp_path,
    )


def test_loader_error_paths_and_combined_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dataset loading should fail clearly and route combined modes."""
    import soc_ready_ids.data.loader as loader

    with pytest.raises(ValueError, match="Unsupported dataset"):
        loader.normalize_dataset_name("unknown")

    duplicate = pd.DataFrame([[1, 2]], columns=[" Label ", "\ufeff Label "])
    assert list(loader.normalize_columns(duplicate).columns) == ["Label"]

    with pytest.raises(FileNotFoundError):
        loader.iter_dataset_files(tmp_path / "missing", {".csv"})
    unsupported = tmp_path / "data.txt"
    unsupported.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected one of"):
        loader.iter_dataset_files(unsupported, {".csv"})
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No supported dataset files"):
        loader.iter_dataset_files(empty_dir, {".csv"})

    bad_cicids = tmp_path / "bad_cicids.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(bad_cicids, index=False)
    with pytest.raises(KeyError, match="Label"):
        loader.load_cicids2017(bad_cicids)

    bad_bot = tmp_path / "bad_bot.csv"
    pd.DataFrame({"feature": [1, 2]}).to_csv(bad_bot, index=False)
    with pytest.raises(KeyError, match="BoT-IoT"):
        loader.load_bot_iot(bad_bot)

    monkeypatch.setattr(
        loader.pd,
        "read_csv",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("broken csv")),
    )
    with pytest.raises(ValueError, match="Could not read"):
        loader.read_csv_with_encoding(tmp_path / "unreadable.csv")

    monkeypatch.setattr(
        loader.pd,
        "read_parquet",
        lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("no pyarrow")),
    )
    with pytest.raises(RuntimeError, match="pyarrow"):
        loader.read_parquet_file(tmp_path / "flows.parquet")

    config = _config(tmp_path, "combined")
    with pytest.raises(ValueError, match="combined dataset"):
        loader.dataset_raw_path(config, "combined")
    fallback = ProjectConfig(
        values={"paths": {"raw_data_dir": "raw"}, "data": {"dataset": "cicids2017"}},
        root=tmp_path,
    )
    assert loader.dataset_raw_path(fallback, "bot-iot") == tmp_path / "raw" / "bot-iot"

    cicids = pd.DataFrame({"Label": ["BENIGN", "DDoS"]})
    bot = pd.DataFrame({"category": ["Normal", "DoS"]})
    row_caps: list[tuple[str, int | None]] = []
    monkeypatch.setattr(
        loader,
        "load_cicids2017",
        lambda path, max_rows=None: row_caps.append(("cicids", max_rows)) or cicids,
    )
    monkeypatch.setattr(
        loader,
        "load_bot_iot",
        lambda path, max_rows=None: row_caps.append(("bot", max_rows)) or bot,
    )
    monkeypatch.setattr(
        loader,
        "harmonize_datasets",
        lambda left, right: pd.DataFrame(
            {"source_dataset": ["cicids2017", "bot-iot"], "common_label": ["BENIGN", "DoS"]}
        ),
    )
    config.values["data"]["max_rows"] = {"combined": 5}
    config.values["data"]["combined_feature_mode"] = "shared"
    combined = loader.load_dataset(config, "combined")
    assert combined["source_dataset"].tolist() == ["cicids2017", "bot-iot"]
    assert row_caps == [("cicids", 3), ("bot", 2)]

    with pytest.raises(ValueError, match="--input is not supported"):
        loader.load_dataset(config, "combined", source=tmp_path)
    config.values["data"]["combined_feature_mode"] = "invalid"
    with pytest.raises(ValueError, match="Unsupported data.combined_feature_mode"):
        loader.load_dataset(config, "combined")


def test_preprocessor_edge_paths_and_selected_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preprocessing should handle invalid matrices and selected feature reloads."""
    import soc_ready_ids.data.preprocessor as preprocessor
    from soc_ready_ids.data.preprocessor import (
        build_feature_matrix,
        clean_label,
        load_processed_arrays,
        preprocess_dataframe,
        save_preprocessed,
    )

    assert clean_label("true") == "ATTACK"
    assert clean_label("recon") == "Reconnaissance"
    with pytest.raises(KeyError, match="No label column"):
        preprocessor.infer_label_column(pd.DataFrame({"x": [1]}), "missing")
    assert preprocessor.configured_label_column(_config(tmp_path), "cicids2017") == "Label"
    assert preprocessor._dataset_drop_columns(
        ProjectConfig({"data": {"drop_columns": ["a", "b"]}}, tmp_path), "cicids2017"
    ) == ["a", "b"]

    numeric_strings = pd.DataFrame(
        {
            "feature": ["1", "2", "3"],
            "category": ["x", "y", "x"],
            "Label": ["BENIGN", "DDoS", "BENIGN"],
        }
    )
    X, _, _ = build_feature_matrix(
        numeric_strings,
        "Label",
        [],
        drop_invalid_rows=False,
        numeric_detection_threshold=0.6,
    )
    assert "feature" in X.columns
    assert "category=x" in X.columns

    with pytest.raises(ValueError, match="No usable feature"):
        build_feature_matrix(
            pd.DataFrame({"id": ["a", "b"], "Label": ["BENIGN", "DDoS"]}),
            "Label",
            ["id"],
        )
    with pytest.raises(ValueError, match="constant"):
        build_feature_matrix(
            pd.DataFrame({"feature": [1, 1, 1, 1], "Label": ["BENIGN", "DDoS", "BENIGN", "DDoS"]}),
            "Label",
            [],
        )

    frame = pd.DataFrame(
        {
            "feature": range(12),
            "source_dataset": ["cicids2017", "bot-iot"] * 6,
            "Label": ["BENIGN"] * 6 + ["DDoS"] * 6,
        }
    )
    config = _config(tmp_path)
    config.values["data"]["metadata_columns"] = ["source_dataset"]
    artifacts = preprocess_dataframe(frame, config)
    paths = save_preprocessed(artifacts, tmp_path / "processed")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["source_distribution"] == {"cicids2017": 6, "bot-iot": 6}

    artifacts.X_train[["feature"]].to_csv(
        tmp_path / "processed" / "X_train_selected.csv", index=False
    )
    artifacts.X_test[["feature"]].to_csv(
        tmp_path / "processed" / "X_test_selected.csv", index=False
    )
    X_train, X_test, _, _, _, feature_columns = load_processed_arrays(tmp_path / "processed")
    assert list(X_train.columns) == ["feature"]
    assert list(X_test.columns) == ["feature"]
    assert feature_columns == ["feature"]

    with pytest.raises(ValueError, match="at least two target classes"):
        preprocess_dataframe(
            pd.DataFrame({"feature": [1, 2, 3], "Label": ["BENIGN"] * 3}),
            config,
        )

    raw_frame = pd.DataFrame(
        {"feature": range(8), "Label": ["BENIGN"] * 4 + ["DDoS"] * 4}
    )
    config.values["paths"]["processed_data_dir"] = "from_config/{dataset}"
    monkeypatch.setattr(preprocessor, "load_dataset", lambda *args, **kwargs: raw_frame)
    loaded_artifacts = preprocessor.preprocess_from_config(config, "cicids2017")
    assert loaded_artifacts.retained_rows == 8
    assert (tmp_path / "from_config" / "cicids2017" / "X_train.csv").exists()


def test_shap_helpers_and_fallback_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SHAP utility branches should handle list, 3D, deep, and fallback paths."""
    import soc_ready_ids.explainability.shap_explainer as shap_module
    from soc_ready_ids.explainability.shap_explainer import ShapIDSExplainer

    values_list = [np.array([[1.0, -2.0]]), np.array([[3.0, 4.0]])]
    values_3d = np.dstack(values_list)
    assert shap_module._mean_abs_shap(values_list).tolist() == [2.0, 3.0]
    assert shap_module._mean_abs_shap(values_3d).tolist() == [2.0, 3.0]
    assert shap_module._class_shap_values(values_list, 4, 2).tolist() == [[3.0, 4.0]]
    assert shap_module._class_shap_values(values_3d, 1, 2).tolist() == [[3.0, 4.0]]
    assert shap_module._class_shap_values(np.array([1.0, 2.0]), 0, 2).tolist() == [
        [-1.0, -2.0]
    ]
    assert shap_module._class_base_value([], 0, 2) == 0.0
    assert shap_module._class_base_value([0.5], 0, 2) == -0.5
    assert shap_module._class_base_value([0.2, 0.8], 9, 2) == 0.8

    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(TreeExplainer=object))
    explainer = ShapIDSExplainer(
        {
            "model": None,
            "model_name": "none",
            "class_names": ["BENIGN", "DDoS"],
            "feature_columns": ["f1", "f2"],
        },
        tmp_path,
    )
    with pytest.raises(ValueError, match="does not contain a model"):
        explainer._build_explainer()

    deep = ShapIDSExplainer(
        {
            "model": object(),
            "model_name": "deep",
            "class_names": ["BENIGN", "ATTACK"],
            "feature_columns": ["f1"],
            "explainability_type": "deep",
        },
        tmp_path,
    )
    with pytest.raises(ValueError, match="background data"):
        deep._build_explainer()

    class LocalModel:
        def predict(self, X):
            return np.array([1])

    local = ShapIDSExplainer(
        {
            "model": LocalModel(),
            "model_name": "local",
            "class_names": ["BENIGN", "DDoS"],
            "feature_columns": ["f1", "f2"],
            "background_data": [[0.0, 1.0], [2.0, 3.0]],
        },
        tmp_path,
    )
    monkeypatch.setattr(local, "_shap_values", lambda frame: (_ for _ in ()).throw(RuntimeError("bad")))
    monkeypatch.setattr(
        local,
        "_save_local_plot",
        lambda row, values, base, output: Path(output).write_text("plot", encoding="utf-8")
        or Path(output),
    )
    payload = local.local_explanation(
        pd.Series({"f1": 3.0, "f2": 1.0}),
        "fallback-alert",
        output_json_dir=tmp_path / "json",
    )
    assert payload["alert_id"] == "fallback-alert"
    assert payload["top_features"][0]["feature"] == "f1"

    fake_shap = SimpleNamespace(
        Explanation=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
        plots=SimpleNamespace(waterfall=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(sys.modules, "shap", fake_shap)
    plotter = ShapIDSExplainer(
        {
            "model": LocalModel(),
            "model_name": "plotter",
            "class_names": ["BENIGN", "DDoS"],
            "feature_columns": ["f1", "f2"],
        },
        tmp_path,
    )
    path = plotter._save_local_plot(
        pd.Series({"f1": 1.0, "f2": -2.0}),
        np.array([0.3, -0.7]),
        0.1,
        tmp_path / "fallback_plot.png",
    )
    assert path.exists()
    assert path.with_suffix(".pdf").exists()


def test_common_model_artifacts_and_probability_fallbacks(tmp_path: Path) -> None:
    """Model utilities should support no-probability and artifact branches."""
    from soc_ready_ids.models.common import (
        evaluate_predictions,
        load_best_model,
        predict_proba_safe,
        save_detection_fpr_curve,
        save_model_artifact,
    )

    assert predict_proba_safe(FailingProbabilityModel(), pd.DataFrame({"f1": [1.0]})) is None
    curve = save_detection_fpr_curve(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 0]),
        ["BENIGN", "DDoS"],
        tmp_path / "fallback_curve.png",
        y_proba=None,
    )
    assert curve.exists()

    metrics = evaluate_predictions(
        [0, 0, 0],
        [0, 0, 0],
        ["BENIGN", "DDoS"],
        tmp_path / "metrics",
        "single_class",
        np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]),
    )
    assert metrics["roc_auc_per_class"] == {"BENIGN": None, "DDoS": None}
    assert metrics["roc_auc_macro_ovr"] is None
    assert metrics["binary_attack_roc_auc"] is None

    model_dir = tmp_path / "models"
    artifact_path = save_model_artifact(
        ArtifactModel(),
        "artifact_model",
        model_dir,
        ["BENIGN"],
        ["f1"],
        {"f1_macro": 1.0},
        extra={"explainability_type": "tree"},
    )
    (model_dir / "best_model.json").write_text(
        json.dumps({"artifact_path": artifact_path.name}), encoding="utf-8"
    )
    loaded = load_best_model(model_dir)
    assert loaded["model_name"] == "artifact_model"
    assert loaded["explainability_type"] == "tree"

    with pytest.raises(FileNotFoundError):
        load_best_model(tmp_path / "missing_models")


def test_triage_pipeline_fallbacks_and_database_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime pipeline should handle fallback prediction and storage."""
    import soc_ready_ids.triage.triage_pipeline as pipeline_module
    from soc_ready_ids.triage.triage_pipeline import TriagePipeline

    config = _config(tmp_path, "combined")
    processed = config.path("paths.processed_data_dir")
    processed.mkdir(parents=True)
    (processed / "selected_feature_columns.json").write_text(
        json.dumps({"feature_columns": ["proto=tcp", "Flow Packets/s"]}),
        encoding="utf-8",
    )
    (processed / "feature_columns.json").write_text(
        json.dumps(
            {
                "feature_columns": [
                    "proto=tcp",
                    "Flow Packets/s",
                    "SYN Flag Count",
                    "Destination Port",
                ]
            }
        ),
        encoding="utf-8",
    )
    (processed / "scaler.joblib").write_text("bad scaler", encoding="utf-8")

    monkeypatch.setattr(
        pipeline_module,
        "load_best_model",
        lambda path: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_joblib",
        lambda path: (_ for _ in ()).throw(RuntimeError("bad scaler")),
    )
    pipeline = TriagePipeline(config, db_path=tmp_path / "alerts.db")
    assert pipeline.model_artifact is None
    assert pipeline.feature_columns == ["proto=tcp", "Flow Packets/s"]
    assert pipeline.process_batch([])["alerts_before"] == 0
    pipeline.update_batch_metadata(pd.DataFrame())
    assert pipeline.get_alert("missing") is None
    assert pipeline.heuristic_predict(
        pd.DataFrame([{"Destination Port": 22, "SYN Flag Count": 1, "Flow Packets/s": 1}])
    )[0] == "PortScan"

    pipeline.scaler = SimpleNamespace(
        transform=lambda frame: (_ for _ in ()).throw(RuntimeError("scale failed"))
    )
    row = pipeline.vectorize_flow({"source_dataset": "bot-iot", "proto": "tcp"})
    assert row.loc[0, "proto=tcp"] == 1.0

    monkeypatch.setattr(
        pipeline_module,
        "load_best_model",
        lambda path: {
            "model": PredictOnlyModel(),
            "class_names": ["BENIGN"],
            "feature_columns": ["proto=tcp", "Flow Packets/s"],
        },
    )

    class FailingShap:
        def __init__(self, *args, **kwargs):
            pass

        def local_explanation(self, *args, **kwargs):
            raise RuntimeError("xai failed")

    monkeypatch.setattr(pipeline_module, "ShapIDSExplainer", FailingShap)
    model_pipeline = TriagePipeline(config, db_path=tmp_path / "model_alerts.db")
    alert = model_pipeline.process_flow(
        {
            "alert_id": "a1",
            "timestamp": "not-a-date",
            "source_dataset": "bot-iot",
            "proto": "tcp",
            "src_ip": "10.0.0.1",
            "dst_ip": "192.168.0.1",
            "dst_port": 22,
            "asset_criticality": 70,
            "ground_truth": "DDoS",
        }
    )
    assert alert["attack_type"] == "2"
    assert "Recommended first response" in alert["explanation_text"]
    assert model_pipeline.get_alert("a1")["raw_flow"]["alert_id"] == "a1"
    assert model_pipeline.get_alerts(risk_tier=alert["risk_tier"])[0]["alert_id"] == "a1"
    assert model_pipeline.submit_feedback("a1", "true_positive") is True
    assert model_pipeline.submit_feedback("missing", "false_positive") is False
    stats = model_pipeline.stats()
    assert stats["total_alerts"] == 1
    assert stats["true_positive_rate"] == 100.0

    legacy_db = tmp_path / "legacy.db"
    with sqlite3.connect(legacy_db) as connection:
        connection.execute(
            "CREATE TABLE alerts (alert_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL)"
        )
    TriagePipeline._migrate_alert_columns(sqlite3.connect(legacy_db))
    with sqlite3.connect(legacy_db) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(alerts)")}
    assert {"cluster_x", "duplicate_count", "ground_truth"}.issubset(columns)


def test_metric_plot_quality_and_cluster_edge_paths(tmp_path: Path) -> None:
    """Small edge cases should return deterministic metrics or no-op outputs."""
    from soc_ready_ids.evaluation.explanation_quality import (
        _feature_names,
        save_explanation_quality,
        score_explanation,
    )
    from soc_ready_ids.evaluation.results_plotter import (
        generate_all_plots,
        plot_ablation,
        plot_alert_volume,
        plot_explanation_quality,
        plot_model_comparison,
        plot_triage_kpis,
    )
    from soc_ready_ids.evaluation.triage_metrics import (
        feature_coverage,
        mean_explanation_length,
        true_positive_preservation_rate,
    )
    from soc_ready_ids.triage.alert_clusterer import (
        cluster_alerts,
        cluster_embeddings,
        embed_alerts,
        mark_cluster_representatives,
    )

    assert _feature_names("f1, f2") == ["f1", "f2"]
    assert _feature_names({"feature": "f1"}) == []
    empty_score = score_explanation(pd.Series({"explanation_text": ""}))
    assert empty_score["conciseness"] == 0.075
    quality_paths = save_explanation_quality(pd.DataFrame(), tmp_path / "quality")
    assert json.loads(quality_paths["summary"].read_text())["sample_count"] == 0

    before = pd.DataFrame({"ground_truth": ["BENIGN", "DDoS", "DDoS"]})
    after = pd.DataFrame({"ground_truth": ["DDoS"]})
    assert true_positive_preservation_rate(pd.DataFrame(), after) == 0.0
    assert true_positive_preservation_rate(pd.DataFrame({"ground_truth": ["BENIGN"]}), after) == 100.0
    assert true_positive_preservation_rate(before, pd.DataFrame()) == 0.0
    assert true_positive_preservation_rate(before, after) == 50.0
    assert mean_explanation_length(pd.DataFrame()) == 0.0
    assert feature_coverage(pd.DataFrame()) == 0.0
    assert feature_coverage(
        pd.DataFrame(
            {
                "top_features": ['[{"feature": "f1"}]', "bad json", [{"feature": "f2"}]],
                "explanation_text": ["f1 caused the alert", "no feature", "f2 present"],
            }
        )
    ) == 0.667

    assert plot_model_comparison(pd.DataFrame(), tmp_path) is None
    assert plot_ablation(tmp_path / "missing.csv", tmp_path) is None
    assert plot_triage_kpis(tmp_path / "missing.json", tmp_path) is None
    assert plot_explanation_quality(tmp_path / "missing.json", tmp_path) is None
    assert plot_alert_volume(tmp_path / "missing.csv", tmp_path) is None
    empty_alerts = tmp_path / "empty_alerts.csv"
    pd.DataFrame(columns=["timestamp", "attack_type"]).to_csv(empty_alerts, index=False)
    assert plot_alert_volume(empty_alerts, tmp_path) is None

    config_values = {
        "paths": {
            "metrics_dir": "metrics/{dataset}",
            "figure_dir": "figures/{dataset}",
        },
        "data": {"dataset": "cicids2017"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_values), encoding="utf-8")
    outputs = generate_all_plots(str(config_path), "cicids2017")
    assert set(outputs) == {
        "model_comparison",
        "ablation",
        "triage",
        "explanation_quality",
        "alert_volume",
    }

    assert embed_alerts(pd.DataFrame(), ["f1"]).shape == (0, 2)
    assert embed_alerts(pd.DataFrame({"x": [1, 2]}), []).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert cluster_embeddings(np.empty((0, 2))).tolist() == []
    assert cluster_embeddings(np.ones((2, 2)), min_cluster_size=3).tolist() == [-1, -1]
    marked_empty = mark_cluster_representatives(pd.DataFrame(), 5)
    assert marked_empty.empty
    clustered = cluster_alerts(
        pd.DataFrame(
            {
                "alert_id": ["a1", "a2"],
                "timestamp": pd.date_range("2026-01-01", periods=2, freq="s"),
                "attack_type": ["DDoS", "DDoS"],
                "risk_score": [10, 20],
                "f1": [1.0, 2.0],
            }
        ),
        ["f1"],
        {"hdbscan_min_cluster_size": 3},
    )
    assert "cluster_x" in clustered.columns
