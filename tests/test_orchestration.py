"""Tests for training/evaluation orchestration and generated summaries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from soc_ready_ids.config import ProjectConfig


def _write_config(tmp_path: Path, dataset: str = "combined") -> Path:
    """Write a minimal project config with templated artifact paths."""
    values = {
        "project": {"random_state": 42},
        "paths": {
            "processed_data_dir": "data/processed/{dataset}",
            "model_dir": "models/saved/{dataset}",
            "metrics_dir": "results/{dataset}/metrics",
            "figure_dir": "results/{dataset}/figures",
            "report_dir": "reports/{dataset}",
            "shap_dir": "results/{dataset}/shap",
            "lime_dir": "results/{dataset}/lime",
            "explanation_dir": "results/{dataset}/explanations",
            "sqlite_db": "data/runtime/{dataset}/alerts.db",
        },
        "data": {"dataset": dataset},
        "models": {
            "best_model_metric": "f1_macro",
            "random_forest": {"n_estimators": 10},
            "xgboost": {
                "optuna_trials": 1,
                "n_estimators": 10,
                "timeout_seconds": 5,
            },
            "autoencoder": {"epochs": 1, "batch_size": 4},
        },
        "triage": {
            "duplicate_window_seconds": 60,
            "duplicate_threshold": 3,
            "hdbscan_min_cluster_size": 2,
        },
        "evaluation": {"max_base_alerts": 2, "duplicate_repetitions": 2},
        "explainability": {
            "max_background_samples": 2,
            "max_explanation_samples": 2,
            "lime_num_features": 2,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return config_path


def _project_config(tmp_path: Path, dataset: str = "combined") -> ProjectConfig:
    """Return a ProjectConfig matching the test YAML structure."""
    return ProjectConfig(
        values=yaml.safe_load(_write_config(tmp_path, dataset).read_text()),
        root=tmp_path,
    )


def _write_dataset_artifacts(tmp_path: Path, dataset: str) -> None:
    """Create the small JSON/CSV files consumed by report writers."""
    processed = tmp_path / "data" / "processed" / dataset
    model_dir = tmp_path / "models" / "saved" / dataset
    metrics = tmp_path / "results" / dataset / "metrics"
    processed.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    preprocessing = {
        "dataset": dataset,
        "retained_rows": 12,
        "feature_count": 4,
        "classes": ["BENIGN", "DDoS"],
        "class_distribution": {"BENIGN": 6, "DDoS": 6},
    }
    if dataset == "combined":
        preprocessing["source_distribution"] = {"cicids2017": 6, "bot-iot": 6}
        pd.DataFrame({"source_dataset": ["bot-iot", "cicids2017", "bot-iot"]}).to_csv(
            processed / "metadata_test.csv", index=False
        )
        pd.DataFrame({"y_true": [0, 1, 1], "y_pred": [0, 1, 0]}).to_csv(
            metrics / "xgboost_ids_predictions.csv", index=False
        )
    (processed / "preprocessing_manifest.json").write_text(
        json.dumps(preprocessing), encoding="utf-8"
    )
    (model_dir / "best_model.json").write_text(
        json.dumps(
            {
                "best_model": "xgboost_ids",
                "selection_metric": "f1_macro",
                "selection_score": 0.97,
                "all_results": [
                    {
                        "model": "xgboost_ids",
                        "metrics": {
                            "accuracy": 0.98,
                            "f1_macro": 0.97,
                            "binary_attack_f1_macro": 0.99,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (metrics / "triage_metrics.json").write_text(
        json.dumps(
            {
                "alerts_before": 12,
                "alerts_after": 4,
                "alert_reduction_rate": 66.67,
                "true_positive_preservation_rate": 100.0,
                "false_negative_rate_suppressed_real_attacks": 0.0,
                "mean_explanation_length": 21,
                "feature_coverage": 0.9,
                "evaluation_design": "unit-test",
            }
        ),
        encoding="utf-8",
    )
    (metrics / "explanation_quality_summary.json").write_text(
        json.dumps(
            {
                "completeness": 0.9,
                "actionability": 0.8,
                "conciseness": 1.0,
                "scs_overall": 0.9,
                "sample_count": 4,
            }
        ),
        encoding="utf-8",
    )
    (metrics / "ids_model_comparison.md").write_text("| model | f1 |\n", encoding="utf-8")
    (metrics / "ablation_study.md").write_text("| scenario | score |\n", encoding="utf-8")


def test_dataset_summary_collects_and_writes_completed_runs(tmp_path: Path) -> None:
    """Top-level summary should include only completed dataset runs."""
    from soc_ready_ids.evaluation.dataset_summary import (
        collect_dataset_results,
        write_combined_summary,
    )

    config_path = _write_config(tmp_path)
    _write_dataset_artifacts(tmp_path, "cicids2017")
    _write_dataset_artifacts(tmp_path, "combined")

    table, details = collect_dataset_results(str(config_path))
    output = write_combined_summary(str(config_path))

    assert set(table["dataset"]) == {"cicids2017", "combined"}
    assert details["combined"]["preprocessing"]["source_distribution"]["bot-iot"] == 6
    text = output.read_text(encoding="utf-8")
    assert "Combined Real-Dataset Results" in text
    assert "Source distribution: cicids2017=6, bot-iot=6" in text


def test_dataset_summary_rejects_empty_project(tmp_path: Path) -> None:
    """An empty project should fail clearly instead of writing a blank report."""
    from soc_ready_ids.evaluation.dataset_summary import write_combined_summary

    with pytest.raises(RuntimeError, match="No completed dataset runs"):
        write_combined_summary(str(_write_config(tmp_path)))


def test_source_metrics_section_and_label_alignment(tmp_path: Path) -> None:
    """Combined reports should persist source-specific held-out metrics."""
    from soc_ready_ids.evaluation.run_all import (
        _labels_for_artifact,
        _source_metrics_section,
    )

    processed = tmp_path / "processed"
    metrics = tmp_path / "metrics"
    processed.mkdir()
    metrics.mkdir()
    pd.DataFrame({"source_dataset": ["bot-iot", "cicids2017", "bot-iot"]}).to_csv(
        processed / "metadata_test.csv", index=False
    )
    pd.DataFrame({"y_true": [0, 1, 1], "y_pred": [0, 1, 0]}).to_csv(
        metrics / "best_predictions.csv", index=False
    )

    section = _source_metrics_section("combined", processed, metrics, "best")

    assert "Unified Model by Source Dataset" in section
    assert (metrics / "source_dataset_metrics.csv").exists()
    assert _source_metrics_section("cicids2017", processed, metrics, "best") == ""
    assert _labels_for_artifact(
        np.array([0, 1, 2]), ["BENIGN", "DDoS", "Recon"], ["BENIGN", "ATTACK"]
    ) == [0, 1, 1]
    assert _labels_for_artifact(
        np.array([0, 1]), ["BENIGN", "DDoS"], ["BENIGN", "DDoS"]
    ) == [0, 1]

    bad_metadata = tmp_path / "bad_metadata"
    bad_metadata.mkdir()
    pd.DataFrame({"source": ["bot-iot"]}).to_csv(
        bad_metadata / "metadata_test.csv", index=False
    )
    assert _source_metrics_section("combined", bad_metadata, metrics, "best") == ""


def test_write_results_summary_uses_optional_tables(tmp_path: Path) -> None:
    """Dataset report should include model, triage, quality, and source metrics."""
    from soc_ready_ids.evaluation.run_all import write_results_summary

    config_path = _write_config(tmp_path)
    _write_dataset_artifacts(tmp_path, "combined")

    report = write_results_summary(str(config_path), "combined")

    text = report.read_text(encoding="utf-8")
    assert "selected downstream model is **xgboost_ids**" in text
    assert "Unified Model by Source Dataset" in text
    assert "Overall SCS: 0.9" in text


def test_run_all_evaluations_orchestrates_without_heavy_xai(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluation orchestration should connect every phase and save a manifest."""
    import soc_ready_ids.evaluation.run_all as run_all

    config_path = _write_config(tmp_path, "cicids2017")
    config = _project_config(tmp_path, "cicids2017")
    metrics_dir = config.path("paths.metrics_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": object(),
        "class_names": ["BENIGN", "ATTACK"],
        "feature_columns": ["f1", "f2"],
    }
    X_train = pd.DataFrame({"f1": [0.0, 1.0], "f2": [1.0, 0.0]})
    X_test = pd.DataFrame({"f1": [0.5], "f2": [0.2]})
    y_test = np.array([1])
    label_encoder = SimpleNamespace(classes_=np.array(["BENIGN", "DDoS"]))
    calls: list[str] = []

    class FakeShap:
        def __init__(self, artifact_arg, output_dir, background):
            assert artifact_arg is artifact
            assert list(background.columns) == ["f1", "f2"]

        def save_global_plots(self, X, max_samples):
            calls.append(f"shap:{max_samples}:{len(X)}")
            path = tmp_path / "summary.png"
            path.write_text("plot", encoding="utf-8")
            return {"summary": path}

        def local_explanation(self, row, alert_id, output_json_dir):
            calls.append(alert_id)
            return {"alert_id": alert_id, "top_features": []}

    monkeypatch.setattr(run_all, "collect_model_metrics", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        run_all,
        "save_metric_table",
        lambda table, path: {"csv": tmp_path / "metrics.csv"},
    )
    monkeypatch.setattr(run_all, "load_best_model", lambda path: artifact)
    monkeypatch.setattr(
        run_all,
        "load_processed_arrays",
        lambda path: (X_train, X_test, np.array([0, 1]), y_test, label_encoder, ["f1", "f2"]),
    )
    monkeypatch.setattr(run_all, "ShapIDSExplainer", FakeShap)
    monkeypatch.setattr(
        run_all,
        "generate_lime_reports_by_alert_type",
        lambda *args: [tmp_path / "lime.html"],
    )
    monkeypatch.setattr(
        run_all,
        "evaluate_triage_pipeline",
        lambda config_arg: (
            pd.DataFrame({"alert_id": ["a1"], "f1": [1.0], "f2": [2.0]}),
            pd.DataFrame(
                {
                    "alert_id": ["a1"],
                    "top_features": [[]],
                    "explanation_text": ["Recommended first response."],
                    "recommended_action": ["Review"],
                }
            ),
            {"alert_reduction_rate": 0.0},
        ),
    )
    monkeypatch.setattr(
        run_all,
        "evaluate_explanations",
        lambda after: pd.DataFrame({"scs_overall": [1.0]}),
    )
    monkeypatch.setattr(
        run_all,
        "save_explanation_quality",
        lambda scores, path: {"summary": tmp_path / "quality.json"},
    )
    monkeypatch.setattr(
        run_all,
        "run_ablation",
        lambda before, features, triage: pd.DataFrame({"scenario": ["full_pipeline"]}),
    )
    monkeypatch.setattr(
        run_all,
        "save_ablation_results",
        lambda ablation, path: {"csv": tmp_path / "ablation.csv"},
    )
    monkeypatch.setattr(
        run_all,
        "generate_all_plots",
        lambda config_path_arg, dataset: {"figure": str(tmp_path / "figure.png")},
    )
    monkeypatch.setattr(
        run_all,
        "write_results_summary",
        lambda config_path_arg, dataset: tmp_path / "report.md",
    )

    manifest = run_all.run_all_evaluations(str(config_path), "cicids2017")

    assert manifest["lime_reports"] == [str(tmp_path / "lime.html")]
    assert manifest["triage_metrics"]["alert_reduction_rate"] == 0.0
    assert "evaluation-example-alert" in calls
    saved = json.loads((metrics_dir / "evaluation_manifest.json").read_text())
    assert saved["report"] == str(tmp_path / "report.md")


def test_train_all_selects_best_and_records_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training orchestrator should pick the best successful model."""
    import soc_ready_ids.models.train_all as trainer

    config_path = _write_config(tmp_path, "cicids2017")
    config = _project_config(tmp_path, "cicids2017")
    model_dir = config.path("paths.model_dir")
    X_train = pd.DataFrame({"f1": [0.0, 1.0], "f2": [1.0, 0.0]})
    X_test = pd.DataFrame({"f1": [0.5], "f2": [0.2]})
    label_encoder = SimpleNamespace(classes_=np.array(["BENIGN", "DDoS"]))
    received_estimators: list[int] = []

    monkeypatch.setattr(
        trainer,
        "load_processed_arrays",
        lambda path: (
            X_train,
            X_test,
            np.array([0, 1]),
            np.array([1]),
            label_encoder,
            ["f1", "f2"],
        ),
    )

    def fake_rf(*args):
        received_estimators.append(args[6]["models"]["random_forest"]["n_estimators"])
        return object(), {"f1_macro": 0.8, "accuracy": 0.8}

    def fake_xgb(*args):
        return object(), {"f1_macro": 0.92, "accuracy": 0.9}

    monkeypatch.setattr(trainer, "train_random_forest", fake_rf)
    monkeypatch.setattr(trainer, "train_xgboost", fake_xgb)
    monkeypatch.setattr(
        trainer,
        "train_autoencoder",
        lambda *args: SimpleNamespace(metrics={"f1_macro": 0.7, "accuracy": 0.7}),
    )

    manifest = trainer.train_all(
        str(config_path),
        ["random_forest", "xgboost_ids"],
        smoke=True,
        dataset="cicids2017",
    )

    assert manifest["best_model"] == "xgboost_ids"
    assert manifest["smoke_mode"] is True
    assert received_estimators == [80]
    assert (model_dir / "best_model.json").exists()

    monkeypatch.setattr(
        trainer,
        "train_random_forest",
        lambda *args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="Required model training failed"):
        trainer.train_all(
            str(config_path),
            ["random_forest"],
            strict=True,
            dataset="cicids2017",
        )
    assert (model_dir / "training_failures.json").exists()

    with pytest.raises(RuntimeError, match="No model trained successfully"):
        trainer.train_all(
            str(config_path),
            ["random_forest"],
            strict=False,
            dataset="cicids2017",
        )


def test_build_evaluation_alerts_uses_model_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Held-out alert builder should duplicate predictions with explanations."""
    import soc_ready_ids.evaluation.pipeline_evaluator as evaluator

    config = _project_config(tmp_path, "combined")
    processed = config.path("paths.processed_data_dir")
    processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Src IP": ["10.1.1.1", "10.1.1.2"],
            "Dst IP": ["192.168.0.1", "192.168.0.2"],
        }
    ).to_csv(processed / "metadata_test.csv", index=False)
    X_test = pd.DataFrame({"f1": [2.0, -1.0], "f2": [0.5, 3.0]})
    label_encoder = SimpleNamespace(classes_=np.array(["BENIGN", "DDoS"]))

    class FakeModel:
        def predict(self, X):
            return np.array([0, 1])

    monkeypatch.setattr(
        evaluator,
        "load_processed_arrays",
        lambda path: (
            pd.DataFrame({"f1": [0.0], "f2": [1.0]}),
            X_test,
            np.array([0]),
            np.array([0, 1]),
            label_encoder,
            ["f1", "f2"],
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "load_best_model",
        lambda path: {
            "model": FakeModel(),
            "class_names": ["BENIGN", "DDoS"],
            "feature_columns": ["f1", "f2"],
        },
    )
    monkeypatch.setattr(
        evaluator,
        "predict_proba_safe",
        lambda model, X: np.array([[0.9, 0.1], [0.2, 0.8]]),
    )

    alerts, feature_columns = evaluator.build_evaluation_alerts(config)

    assert feature_columns == ["f1", "f2"]
    assert len(alerts) == 4
    assert alerts["incident_id"].nunique() == 2
    assert alerts.iloc[0]["src_ip"] == "10.1.1.1"
    assert alerts.iloc[2]["attack_type"] == "DDoS"
    assert alerts.iloc[2]["top_features"][0]["feature"] == "f2"
    assert "Recommended first response" in alerts.iloc[0]["explanation_text"]

