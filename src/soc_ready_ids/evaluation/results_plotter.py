"""Publication-ready result plots for dissertation reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import normalize_dataset_name
from soc_ready_ids.evaluation.ids_metrics import collect_model_metrics
from soc_ready_ids.utils.io import ensure_dir

sns.set_theme(style="whitegrid", context="paper")


def save_figure(
    figure: plt.Figure, output_base: Path
) -> dict[str, Path]:
    """Save a figure as a 300-DPI PNG and vector PDF."""
    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {"png": png_path, "pdf": pdf_path}


def plot_model_comparison(
    metrics: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path] | None:
    """Plot accuracy, macro F1, and macro ROC-AUC across models."""
    if metrics.empty:
        return None
    directory = ensure_dir(output_dir)
    columns = [
        column
        for column in ("accuracy", "f1_macro", "roc_auc_macro_ovr")
        if column in metrics
    ]
    long = metrics.melt(
        id_vars=["model"],
        value_vars=columns,
        var_name="metric",
        value_name="score",
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=long,
        x="model",
        y="score",
        hue="metric",
        ax=axis,
    )
    axis.set_ylim(0, 1)
    axis.set_title("IDS Model Comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    return save_figure(figure, directory / "ids_model_comparison")


def plot_ablation(
    ablation_csv: str | Path, output_dir: str | Path
) -> dict[str, Path] | None:
    """Plot reduction and true-positive preservation by ablation scenario."""
    path = Path(ablation_csv)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    directory = ensure_dir(output_dir)
    long = frame.melt(
        id_vars=["scenario"],
        value_vars=[
            "alert_reduction_rate",
            "true_positive_preservation_rate",
        ],
        var_name="metric",
        value_name="percent",
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=long,
        x="scenario",
        y="percent",
        hue="metric",
        ax=axis,
    )
    axis.set_title("Triage Ablation Study")
    axis.set_ylabel("Percent")
    axis.set_xlabel("Scenario")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    return save_figure(
        figure, directory / "triage_ablation_comparison"
    )


def plot_triage_kpis(
    metrics_json: str | Path, output_dir: str | Path
) -> dict[str, Path] | None:
    """Plot core alert-reduction and safety KPIs."""
    path = Path(metrics_json)
    if not path.exists():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    labels = [
        "Alert reduction",
        "TP preservation",
        "Suppressed-attack FNR",
    ]
    values = [
        metrics.get("alert_reduction_rate", 0.0),
        metrics.get("true_positive_preservation_rate", 0.0),
        metrics.get(
            "false_negative_rate_suppressed_real_attacks", 0.0
        ),
    ]
    directory = ensure_dir(output_dir)
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(x=labels, y=values, color="#2a9d8f", ax=axis)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Percent")
    axis.set_title("Triage Performance and Safety")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    return save_figure(figure, directory / "triage_kpis")


def plot_explanation_quality(
    summary_json: str | Path, output_dir: str | Path
) -> dict[str, Path] | None:
    """Plot SCS explanation quality dimensions."""
    path = Path(summary_json)
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    labels = [
        "Completeness",
        "Actionability",
        "Conciseness",
        "Overall SCS",
    ]
    values = [
        summary.get("completeness", 0.0),
        summary.get("actionability", 0.0),
        summary.get("conciseness", 0.0),
        summary.get("scs_overall", 0.0),
    ]
    directory = ensure_dir(output_dir)
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(x=labels, y=values, color="#457b9d", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("SCS Explanation Quality")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    return save_figure(figure, directory / "explanation_quality_scs")


def plot_alert_volume(
    alerts_csv: str | Path, output_dir: str | Path
) -> dict[str, Path] | None:
    """Plot alert volume over time by predicted attack type."""
    path = Path(alerts_csv)
    if not path.exists():
        return None
    alerts = pd.read_csv(path)
    if alerts.empty or "timestamp" not in alerts:
        return None
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"], errors="coerce")
    volume = (
        alerts.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .groupby("attack_type")
        .resample("1min")
        .size()
        .rename("alerts")
        .reset_index()
    )
    directory = ensure_dir(output_dir)
    figure, axis = plt.subplots(figsize=(11, 5))
    sns.lineplot(
        data=volume,
        x="timestamp",
        y="alerts",
        hue="attack_type",
        ax=axis,
    )
    axis.set_title("Alert Volume Over Time")
    axis.set_xlabel("Time")
    axis.set_ylabel("Alerts per minute")
    figure.tight_layout()
    return save_figure(figure, directory / "alert_volume_timeseries")


def generate_all_plots(
    config_path: str = "config.yaml", dataset: str | None = None
) -> dict[str, object]:
    """Generate every available publication-ready figure."""
    config = load_config(config_path)
    selected_dataset = normalize_dataset_name(
        dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(selected_dataset)
    metrics_dir = config.path("paths.metrics_dir")
    figure_dir = config.path("paths.figure_dir")
    metrics = collect_model_metrics(metrics_dir)
    return {
        "model_comparison": plot_model_comparison(metrics, figure_dir),
        "ablation": plot_ablation(
            metrics_dir / "ablation_study.csv", figure_dir
        ),
        "triage": plot_triage_kpis(
            metrics_dir / "triage_metrics.json", figure_dir
        ),
        "explanation_quality": plot_explanation_quality(
            metrics_dir / "explanation_quality_summary.json", figure_dir
        ),
        "alert_volume": plot_alert_volume(
            metrics_dir / "triage_alerts_before.csv", figure_dir
        ),
    }


def main(argv: Iterable[str] | None = None) -> None:
    """Generate dissertation figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(list(argv) if argv is not None else None)
    generate_all_plots(args.config)


if __name__ == "__main__":
    main()
