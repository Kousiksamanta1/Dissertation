"""Feature selection using mutual information and recursive elimination."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.model_selection import train_test_split

from soc_ready_ids.config import load_config
from soc_ready_ids.data.loader import SUPPORTED_DATASETS, normalize_dataset_name
from soc_ready_ids.data.preprocessor import load_processed_arrays
from soc_ready_ids.utils.io import ensure_dir, save_json
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


def configured_top_k(config: object, dataset: str) -> int:
    """Return the dataset-specific configured feature count."""
    value = config.get("data.top_k_features", 20)  # type: ignore[attr-defined]
    if isinstance(value, dict):
        selected = value.get(dataset, value.get("default", 20))
        return int(selected)
    return int(value)


def sample_for_feature_selection(
    X: pd.DataFrame,
    y: np.ndarray,
    max_rows: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a deterministic stratified sample for scalable RFE fitting."""
    if max_rows is None or len(X) <= max_rows:
        return X, y
    sampled_X, _, sampled_y, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        random_state=random_state,
        stratify=y,
    )
    LOGGER.info(
        "Using a stratified %s-row sample for feature selection from %s rows",
        len(sampled_X),
        len(X),
    )
    return sampled_X, sampled_y


def select_features(
    X: pd.DataFrame,
    y: pd.Series | list[int] | np.ndarray,
    top_k: int = 20,
    random_state: int = 42,
    n_estimators: int = 120,
    rfe_step: float = 0.1,
) -> pd.DataFrame:
    """Rank features by combined mutual-information and RFE rank."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if X.empty:
        raise ValueError("Feature selection received an empty feature matrix")
    if len(np.unique(np.asarray(y))) < 2:
        raise ValueError("Feature selection requires at least two target classes")
    top_k = min(top_k, X.shape[1])

    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_rank = pd.Series(mi_scores, index=X.columns).rank(
        ascending=False, method="average"
    )

    estimator = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rfe = RFE(
        estimator=estimator,
        n_features_to_select=top_k,
        step=rfe_step,
    )
    rfe.fit(X, y)
    rfe_rank = pd.Series(rfe.ranking_, index=X.columns)

    combined = pd.DataFrame(
        {
            "feature": X.columns,
            "mutual_information": mi_scores,
            "mi_rank": mi_rank.values,
            "rfe_rank": rfe_rank.values,
            "rfe_selected": rfe.support_,
        }
    )
    combined["combined_rank"] = combined[["mi_rank", "rfe_rank"]].mean(
        axis=1
    )
    combined = combined.sort_values(
        ["combined_rank", "mutual_information"],
        ascending=[True, False],
    ).reset_index(drop=True)
    combined["selected"] = combined.index < top_k
    LOGGER.info(
        "Selected top-%s features: %s",
        top_k,
        combined.loc[combined["selected"], "feature"].tolist(),
    )
    return combined


def save_feature_selection(
    ranking: pd.DataFrame,
    figure_dir: str | Path,
    processed_dir: str | Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    top_k: int = 20,
) -> dict[str, Path]:
    """Persist rankings, selected matrices, and 300-DPI importance plots."""
    figures = ensure_dir(figure_dir)
    processed = ensure_dir(processed_dir)
    top = ranking.loc[ranking["selected"]].head(top_k).copy()
    plot_top = top.head(20).copy()
    selected_features = top["feature"].tolist()

    paths = {
        "json": figures / "top_features.json",
        "csv": figures / "feature_ranking.csv",
        "png": figures / "feature_importance_top20.png",
        "pdf": figures / "feature_importance_top20.pdf",
        "selected_columns": processed / "selected_feature_columns.json",
        "X_train_selected": processed / "X_train_selected.csv",
        "X_test_selected": processed / "X_test_selected.csv",
    }

    ranking.to_csv(paths["csv"], index=False)
    save_json(
        {
            "top_features": plot_top["feature"].tolist(),
            "selected_feature_count": len(selected_features),
        },
        paths["json"],
    )
    save_json(
        {"feature_columns": selected_features}, paths["selected_columns"]
    )
    X_train[selected_features].to_csv(
        paths["X_train_selected"], index=False
    )
    X_test[selected_features].to_csv(paths["X_test_selected"], index=False)

    figure, axis = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=plot_top,
        x="mutual_information",
        y="feature",
        color="#2a9d8f",
        ax=axis,
    )
    axis.set_title("Top IDS Features by Mutual Information and RFE")
    axis.set_xlabel("Mutual information")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(paths["png"], dpi=300, bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved feature-selection artifacts under %s", figures)
    return paths


def main(argv: Iterable[str] | None = None) -> None:
    """Select and persist the configured number of top IDS features."""
    parser = argparse.ArgumentParser(description="Select top IDS features.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", choices=list(SUPPORTED_DATASETS))
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    dataset = normalize_dataset_name(
        args.dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(dataset)
    processed_dir = config.path("paths.processed_data_dir")
    X_train, X_test, y_train, _, _, _ = load_processed_arrays(
        processed_dir,
        prefer_selected=False,
    )
    top_k = configured_top_k(config, dataset)
    random_state = int(config.get("project.random_state", 42))
    max_rows_value = config.get("feature_selection.max_rows", 100000)
    max_rows = int(max_rows_value) if max_rows_value is not None else None
    selection_X, selection_y = sample_for_feature_selection(
        X_train,
        y_train,
        max_rows=max_rows,
        random_state=random_state,
    )
    ranking = select_features(
        selection_X,
        selection_y,
        top_k=top_k,
        random_state=random_state,
        n_estimators=int(config.get("feature_selection.rfe_estimators", 120)),
        rfe_step=float(config.get("feature_selection.rfe_step", 0.1)),
    )
    save_feature_selection(
        ranking,
        config.path("paths.figure_dir"),
        processed_dir,
        X_train,
        X_test,
        top_k=top_k,
    )


if __name__ == "__main__":
    main()
