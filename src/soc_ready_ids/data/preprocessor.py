"""Leakage-safe preprocessing for CICIDS2017 and BoT-IoT."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from soc_ready_ids.config import ProjectConfig, load_config
from soc_ready_ids.data.loader import (
    SUPPORTED_DATASETS,
    load_dataset,
    normalize_columns,
    normalize_dataset_name,
)
from soc_ready_ids.utils.io import (
    ensure_dir,
    load_joblib,
    save_joblib,
    save_json,
)
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PreprocessArtifacts:
    """In-memory result of one preprocessing run."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    y_train_labels: pd.Series
    y_test_labels: pd.Series
    metadata_train: pd.DataFrame
    metadata_test: pd.DataFrame
    feature_columns: list[str]
    scaler: StandardScaler
    label_encoder: LabelEncoder
    dataset_name: str
    input_rows: int
    retained_rows: int


def infer_label_column(
    df: pd.DataFrame, configured_label: str = "Label"
) -> str:
    """Find a target column despite case and whitespace differences."""
    columns = {column.lower(): column for column in df.columns}
    candidates = (
        configured_label,
        "Label",
        "label",
        "category",
        "attack_type",
        "class",
        "target",
        "attack",
        "subcategory",
    )
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    raise KeyError(
        f"No label column found. Available columns: {list(df.columns)}"
    )


def clean_label(value: object) -> str:
    """Normalize common CICIDS2017 and BoT-IoT attack labels."""
    text = " ".join(
        str(value).strip().strip("\"'").replace("_", " ").split()
    )
    normalized = text.upper()
    if normalized in {"BENIGN", "NORMAL", "0", "FALSE"}:
        return "BENIGN"
    canonical = {
        "DDOS": "DDoS",
        "DOS": "DoS",
        "RECONNAISSANCE": "Reconnaissance",
        "RECON": "Reconnaissance",
        "THEFT": "Theft",
        "1": "ATTACK",
        "TRUE": "ATTACK",
    }
    return canonical.get(normalized, text)


def configured_label_column(
    config: ProjectConfig, dataset_name: str
) -> str:
    """Return the configured label column for a specific dataset."""
    configured = config.get("data.label_columns", {})
    if isinstance(configured, dict):
        value = configured.get(dataset_name)
        if value:
            return str(value)
    return str(config.get("data.label_column", "Label"))


def _valid_stratify_target(y: np.ndarray) -> np.ndarray | None:
    """Return y when every class has enough rows for stratification."""
    values, counts = np.unique(y, return_counts=True)
    return y if len(values) > 1 and counts.min() >= 2 else None


def _present_columns(
    df: pd.DataFrame, requested_columns: Iterable[str]
) -> list[str]:
    """Resolve configured column names case-insensitively."""
    lookup = {column.lower(): column for column in df.columns}
    resolved: list[str] = []
    for requested in requested_columns:
        actual = lookup.get(str(requested).lower())
        if actual is not None and actual not in resolved:
            resolved.append(actual)
    return resolved


def split_metadata(
    df: pd.DataFrame, metadata_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate identifiers and provenance from candidate model features."""
    present = _present_columns(df, metadata_columns)
    metadata = (
        df[present].copy() if present else pd.DataFrame(index=df.index)
    )
    features = df.drop(columns=present, errors="ignore")
    return features, metadata


def _coerce_numeric_like_columns(
    features: pd.DataFrame, threshold: float = 0.98
) -> pd.DataFrame:
    """Convert object columns to numeric when nearly all values are numeric."""
    converted = features.copy()
    for column in converted.select_dtypes(
        include=["object", "string"]
    ).columns:
        values = converted[column].astype("string").str.strip()
        numeric = pd.to_numeric(values, errors="coerce")
        non_null = values.notna().sum()
        numeric_ratio = (
            float(numeric.notna().sum()) / float(non_null)
            if non_null
            else 0.0
        )
        if numeric_ratio >= threshold:
            converted[column] = numeric
    return converted


def build_feature_matrix(
    df: pd.DataFrame,
    label_column: str,
    metadata_columns: list[str],
    drop_columns: list[str] | None = None,
    *,
    drop_invalid_rows: bool = True,
    numeric_detection_threshold: float = 0.98,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Clean invalid rows and create a numeric, leakage-safe feature matrix."""
    working = normalize_columns(df.copy())
    input_rows = len(working)
    label_column = infer_label_column(working, label_column)

    valid_label = working[label_column].notna()
    working = working.loc[valid_label].copy()
    y_labels = working[label_column].map(clean_label)

    excluded = _present_columns(
        working, [label_column, *(drop_columns or [])]
    )
    working = working.drop(columns=excluded, errors="ignore")
    features, metadata = split_metadata(working, metadata_columns)
    features = _coerce_numeric_like_columns(
        features, threshold=numeric_detection_threshold
    )
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")

    if drop_invalid_rows:
        valid_rows = ~features.isna().any(axis=1)
        dropped = int((~valid_rows).sum())
        if dropped:
            LOGGER.info(
                "Dropped %s rows containing NaN or infinite feature values",
                dropped,
            )
        features = features.loc[valid_rows]
        y_labels = y_labels.loc[valid_rows]
        metadata = metadata.loc[valid_rows]

    numeric_parts: list[pd.DataFrame] = []
    numeric = features.select_dtypes(include=[np.number, "bool"]).copy()
    if not numeric.empty:
        numeric_parts.append(numeric.astype(float))

    categorical = features.select_dtypes(
        exclude=[np.number, "bool"]
    ).copy()
    if not categorical.empty:
        categorical = categorical.fillna("missing").astype(str)
        numeric_parts.append(
            pd.get_dummies(categorical, prefix_sep="=", dtype=float)
        )

    if not numeric_parts:
        raise ValueError("No usable feature columns remained after preprocessing.")

    X = pd.concat(numeric_parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.empty:
        raise ValueError("All feature columns were constant after cleaning.")
    if X.columns.duplicated().any():
        X = X.loc[:, ~X.columns.duplicated()].copy()

    LOGGER.info(
        "Feature matrix retained %s/%s rows and %s encoded features",
        len(X),
        input_rows,
        X.shape[1],
    )
    return X.astype(float), y_labels, metadata


def _dataset_drop_columns(
    config: ProjectConfig, dataset_name: str
) -> list[str]:
    """Return label-derived columns that must not enter model features."""
    configured = config.get("data.drop_columns", {})
    if isinstance(configured, dict):
        values = configured.get(dataset_name, [])
        return [str(value) for value in values]
    if isinstance(configured, list):
        return [str(value) for value in configured]
    return []


def _drop_invalid_rows(
    config: ProjectConfig, dataset_name: str
) -> bool:
    """Resolve dataset-specific missing-value handling."""
    configured = config.get("data.drop_invalid_rows", True)
    if isinstance(configured, dict):
        return bool(configured.get(dataset_name, True))
    return bool(configured)


def preprocess_dataframe(
    df: pd.DataFrame,
    config: ProjectConfig,
    dataset_name: str | None = None,
) -> PreprocessArtifacts:
    """Preprocess a dataframe into stratified, scaled 80/20 splits."""
    canonical = normalize_dataset_name(
        dataset_name or str(config.get("data.dataset", "cicids2017"))
    )
    label_column = configured_label_column(config, canonical)
    metadata_columns = list(config.get("data.metadata_columns", []))
    X, y_labels, metadata = build_feature_matrix(
        df,
        label_column,
        metadata_columns,
        drop_columns=_dataset_drop_columns(config, canonical),
        drop_invalid_rows=_drop_invalid_rows(config, canonical),
        numeric_detection_threshold=float(
            config.get("data.numeric_detection_threshold", 0.98)
        ),
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    if len(label_encoder.classes_) < 2:
        raise ValueError(
            "Preprocessing requires at least two target classes after cleaning."
        )

    stratify = _valid_stratify_target(y)
    test_size = float(config.get("data.test_size", 0.2))
    random_state = int(config.get("project.random_state", 42))
    indices = X.index.to_numpy()
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify is not None else None,
    )

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_by_index = pd.Series(y, index=X.index)
    y_train = y_by_index.loc[train_indices].to_numpy()
    y_test = y_by_index.loc[test_indices].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index,
    )

    artifacts = PreprocessArtifacts(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        y_train_labels=y_labels.loc[train_indices].reset_index(drop=True),
        y_test_labels=y_labels.loc[test_indices].reset_index(drop=True),
        metadata_train=metadata.loc[train_indices].reset_index(drop=True),
        metadata_test=metadata.loc[test_indices].reset_index(drop=True),
        feature_columns=list(X.columns),
        scaler=scaler,
        label_encoder=label_encoder,
        dataset_name=canonical,
        input_rows=len(df),
        retained_rows=len(X),
    )
    LOGGER.info(
        "Preprocessed %s: train=%s test=%s classes=%s",
        canonical,
        X_train_scaled.shape,
        X_test_scaled.shape,
        list(label_encoder.classes_),
    )
    return artifacts


def save_preprocessed(
    artifacts: PreprocessArtifacts, output_dir: str | Path
) -> dict[str, Path]:
    """Persist processed splits, metadata, transformers, and a run manifest."""
    directory = ensure_dir(output_dir)
    paths = {
        "X_train": directory / "X_train.csv",
        "X_test": directory / "X_test.csv",
        "y_train": directory / "y_train.csv",
        "y_test": directory / "y_test.csv",
        "metadata_train": directory / "metadata_train.csv",
        "metadata_test": directory / "metadata_test.csv",
        "scaler": directory / "scaler.joblib",
        "label_encoder": directory / "label_encoder.joblib",
        "feature_columns": directory / "feature_columns.json",
        "manifest": directory / "preprocessing_manifest.json",
    }
    artifacts.X_train.to_csv(paths["X_train"], index=False)
    artifacts.X_test.to_csv(paths["X_test"], index=False)
    pd.DataFrame(
        {
            "label_encoded": artifacts.y_train,
            "label": artifacts.y_train_labels,
        }
    ).to_csv(paths["y_train"], index=False)
    pd.DataFrame(
        {
            "label_encoded": artifacts.y_test,
            "label": artifacts.y_test_labels,
        }
    ).to_csv(paths["y_test"], index=False)
    artifacts.metadata_train.to_csv(paths["metadata_train"], index=False)
    artifacts.metadata_test.to_csv(paths["metadata_test"], index=False)
    save_joblib(artifacts.scaler, paths["scaler"])
    save_joblib(artifacts.label_encoder, paths["label_encoder"])
    save_json(
        {"feature_columns": artifacts.feature_columns},
        paths["feature_columns"],
    )
    metadata_all = pd.concat(
        [artifacts.metadata_train, artifacts.metadata_test],
        ignore_index=True,
    )
    source_distribution: dict[str, int] = {}
    train_source_distribution: dict[str, int] = {}
    test_source_distribution: dict[str, int] = {}
    if "source_dataset" in metadata_all.columns:
        source_distribution = {
            str(name): int(count)
            for name, count in metadata_all[
                "source_dataset"
            ].value_counts().items()
        }
        train_source_distribution = {
            str(name): int(count)
            for name, count in artifacts.metadata_train[
                "source_dataset"
            ].value_counts().items()
        }
        test_source_distribution = {
            str(name): int(count)
            for name, count in artifacts.metadata_test[
                "source_dataset"
            ].value_counts().items()
        }
    save_json(
        {
            "dataset": artifacts.dataset_name,
            "input_rows": artifacts.input_rows,
            "retained_rows": artifacts.retained_rows,
            "train_rows": len(artifacts.X_train),
            "test_rows": len(artifacts.X_test),
            "feature_count": len(artifacts.feature_columns),
            "classes": artifacts.label_encoder.classes_.tolist(),
            "class_distribution": {
                str(label): int(count)
                for label, count in pd.concat(
                    [artifacts.y_train_labels, artifacts.y_test_labels]
                ).value_counts().items()
            },
            "train_class_distribution": {
                str(label): int(count)
                for label, count in artifacts.y_train_labels.value_counts().items()
            },
            "test_class_distribution": {
                str(label): int(count)
                for label, count in artifacts.y_test_labels.value_counts().items()
            },
            "source_distribution": source_distribution,
            "train_source_distribution": train_source_distribution,
            "test_source_distribution": test_source_distribution,
        },
        paths["manifest"],
    )
    LOGGER.info("Saved preprocessed artifacts under %s", directory)
    return paths


def load_processed_arrays(
    processed_dir: str | Path,
    prefer_selected: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    LabelEncoder,
    list[str],
]:
    """Load preprocessed train/test arrays and encoders from disk."""
    directory = Path(processed_dir)
    selected_train = directory / "X_train_selected.csv"
    selected_test = directory / "X_test_selected.csv"
    use_selected = prefer_selected and selected_train.exists() and selected_test.exists()
    X_train = pd.read_csv(selected_train if use_selected else directory / "X_train.csv")
    X_test = pd.read_csv(selected_test if use_selected else directory / "X_test.csv")
    y_train = pd.read_csv(directory / "y_train.csv")[
        "label_encoded"
    ].to_numpy()
    y_test = pd.read_csv(directory / "y_test.csv")[
        "label_encoded"
    ].to_numpy()
    label_encoder = load_joblib(directory / "label_encoder.joblib")
    feature_columns = list(X_train.columns)
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        feature_columns,
    )


def preprocess_from_config(
    config: ProjectConfig,
    dataset: str | None = None,
    source: str | Path | None = None,
    max_rows: int | None = None,
) -> PreprocessArtifacts:
    """Load one configured raw dataset and save all processed artifacts."""
    canonical = normalize_dataset_name(
        dataset or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(canonical)
    frame = load_dataset(
        config,
        dataset=canonical,
        source=source,
        max_rows=max_rows,
    )
    artifacts = preprocess_dataframe(frame, config, dataset_name=canonical)
    save_preprocessed(artifacts, config.path("paths.processed_data_dir"))
    return artifacts


def main(argv: Iterable[str] | None = None) -> None:
    """Run preprocessing for CICIDS2017 or BoT-IoT."""
    parser = argparse.ArgumentParser(description="Preprocess IDS data.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--input", help="Optional dataset file or directory override"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row cap overriding data.max_rows for this run",
    )
    parser.add_argument(
        "--dataset",
        choices=list(SUPPORTED_DATASETS),
        help="Override data.dataset from config.yaml",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    preprocess_from_config(
        config,
        dataset=args.dataset,
        source=args.input,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
