"""Dataset loading utilities for CICIDS2017, BoT-IoT, and combined data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from soc_ready_ids.config import ProjectConfig, load_config
from soc_ready_ids.data.harmonizer import (
    COMMON_FEATURES,
    common_attack_label,
    harmonize_datasets,
)
from soc_ready_ids.utils.logging import get_logger

LOGGER = get_logger(__name__)

SUPPORTED_DATASETS: tuple[str, ...] = (
    "cicids2017",
    "bot-iot",
    "combined",
)

DATASET_ALIASES: dict[str, str] = {
    "cicids": "cicids2017",
    "cicids-2017": "cicids2017",
    "cicids2017": "cicids2017",
    "bot-iot": "bot-iot",
    "bot_iot": "bot-iot",
    "botiot": "bot-iot",
    "combined": "combined",
    "unified": "combined",
    "merged": "combined",
}

# The official BoT-IoT CSV export contains 46 columns. Some mirrors omit the
# header row, so these names are used only when that format is detected.
BOT_IOT_COLUMNS: list[str] = [
    "pkSeqID",
    "stime",
    "flgs",
    "flgs_number",
    "proto",
    "proto_number",
    "saddr",
    "sport",
    "daddr",
    "dport",
    "pkts",
    "bytes",
    "state",
    "state_number",
    "ltime",
    "seq",
    "dur",
    "mean",
    "stddev",
    "sum",
    "min",
    "max",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "srate",
    "drate",
    "TnBPSrcIP",
    "TnBPDstIP",
    "TnP_PSrcIP",
    "TnP_PDstIP",
    "TnP_PerProto",
    "TnP_Per_Dport",
    "AR_P_Proto_P_SrcIP",
    "AR_P_Proto_P_DstIP",
    "N_IN_Conn_P_DstIP",
    "N_IN_Conn_P_SrcIP",
    "AR_P_Proto_P_Sport",
    "AR_P_Proto_P_Dport",
    "Pkts_P_State_P_Protocol_P_DestIP",
    "Pkts_P_State_P_Protocol_P_SrcIP",
    "attack",
    "category",
    "subcategory",
]


def normalize_dataset_name(dataset: str) -> str:
    """Return the canonical configured dataset name."""
    normalized = dataset.strip().lower()
    if normalized not in DATASET_ALIASES:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Choose one of: {', '.join(SUPPORTED_DATASETS)}"
        )
    return DATASET_ALIASES[normalized]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip byte-order marks, quotes, and repeated spaces from column names."""
    renamed = {
        column: " ".join(
            str(column).replace("\ufeff", "").strip().strip("\"'").split()
        )
        for column in df.columns
    }
    normalized = df.rename(columns=renamed)
    if normalized.columns.duplicated().any():
        duplicate_names = normalized.columns[normalized.columns.duplicated()].tolist()
        LOGGER.warning("Dropping duplicate columns after normalization: %s", duplicate_names)
        normalized = normalized.loc[:, ~normalized.columns.duplicated()].copy()
    return normalized


def read_csv_with_encoding(
    path: str | Path,
    nrows: int | None = None,
    *,
    header: int | str | None = "infer",
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Read a CSV using common encodings found in public IDS mirrors."""
    csv_path = Path(path)
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            frame = pd.read_csv(
                csv_path,
                encoding=encoding,
                low_memory=False,
                nrows=nrows,
                header=header,
                names=names,
                on_bad_lines="skip",
            )
            frame = normalize_columns(frame)
            LOGGER.info(
                "Loaded %s with encoding=%s and shape=%s",
                csv_path.name,
                encoding,
                frame.shape,
            )
            return frame
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(
        f"Could not read {csv_path}. Tried encodings: {' | '.join(errors)}"
    )


def read_parquet_file(
    path: str | Path, nrows: int | None = None
) -> pd.DataFrame:
    """Read Parquet and preserve label classes in deterministic capped runs."""
    parquet_path = Path(path)
    try:
        if nrows is None:
            frame = pd.read_parquet(parquet_path)
        else:
            frame = pd.read_parquet(parquet_path)
            if len(frame) > nrows:
                label_column = next(
                    (
                        column
                        for column in frame.columns
                        if str(column).strip().lower()
                        in {"label", "category", "attack"}
                    ),
                    None,
                )
                if (
                    label_column is not None
                    and frame[label_column].nunique(dropna=True) <= nrows
                ):
                    groups = frame.groupby(
                        label_column,
                        sort=True,
                        dropna=True,
                        observed=True,
                    )
                    required = pd.concat(
                        [
                            group.sample(
                                n=min(5, len(group)),
                                random_state=42,
                            )
                            for _, group in groups
                        ]
                    )
                    if len(required) > nrows:
                        required = groups.sample(n=1, random_state=42)
                    remaining = frame.drop(index=required.index)
                    extra_count = nrows - len(required)
                    extra = (
                        remaining.sample(n=extra_count, random_state=42)
                        if extra_count
                        else remaining.head(0)
                    )
                    frame = pd.concat([required, extra]).sort_index()
                else:
                    frame = frame.sample(n=nrows, random_state=42).sort_index()
    except ImportError as exc:
        raise RuntimeError(
            "Install the pinned pyarrow dependency to load Parquet datasets."
        ) from exc
    frame = normalize_columns(frame)
    LOGGER.info("Loaded %s with shape=%s", parquet_path.name, frame.shape)
    return frame


def iter_dataset_files(
    source: str | Path, extensions: set[str]
) -> list[Path]:
    """Return supported dataset files below a file or directory."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in extensions:
            expected = ", ".join(sorted(extensions))
            raise ValueError(
                f"Expected one of {expected}, received: {path}"
            )
        return [path]
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in extensions
    )
    if not files:
        expected = ", ".join(sorted(extensions))
        raise FileNotFoundError(
            f"No supported dataset files ({expected}) found under {path}"
        )
    return files


def iter_csv_files(source: str | Path) -> list[Path]:
    """Return a CSV file or all CSV files recursively below a directory."""
    return iter_dataset_files(source, {".csv"})


def _load_file_collection(
    source: str | Path,
    reader: Callable[[Path, int | None], pd.DataFrame],
    max_rows: int | None,
    dataset_name: str,
    extensions: set[str],
    ignored_filenames: set[str] | None = None,
) -> pd.DataFrame:
    """Load dataset files while distributing an optional row cap."""
    frames: list[pd.DataFrame] = []
    remaining = max_rows
    ignored = {
        value.lower() for value in (ignored_filenames or set())
    }
    files = [
        path
        for path in iter_dataset_files(source, extensions)
        if path.name.lower() not in ignored
    ]

    for index, dataset_file in enumerate(files):
        if remaining is not None and remaining <= 0:
            break
        requested_rows = None
        if remaining is not None:
            files_left = len(files) - index
            requested_rows = max(1, math.ceil(remaining / files_left))
        frame = reader(dataset_file, requested_rows)
        if frame.empty:
            LOGGER.warning("Skipping empty dataset file: %s", dataset_file)
            continue
        if remaining is not None and len(frame) > remaining:
            frame = frame.head(remaining)
        frame["source_file"] = dataset_file.name
        frame["source_dataset"] = dataset_name
        frames.append(frame)
        if remaining is not None:
            remaining -= len(frame)

    if not frames:
        raise ValueError(f"No {dataset_name} rows were loaded from {source}.")
    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    LOGGER.info("Merged %s dataset shape=%s", dataset_name, merged.shape)
    return normalize_columns(merged)


def load_cicids2017(source: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load and merge CICIDS2017 machine-learning CSV or Parquet files."""

    def reader(dataset_file: Path, nrows: int | None) -> pd.DataFrame:
        """Read one CICIDS2017 source file."""
        if dataset_file.suffix.lower() == ".parquet":
            return read_parquet_file(dataset_file, nrows=nrows)
        return read_csv_with_encoding(dataset_file, nrows=nrows)

    frame = _load_file_collection(
        source,
        reader,
        max_rows,
        "cicids2017",
        {".csv", ".parquet"},
    )
    if not any(column.lower() == "label" for column in frame.columns):
        raise KeyError(
            "CICIDS2017 CSV files must contain a 'Label' column. "
            f"Available columns: {list(frame.columns)}"
        )
    return frame


def _looks_like_headerless_bot_iot(frame: pd.DataFrame) -> bool:
    """Detect BoT-IoT exports where the first data row became the header."""
    columns = {str(column).strip().lower() for column in frame.columns}
    has_known_header = bool(columns.intersection({"pkseqid", "category", "subcategory"}))
    return not has_known_header and len(frame.columns) == len(BOT_IOT_COLUMNS)


def _clean_bot_iot_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove wrapping whitespace and quotes from BoT-IoT text columns."""
    cleaned = frame.copy()
    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        cleaned[column] = (
            cleaned[column]
            .astype("string")
            .str.strip()
            .str.strip("\"'")
            .replace({"": pd.NA})
        )
    return cleaned


def load_bot_iot(source: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load and merge official or headerless BoT-IoT CSV exports."""

    def reader(csv_file: Path, nrows: int | None) -> pd.DataFrame:
        """Read one BoT-IoT CSV chunk and repair headerless exports."""
        frame = read_csv_with_encoding(csv_file, nrows=nrows)
        if _looks_like_headerless_bot_iot(frame):
            frame = read_csv_with_encoding(
                csv_file,
                nrows=nrows,
                header=None,
                names=BOT_IOT_COLUMNS,
            )
        unnamed = [
            column for column in frame.columns if str(column).lower().startswith("unnamed:")
        ]
        return _clean_bot_iot_strings(frame.drop(columns=unnamed, errors="ignore"))

    frame = _load_file_collection(
        source,
        reader,
        max_rows,
        "bot-iot",
        {".csv"},
        ignored_filenames={"data_names.csv"},
    )
    columns = {column.lower(): column for column in frame.columns}
    if not {"category", "attack", "subcategory"}.intersection(columns):
        raise KeyError(
            "BoT-IoT CSV files must contain one of 'category', 'subcategory', or "
            f"'attack'. Available columns: {list(frame.columns)}"
        )
    return frame


def merge_source_native_datasets(
    cicids2017: pd.DataFrame,
    bot_iot: pd.DataFrame,
) -> pd.DataFrame:
    """Merge both raw schemas with common labels and derived shared features."""
    cicids = cicids2017.copy()
    bot = bot_iot.copy()
    cicids_label = next(
        (
            column
            for column in cicids.columns
            if str(column).strip().lower() == "label"
        ),
        "Label",
    )
    bot_label = next(
        (
            column
            for column in bot.columns
            if str(column).strip().lower() == "category"
        ),
        "category",
    )
    cicids["common_label"] = cicids[cicids_label].map(common_attack_label)
    bot["common_label"] = bot[bot_label].map(common_attack_label)
    source_native = pd.concat([cicids, bot], ignore_index=True, sort=False)

    shared = harmonize_datasets(cicids2017, bot_iot)
    for feature in COMMON_FEATURES:
        source_native[feature] = shared[feature].to_numpy()
    source_native["source_dataset"] = shared["source_dataset"].to_numpy()
    if "source_file" in shared.columns:
        source_native["source_file"] = shared["source_file"].to_numpy()
    return normalize_columns(source_native)


def dataset_raw_path(config: ProjectConfig, dataset: str) -> Path:
    """Resolve the configured raw directory for a supported dataset."""
    canonical = normalize_dataset_name(dataset)
    if canonical == "combined":
        raise ValueError(
            "The combined dataset uses both configured raw dataset paths."
        )
    key = {
        "cicids2017": "paths.cicids2017_raw_dir",
        "bot-iot": "paths.bot_iot_raw_dir",
    }[canonical]
    configured = config.get(key)
    if configured is not None:
        return config.path(key)
    return config.path("paths.raw_data_dir") / canonical


def load_dataset(
    config: ProjectConfig,
    dataset: str | None = None,
    source: str | Path | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load a supported dataset using configuration or explicit overrides."""
    canonical = normalize_dataset_name(
        dataset or str(config.get("data.dataset", "cicids2017"))
    )
    if max_rows is None:
        max_rows_value = config.get("data.max_rows")
        if isinstance(max_rows_value, dict):
            max_rows_value = max_rows_value.get(canonical)
        max_rows = (
            int(max_rows_value) if max_rows_value is not None else None
        )
    if canonical == "combined":
        if source is not None:
            raise ValueError(
                "A combined run reads both paths from config.yaml; "
                "--input is not supported."
            )
        total_rows = max_rows
        cicids_rows = (
            None if total_rows is None else (total_rows + 1) // 2
        )
        bot_iot_rows = (
            None if total_rows is None else total_rows // 2
        )
        cicids = load_cicids2017(
            dataset_raw_path(config, "cicids2017"),
            max_rows=cicids_rows,
        )
        bot_iot = load_bot_iot(
            dataset_raw_path(config, "bot-iot"),
            max_rows=bot_iot_rows,
        )
        feature_mode = str(
            config.get("data.combined_feature_mode", "union")
        ).strip().lower()
        if feature_mode in {"shared", "harmonized", "common"}:
            combined = harmonize_datasets(cicids, bot_iot)
        elif feature_mode in {"union", "source-native", "source_native", "rich"}:
            combined = merge_source_native_datasets(cicids, bot_iot)
        else:
            raise ValueError(
                "Unsupported data.combined_feature_mode "
                f"'{feature_mode}'. Use 'union' or 'shared'."
            )
        LOGGER.info(
            "Built combined dataset mode=%s shape=%s source_counts=%s",
            feature_mode,
            combined.shape,
            combined["source_dataset"].value_counts().to_dict(),
        )
        return combined
    dataset_source = (
        Path(source)
        if source is not None
        else dataset_raw_path(config, canonical)
    )
    if canonical == "cicids2017":
        return load_cicids2017(dataset_source, max_rows=max_rows)
    return load_bot_iot(dataset_source, max_rows=max_rows)


def main(argv: Iterable[str] | None = None) -> None:
    """Load a configured real IDS dataset."""
    parser = argparse.ArgumentParser(
        description="Load CICIDS2017 CSV/Parquet or BoT-IoT CSV files."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--dataset",
        choices=list(SUPPORTED_DATASETS),
        help="Override data.dataset from config.yaml",
    )
    parser.add_argument(
        "--input", help="Optional dataset file or directory override"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional row cap overriding data.max_rows for this run",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    frame = load_dataset(
        config,
        dataset=args.dataset,
        source=args.input,
        max_rows=args.max_rows,
    )
    LOGGER.info("Loaded dataset with columns=%s", list(frame.columns))


if __name__ == "__main__":
    main()
