"""Tests for Phase 1 data loading and preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from soc_ready_ids.config import ProjectConfig
from soc_ready_ids.data.loader import (
    BOT_IOT_COLUMNS,
    load_dataset,
    load_bot_iot,
    load_cicids2017,
)
from soc_ready_ids.data.harmonizer import (
    COMMON_FEATURES,
    common_attack_label,
    harmonize_datasets,
)
from soc_ready_ids.data.preprocessor import (
    build_feature_matrix,
    clean_label,
    preprocess_dataframe,
    save_preprocessed,
)


def test_clean_label_normalizes_benign() -> None:
    """Benign and BoT-IoT Normal variants should normalize to BENIGN."""
    assert clean_label("BENIGN") == "BENIGN"
    assert clean_label(" normal ") == "BENIGN"
    assert clean_label("DDoS") == "DDoS"


def test_common_attack_taxonomy() -> None:
    """Detailed labels should map into a shared cross-dataset taxonomy."""
    assert common_attack_label("BENIGN") == "BENIGN"
    assert common_attack_label("DoS Hulk") == "DoS"
    assert common_attack_label("DDoS") == "DDoS"
    assert common_attack_label("PortScan") == "Reconnaissance"
    assert common_attack_label("Web Attack - XSS") == "Other Attack"


def test_harmonize_datasets_uses_shared_features_and_units() -> None:
    """Both source schemas should produce one unit-normalized feature frame."""
    cicids = pd.DataFrame(
        {
            "Protocol": [6],
            "Flow Duration": [2_000_000],
            "Total Fwd Packets": [4],
            "Total Backward Packets": [2],
            "Fwd Packets Length Total": [400],
            "Bwd Packets Length Total": [100],
            "Flow Packets/s": [3],
            "Fwd Packets/s": [2],
            "Bwd Packets/s": [1],
            "Flow Bytes/s": [250],
            "Label": ["PortScan"],
            "source_file": ["cic.parquet"],
        }
    )
    bot_iot = pd.DataFrame(
        {
            "proto": ["udp"],
            "dur": [2.0],
            "pkts": [6],
            "spkts": [4],
            "dpkts": [2],
            "bytes": [500],
            "sbytes": [400],
            "dbytes": [100],
            "rate": [3],
            "srate": [2],
            "drate": [1],
            "category": ["DDoS"],
            "source_file": ["bot.csv"],
        }
    )

    combined = harmonize_datasets(cicids, bot_iot)

    assert list(combined.columns[: len(COMMON_FEATURES)]) == list(
        COMMON_FEATURES
    )
    assert combined["duration_seconds"].tolist() == [2.0, 2.0]
    assert combined["protocol_number"].tolist() == [6.0, 17.0]
    assert combined["common_label"].tolist() == [
        "Reconnaissance",
        "DDoS",
    ]
    assert set(combined["source_dataset"]) == {
        "cicids2017",
        "bot-iot",
    }


def test_load_bot_iot_handles_headerless_csv(tmp_path: Path) -> None:
    """BoT-IoT mirrors without headers should be loaded with official names."""
    rows = []
    for index, category in enumerate(["Normal", "DDoS", "DoS", "Reconnaissance"]):
        row = [0] * len(BOT_IOT_COLUMNS)
        row[BOT_IOT_COLUMNS.index("pkSeqID")] = index + 1
        row[BOT_IOT_COLUMNS.index("stime")] = 1.0 + index
        row[BOT_IOT_COLUMNS.index("proto")] = "tcp"
        row[BOT_IOT_COLUMNS.index("saddr")] = f"10.0.0.{index + 1}"
        row[BOT_IOT_COLUMNS.index("daddr")] = "192.168.1.10"
        row[BOT_IOT_COLUMNS.index("pkts")] = 10 + index
        row[BOT_IOT_COLUMNS.index("bytes")] = 100 + index
        row[BOT_IOT_COLUMNS.index("attack")] = 0 if category == "Normal" else 1
        row[BOT_IOT_COLUMNS.index("category")] = category
        row[BOT_IOT_COLUMNS.index("subcategory")] = "normal" if category == "Normal" else "tcp"
        rows.append(row)
    path = tmp_path / "bot_iot_headerless.csv"
    pd.DataFrame(rows).to_csv(path, index=False, header=False)

    loaded = load_bot_iot(path)

    assert loaded.shape[0] == 4
    assert "category" in loaded.columns
    assert "source_file" in loaded.columns
    assert set(loaded["category"]) == {"Normal", "DDoS", "DoS", "Reconnaissance"}


def test_load_cicids_parquet_distributes_row_cap(tmp_path: Path) -> None:
    """A capped Parquet load should sample across all supplied files."""
    benign = pd.DataFrame(
        {"Flow Duration": range(10), "Label": ["BENIGN"] * 10}
    )
    attacks = pd.DataFrame(
        {"Flow Duration": range(10, 20), "Label": ["DDoS"] * 10}
    )
    benign.to_parquet(tmp_path / "benign.parquet", index=False)
    attacks.to_parquet(tmp_path / "ddos.parquet", index=False)

    loaded = load_cicids2017(tmp_path, max_rows=6)

    assert len(loaded) == 6
    assert set(loaded["source_file"]) == {
        "benign.parquet",
        "ddos.parquet",
    }
    assert set(loaded["Label"]) == {"BENIGN", "DDoS"}


def test_load_cicids_parquet_cap_preserves_rare_class(
    tmp_path: Path,
) -> None:
    """A capped file should retain a rare attack label beyond its first rows."""
    frame = pd.DataFrame(
        {
            "Flow Duration": range(20),
            "Label": ["BENIGN"] * 18 + ["DDoS"] * 2,
        }
    )
    path = tmp_path / "mixed.parquet"
    frame.to_parquet(path, index=False)

    loaded = load_cicids2017(path, max_rows=7)

    assert len(loaded) == 7
    assert set(loaded["Label"]) == {"BENIGN", "DDoS"}
    assert (loaded["Label"] == "DDoS").sum() == 2


def test_load_bot_iot_skips_header_only_metadata_file(
    tmp_path: Path,
) -> None:
    """The downloaded data_names.csv file should not become a data row."""
    columns = ["pkSeqID", "pkts", "attack", "category"]
    pd.DataFrame(columns=columns).to_csv(
        tmp_path / "data_names.csv", index=False
    )
    pd.DataFrame(
        [
            [1, 10, 0, "Normal"],
            [2, 20, 1, "DDoS"],
        ],
        columns=columns,
    ).to_csv(tmp_path / "data_1.csv", index=False)

    loaded = load_bot_iot(tmp_path)

    assert len(loaded) == 2
    assert set(loaded["category"]) == {"Normal", "DDoS"}


def test_load_dataset_uses_dataset_specific_row_cap(
    tmp_path: Path,
) -> None:
    """Dataset-specific configured caps should be applied automatically."""
    source = tmp_path / "cicids"
    source.mkdir()
    pd.DataFrame(
        {
            "Flow Duration": range(20),
            "Label": ["BENIGN"] * 10 + ["DDoS"] * 10,
        }
    ).to_parquet(source / "flows.parquet", index=False)
    config = ProjectConfig(
        values={
            "paths": {"cicids2017_raw_dir": str(source)},
            "data": {
                "dataset": "cicids2017",
                "max_rows": {"cicids2017": 6, "bot-iot": 4},
            },
        },
        root=tmp_path,
    )

    loaded = load_dataset(config, dataset="cicids2017")

    assert len(loaded) == 6
    assert set(loaded["Label"]) == {"BENIGN", "DDoS"}


def test_load_combined_allocates_rows_equally(
    tmp_path: Path,
) -> None:
    """A combined cap should be split evenly across both real sources."""
    cicids_source = tmp_path / "cicids"
    bot_source = tmp_path / "bot"
    cicids_source.mkdir()
    bot_source.mkdir()
    pd.DataFrame(
        {
            "Protocol": [6] * 10,
            "Flow Duration": range(1_000_000, 11_000_000, 1_000_000),
            "Total Fwd Packets": range(1, 11),
            "Total Backward Packets": range(1, 11),
            "Fwd Packets Length Total": range(100, 1100, 100),
            "Bwd Packets Length Total": range(100, 1100, 100),
            "Label": ["BENIGN"] * 5 + ["DDoS"] * 5,
        }
    ).to_parquet(cicids_source / "flows.parquet", index=False)
    pd.DataFrame(
        {
            "proto": ["tcp"] * 10,
            "dur": range(1, 11),
            "pkts": range(2, 22, 2),
            "spkts": range(1, 11),
            "dpkts": range(1, 11),
            "bytes": range(200, 2200, 200),
            "sbytes": range(100, 1100, 100),
            "dbytes": range(100, 1100, 100),
            "category": ["Normal"] * 5 + ["DoS"] * 5,
            "attack": [0] * 5 + [1] * 5,
        }
    ).to_csv(bot_source / "flows.csv", index=False)
    config = ProjectConfig(
        values={
            "paths": {
                "cicids2017_raw_dir": str(cicids_source),
                "bot_iot_raw_dir": str(bot_source),
            },
            "data": {
                "dataset": "combined",
                "max_rows": {"combined": 8},
            },
        },
        root=tmp_path,
    )

    loaded = load_dataset(config, dataset="combined")

    assert len(loaded) == 8
    assert loaded["source_dataset"].value_counts().to_dict() == {
        "cicids2017": 4,
        "bot-iot": 4,
    }
    assert set(COMMON_FEATURES).issubset(loaded.columns)


def test_build_feature_matrix_drops_invalid_rows() -> None:
    """Rows with NaN or Inf feature values should be removed."""
    frame = pd.DataFrame(
        {
            "Flow ID": ["a", "b", "c", "d"],
            "Flow Duration": [1.0, np.inf, 3.0, 4.0],
            "Flow Packets/s": [10.0, 20.0, 30.0, 40.0],
            "protocol_type": ["tcp", "udp", "tcp", "icmp"],
            "Label": ["BENIGN", "DDoS", "BENIGN", "PortScan"],
        }
    )
    X, y, metadata = build_feature_matrix(frame, "Label", ["Flow ID"])
    assert X.shape[0] == 3
    assert "Flow ID" not in X.columns
    assert not X.isna().any().any()
    assert set(y) == {"BENIGN", "PortScan"}
    assert list(metadata.columns) == ["Flow ID"]


def test_preprocess_dataframe_returns_scaled_splits() -> None:
    """Preprocessor should return non-empty scaled train/test splits."""
    frame = pd.DataFrame(
        {
            "Flow Duration": list(range(20)),
            "Flow Packets/s": [float(i * 2) for i in range(20)],
            "Label": ["BENIGN"] * 10 + ["DDoS"] * 10,
        }
    )
    config = ProjectConfig(
        values={
            "project": {"random_state": 42},
            "data": {
                "dataset": "cicids2017",
                "label_column": "Label",
                "label_columns": {"cicids2017": "Label"},
                "test_size": 0.2,
                "metadata_columns": [],
                "drop_columns": {"cicids2017": []},
                "drop_invalid_rows": True,
            },
        },
        root=Path.cwd(),
    )
    artifacts = preprocess_dataframe(frame, config)
    assert artifacts.X_train.shape[0] == 16
    assert artifacts.X_test.shape[0] == 4
    assert len(artifacts.label_encoder.classes_) == 2
    assert np.isclose(artifacts.X_train.mean().abs().max(), 0.0)


def test_bot_iot_preprocessing_removes_label_leakage(tmp_path: Path) -> None:
    """BoT-IoT target-derived columns should not enter the model matrix."""
    frame = pd.DataFrame(
        {
            "pkSeqID": range(20),
            "saddr": [f"10.0.0.{i}" for i in range(20)],
            "daddr": ["192.168.1.5"] * 20,
            "stime": [1000.0 + i for i in range(20)],
            "pkts": [10 + i for i in range(20)],
            "bytes": [200 + i * 3 for i in range(20)],
            "dur": [0.5 + i / 100 for i in range(20)],
            "proto": ["tcp", "udp"] * 10,
            "attack": [0] * 10 + [1] * 10,
            "category": ["Normal"] * 10 + ["DDoS"] * 10,
            "subcategory": ["normal"] * 10 + ["tcp"] * 10,
        }
    )
    config = ProjectConfig(
        values={
            "project": {"random_state": 42},
            "data": {
                "dataset": "bot-iot",
                "label_column": "Label",
                "label_columns": {"bot-iot": "category"},
                "test_size": 0.2,
                "metadata_columns": ["pkSeqID", "saddr", "daddr", "stime"],
                "drop_columns": {"bot-iot": ["attack", "category", "subcategory"]},
                "drop_invalid_rows": True,
            },
        },
        root=Path.cwd(),
    )

    artifacts = preprocess_dataframe(frame, config, dataset_name="bot-iot")
    paths = save_preprocessed(artifacts, tmp_path)

    assert not {"attack", "category", "subcategory", "saddr", "daddr"}.intersection(
        artifacts.feature_columns
    )
    assert "proto=tcp" in artifacts.feature_columns
    assert artifacts.metadata_train.shape[1] == 4
    assert paths["manifest"].exists()


def test_bot_iot_expected_missing_ports_are_imputed() -> None:
    """Normal ARP rows should survive dataset-specific missing-value handling."""
    frame = pd.DataFrame(
        {
            "pkSeqID": range(20),
            "sport": [np.nan] * 10 + list(range(10)),
            "dport": [np.nan] * 10 + [80] * 10,
            "pkts": list(range(1, 21)),
            "proto": ["arp"] * 10 + ["tcp"] * 10,
            "attack": [0] * 10 + [1] * 10,
            "category": ["Normal"] * 10 + ["DDoS"] * 10,
            "subcategory": ["Normal"] * 10 + ["TCP"] * 10,
        }
    )
    config = ProjectConfig(
        values={
            "project": {"random_state": 42},
            "data": {
                "dataset": "bot-iot",
                "label_columns": {"bot-iot": "category"},
                "test_size": 0.2,
                "metadata_columns": ["pkSeqID"],
                "drop_columns": {
                    "bot-iot": ["attack", "category", "subcategory"]
                },
                "drop_invalid_rows": {"bot-iot": False},
            },
        },
        root=Path.cwd(),
    )

    artifacts = preprocess_dataframe(
        frame, config, dataset_name="bot-iot"
    )

    assert artifacts.retained_rows == 20
    assert not artifacts.X_train.isna().any().any()
    assert set(artifacts.label_encoder.classes_) == {"BENIGN", "DDoS"}
