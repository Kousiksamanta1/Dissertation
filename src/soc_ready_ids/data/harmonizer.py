"""Shared feature and label harmonization for cross-dataset training."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

COMMON_FEATURES: tuple[str, ...] = (
    "duration_seconds",
    "protocol_number",
    "total_packets",
    "source_packets",
    "destination_packets",
    "total_bytes",
    "source_bytes",
    "destination_bytes",
    "packet_rate",
    "source_packet_rate",
    "destination_packet_rate",
    "byte_rate",
    "average_packet_bytes",
    "average_source_packet_bytes",
    "average_destination_packet_bytes",
)

PROTOCOL_NUMBERS: dict[str, int] = {
    "icmp": 1,
    "igmp": 2,
    "tcp": 6,
    "udp": 17,
    "ipv6": 41,
    "ipv6-icmp": 58,
    "icmpv6": 58,
    "arp": 0,
}


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    """Return case-insensitive column aliases."""
    return {str(column).strip().lower(): str(column) for column in frame.columns}


def _numeric(
    frame: pd.DataFrame,
    aliases: list[str],
    default: float = 0.0,
) -> pd.Series:
    """Return the first matching column as a numeric series."""
    lookup = _column_lookup(frame)
    for alias in aliases:
        column = lookup.get(alias.lower())
        if column is not None:
            return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _protocol_number(frame: pd.DataFrame) -> pd.Series:
    """Convert numeric or named IP protocols into IANA protocol numbers."""
    lookup = _column_lookup(frame)
    for alias in ("protocol", "proto_number"):
        column = lookup.get(alias)
        if column is not None:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.notna().any():
                return numeric
    protocol_column = lookup.get("proto")
    if protocol_column is None:
        return pd.Series(-1.0, index=frame.index)
    names = (
        frame[protocol_column]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    numeric = pd.to_numeric(names, errors="coerce")
    return numeric.fillna(names.map(PROTOCOL_NUMBERS)).fillna(-1.0)


def common_attack_label(value: object) -> str:
    """Map detailed IDS labels into a shared five-class taxonomy."""
    text = " ".join(
        str(value)
        .strip()
        .strip("\"'")
        .replace("_", " ")
        .replace("\ufffd", " ")
        .split()
    )
    normalized = text.upper()
    if normalized in {"BENIGN", "NORMAL", "0", "FALSE"}:
        return "BENIGN"
    if "DDOS" in normalized:
        return "DDoS"
    if normalized == "DOS" or normalized.startswith("DOS "):
        return "DoS"
    if "RECON" in normalized or "PORTSCAN" in normalized:
        return "Reconnaissance"
    return "Other Attack"


def _safe_rate(
    numerator: pd.Series,
    duration_seconds: pd.Series,
) -> pd.Series:
    """Compute a per-second rate without producing infinite values."""
    duration = duration_seconds.where(duration_seconds > 0)
    return numerator.divide(duration).replace([np.inf, -np.inf], np.nan)


def _safe_average(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Compute an average without producing infinite values."""
    divisor = denominator.where(denominator > 0)
    return numerator.divide(divisor).replace([np.inf, -np.inf], np.nan)


def _base_harmonized_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    duration_seconds: pd.Series,
    protocol_number: pd.Series,
    source_packets: pd.Series,
    destination_packets: pd.Series,
    source_bytes: pd.Series,
    destination_bytes: pd.Series,
    packet_rate: pd.Series,
    source_packet_rate: pd.Series,
    destination_packet_rate: pd.Series,
    byte_rate: pd.Series,
    labels: pd.Series,
) -> pd.DataFrame:
    """Build the canonical shared feature frame."""
    total_packets = source_packets.fillna(0.0) + destination_packets.fillna(0.0)
    total_bytes = source_bytes.fillna(0.0) + destination_bytes.fillna(0.0)
    harmonized = pd.DataFrame(
        {
            "duration_seconds": duration_seconds,
            "protocol_number": protocol_number,
            "total_packets": total_packets,
            "source_packets": source_packets,
            "destination_packets": destination_packets,
            "total_bytes": total_bytes,
            "source_bytes": source_bytes,
            "destination_bytes": destination_bytes,
            "packet_rate": packet_rate,
            "source_packet_rate": source_packet_rate,
            "destination_packet_rate": destination_packet_rate,
            "byte_rate": byte_rate,
            "average_packet_bytes": _safe_average(
                total_bytes, total_packets
            ),
            "average_source_packet_bytes": _safe_average(
                source_bytes, source_packets
            ),
            "average_destination_packet_bytes": _safe_average(
                destination_bytes, destination_packets
            ),
            "common_label": labels.map(common_attack_label),
            "source_dataset": dataset,
        },
        index=frame.index,
    )
    if "source_file" in frame.columns:
        harmonized["source_file"] = frame["source_file"].astype(str)
    return harmonized.replace([np.inf, -np.inf], np.nan)


def harmonize_cicids2017(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert CICIDS2017 flow fields into the shared schema."""
    duration_seconds = _numeric(
        frame, ["Flow Duration"]
    ).divide(1_000_000.0)
    source_packets = _numeric(
        frame, ["Total Fwd Packets", "Subflow Fwd Packets"]
    )
    destination_packets = _numeric(
        frame, ["Total Backward Packets", "Subflow Bwd Packets"]
    )
    source_bytes = _numeric(
        frame,
        [
            "Fwd Packets Length Total",
            "Total Length of Fwd Packets",
            "Subflow Fwd Bytes",
        ],
    )
    destination_bytes = _numeric(
        frame,
        [
            "Bwd Packets Length Total",
            "Total Length of Bwd Packets",
            "Subflow Bwd Bytes",
        ],
    )
    total_packets = source_packets.fillna(0.0) + destination_packets.fillna(0.0)
    total_bytes = source_bytes.fillna(0.0) + destination_bytes.fillna(0.0)
    labels = frame[
        _column_lookup(frame).get("label", "Label")
    ]
    return _base_harmonized_frame(
        frame,
        dataset="cicids2017",
        duration_seconds=duration_seconds,
        protocol_number=_protocol_number(frame),
        source_packets=source_packets,
        destination_packets=destination_packets,
        source_bytes=source_bytes,
        destination_bytes=destination_bytes,
        packet_rate=_numeric(
            frame, ["Flow Packets/s"]
        ).fillna(_safe_rate(total_packets, duration_seconds)),
        source_packet_rate=_numeric(
            frame, ["Fwd Packets/s"]
        ).fillna(_safe_rate(source_packets, duration_seconds)),
        destination_packet_rate=_numeric(
            frame, ["Bwd Packets/s"]
        ).fillna(_safe_rate(destination_packets, duration_seconds)),
        byte_rate=_numeric(
            frame, ["Flow Bytes/s"]
        ).fillna(_safe_rate(total_bytes, duration_seconds)),
        labels=labels,
    )


def harmonize_bot_iot(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert BoT-IoT flow fields into the shared schema."""
    duration_seconds = _numeric(frame, ["dur"])
    source_packets = _numeric(frame, ["spkts"])
    destination_packets = _numeric(frame, ["dpkts"])
    source_bytes = _numeric(frame, ["sbytes"])
    destination_bytes = _numeric(frame, ["dbytes"])
    total_packets = _numeric(frame, ["pkts"]).fillna(
        source_packets.fillna(0.0) + destination_packets.fillna(0.0)
    )
    total_bytes = _numeric(frame, ["bytes"]).fillna(
        source_bytes.fillna(0.0) + destination_bytes.fillna(0.0)
    )
    label_column = _column_lookup(frame).get("category", "category")
    return _base_harmonized_frame(
        frame,
        dataset="bot-iot",
        duration_seconds=duration_seconds,
        protocol_number=_protocol_number(frame),
        source_packets=source_packets,
        destination_packets=destination_packets,
        source_bytes=source_bytes,
        destination_bytes=destination_bytes,
        packet_rate=_numeric(frame, ["rate"]).fillna(
            _safe_rate(total_packets, duration_seconds)
        ),
        source_packet_rate=_numeric(frame, ["srate"]).fillna(
            _safe_rate(source_packets, duration_seconds)
        ),
        destination_packet_rate=_numeric(frame, ["drate"]).fillna(
            _safe_rate(destination_packets, duration_seconds)
        ),
        byte_rate=_safe_rate(total_bytes, duration_seconds),
        labels=frame[label_column],
    )


def harmonize_datasets(
    cicids2017: pd.DataFrame,
    bot_iot: pd.DataFrame,
) -> pd.DataFrame:
    """Merge both datasets after applying the shared schema."""
    combined = pd.concat(
        [
            harmonize_cicids2017(cicids2017),
            harmonize_bot_iot(bot_iot),
        ],
        ignore_index=True,
        sort=False,
    )
    feature_columns = list(COMMON_FEATURES)
    combined[feature_columns] = (
        combined[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )
    return combined


def harmonize_flow(
    raw_flow: Mapping[str, Any],
    source_dataset: str | None = None,
) -> dict[str, float]:
    """Convert one API/Wazuh flow into the shared feature schema."""
    frame = pd.DataFrame([dict(raw_flow)])
    source = (
        source_dataset
        or str(raw_flow.get("source_dataset", "")).strip().lower()
    )
    if source in {"bot-iot", "bot_iot", "botiot"}:
        harmonized = harmonize_bot_iot(
            frame.assign(category="Normal")
        )
    else:
        harmonized = harmonize_cicids2017(
            frame.assign(Label="BENIGN")
        )
    return {
        feature: float(harmonized.iloc[0][feature])
        for feature in COMMON_FEATURES
    }
