#!/usr/bin/env python3
"""Wazuh stateless active response for SOC-ready IDS enrichment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO


def _extra_arg_value(message: dict[str, Any], name: str) -> str | None:
    """Read a --name=value entry from Wazuh active-response extra arguments."""
    arguments = message.get("parameters", {}).get("extra_args", [])
    for argument in arguments:
        text = str(argument)
        prefix = f"--{name}="
        if text.startswith(prefix):
            return text[len(prefix) :]
    return None


def resolve_project_root(
    message: dict[str, Any] | None = None,
    explicit_root: str | Path | None = None,
) -> Path:
    """Resolve the deployed project root from CLI, environment, or extra args."""
    candidates = [
        explicit_root,
        os.environ.get("SOC_READY_IDS_ROOT"),
        _extra_arg_value(message or {}, "project-root"),
        Path(__file__).resolve().parents[1],
        Path("/opt/soc-ready-ids"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "config.yaml").exists() and (path / "src").exists():
            return path
    raise FileNotFoundError(
        "SOC-ready IDS project root was not found. Set SOC_READY_IDS_ROOT "
        "or pass --project-root through Wazuh extra_args."
    )


def configure_import_path(project_root: Path) -> None:
    """Make the source package importable in a deployed active response."""
    source_path = str(project_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


def read_message(stream: TextIO) -> dict[str, Any]:
    """Read and validate one newline-delimited Wazuh JSON message."""
    line = stream.readline()
    if not line:
        raise ValueError("No active-response JSON was received on STDIN.")
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError("Active-response input is not valid JSON.") from exc
    if not isinstance(message, dict):
        raise ValueError("Active-response input must be a JSON object.")
    return message


def extract_wazuh_alert(message: dict[str, Any]) -> dict[str, Any]:
    """Return the full Wazuh alert nested in an active-response message."""
    alert = message.get("parameters", {}).get("alert", {})
    if not isinstance(alert, dict):
        raise ValueError("The active-response message has no alert object.")
    return alert


def _first(mapping: dict[str, Any], names: list[str]) -> Any:
    """Return the first non-empty alias from a mapping."""
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return None


def _parse_full_log(alert: dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON full_log when the original event used JSON."""
    full_log = alert.get("full_log")
    if not isinstance(full_log, str):
        return {}
    try:
        parsed = json.loads(full_log)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def alert_to_flow(alert: dict[str, Any]) -> dict[str, Any]:
    """Map Wazuh alert fields from CICIDS2017 or BoT-IoT into a raw flow."""
    data = alert.get("data", {})
    data = data if isinstance(data, dict) else {}
    combined = {**_parse_full_log(alert), **data}
    flow: dict[str, Any] = {
        "alert_id": alert.get("id"),
        "timestamp": alert.get("timestamp"),
        "src_ip": _first(combined, ["srcip", "src_ip", "src ip", "saddr"]),
        "dst_ip": _first(combined, ["dstip", "dst_ip", "dst ip", "daddr"]),
        "Destination Port": _first(
            combined,
            ["dstport", "dst_port", "destination port", "dport"],
        ),
        "Flow Packets/s": _first(
            combined,
            ["flow_packets_s", "flow packets/s", "rate", "srate"],
        ),
        "Flow Bytes/s": _first(
            combined, ["flow_bytes_s", "flow bytes/s", "bytes"]
        ),
        "Flow Duration": _first(
            combined, ["flow_duration", "flow duration", "dur"]
        ),
        "SYN Flag Count": _first(
            combined, ["syn_count", "syn flag count"]
        ),
        "Protocol": _first(
            combined, ["protocol", "proto_number", "proto"]
        ),
        "asset_criticality": _first(
            combined, ["asset_criticality", "asset criticality"]
        )
        or 50,
        "ground_truth": _first(
            combined, ["attack_type", "label", "category"]
        ),
        "source_dataset": _first(combined, ["dataset", "source_dataset"]),
    }
    passthrough_aliases = {
        "Total Fwd Packets": ["total fwd packets", "spkts"],
        "Total Backward Packets": ["total backward packets", "dpkts"],
        "Total Length of Fwd Packets": [
            "total length of fwd packets",
            "sbytes",
        ],
        "Total Length of Bwd Packets": [
            "total length of bwd packets",
            "dbytes",
        ],
        "ACK Flag Count": ["ack flag count"],
    }
    for target, aliases in passthrough_aliases.items():
        value = _first(combined, aliases)
        if value is not None:
            flow[target] = value
    for key, value in combined.items():
        if isinstance(value, (str, int, float, bool)) and key not in flow:
            flow[str(key)] = value
    return {key: value for key, value in flow.items() if value is not None}


def enriched_event(
    message: dict[str, Any], triage_result: dict[str, Any]
) -> dict[str, Any]:
    """Build a Wazuh-ingestible JSON event from a triage result."""
    alert = extract_wazuh_alert(message)
    return {
        "integration": "soc-ready-ids",
        "wazuh_alert_id": alert.get("id"),
        "wazuh_rule_id": alert.get("rule", {}).get("id"),
        "agent_id": alert.get("agent", {}).get("id"),
        "agent_name": alert.get("agent", {}).get("name"),
        "timestamp": triage_result.get("timestamp"),
        "alert_id": triage_result.get("alert_id"),
        "src_ip": triage_result.get("src_ip"),
        "dst_ip": triage_result.get("dst_ip"),
        "dst_port": triage_result.get("dst_port"),
        "attack_type": triage_result.get("attack_type"),
        "confidence": triage_result.get("confidence"),
        "risk_score": triage_result.get("risk_score"),
        "risk_tier": triage_result.get("risk_tier"),
        "cluster_id": triage_result.get("cluster_id"),
        "is_suppressed": triage_result.get("is_suppressed"),
        "triage_duration_ms": triage_result.get("triage_duration_ms"),
        "explanation_text": triage_result.get("explanation_text"),
        "recommended_action": triage_result.get("recommended_action"),
        "top_features": triage_result.get("top_features", []),
        "shap_plot_path": triage_result.get("shap_plot_path"),
    }


def append_json_event(event: dict[str, Any], output_log: str | Path) -> Path:
    """Append one enriched event to a JSON-lines log monitored by Wazuh."""
    path = Path(output_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, default=str, separators=(",", ":")) + "\n")
    return path


def process_message(
    message: dict[str, Any],
    *,
    project_root: str | Path | None = None,
    config_path: str | Path | None = None,
    db_path: str | Path | None = None,
    output_log: str | Path | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Run one stateless Wazuh add command through the triage pipeline."""
    command = str(message.get("command", "")).lower()
    if command != "add":
        return {
            "integration": "soc-ready-ids",
            "status": "ignored",
            "reason": f"Unsupported stateless command: {command or 'missing'}",
        }

    root = resolve_project_root(message, project_root)
    configure_import_path(root)
    from soc_ready_ids.config import load_config
    from soc_ready_ids.triage.triage_pipeline import TriagePipeline

    configured_path = Path(
        config_path
        or os.environ.get("SOC_READY_IDS_CONFIG")
        or _extra_arg_value(message, "config")
        or root / "config.yaml"
    )
    enriched_log = Path(
        output_log
        or os.environ.get("SOC_READY_IDS_ENRICHED_LOG")
        or _extra_arg_value(message, "output-log")
        or "/var/ossec/logs/soc-ready-ids-enriched.json"
    )
    config = load_config(configured_path)
    selected_dataset = (
        dataset
        or
        os.environ.get("SOC_READY_IDS_DATASET")
        or _extra_arg_value(message, "dataset")
        or str(config.get("data.dataset", "cicids2017"))
    )
    config = config.for_dataset(selected_dataset)
    database_path = Path(
        db_path
        or os.environ.get("SOC_READY_IDS_DB")
        or _extra_arg_value(message, "db")
        or config.path("paths.sqlite_db")
    )
    pipeline = TriagePipeline(config=config, db_path=database_path)
    result = pipeline.process_flow(alert_to_flow(extract_wazuh_alert(message)))
    event = enriched_event(message, result)
    append_json_event(event, enriched_log)
    return event


def main(argv: list[str] | None = None) -> int:
    """Read one Wazuh message, enrich it, and optionally print mock output."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root")
    parser.add_argument("--config")
    parser.add_argument("--db")
    parser.add_argument("--output-log")
    parser.add_argument("--dataset")
    parser.add_argument("--mock", action="store_true")
    args, _ = parser.parse_known_args(argv)
    try:
        message = read_message(sys.stdin)
        event = process_message(
            message,
            project_root=args.project_root,
            config_path=args.config,
            db_path=args.db,
            output_log=args.output_log,
            dataset=args.dataset,
        )
        if args.mock or os.environ.get("SOC_READY_IDS_MOCK") == "1":
            print(json.dumps(event, indent=2, default=str))
        return 0
    except Exception as exc:
        error = {
            "integration": "soc-ready-ids",
            "status": "error",
            "error": str(exc),
        }
        print(json.dumps(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
