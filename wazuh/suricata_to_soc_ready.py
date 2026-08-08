#!/usr/bin/env python3
"""Bridge Suricata EVE alerts into the SOC-ready IDS Wazuh input log."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _clean(value: Any, default: str = "unknown") -> str:
    """Return a pipe-safe scalar value for the SOC_READY_IDS log format."""
    text = str(value if value not in (None, "") else default)
    return re.sub(r"[\r\n|]+", " ", text).strip() or default


def _float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_time(value: Any) -> datetime | None:
    """Parse a Suricata ISO timestamp."""
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _attack_type(event: dict[str, Any]) -> str:
    """Map Suricata alert text into the project attack taxonomy."""
    alert = event.get("alert", {})
    alert = alert if isinstance(alert, dict) else {}
    text = " ".join(
        str(alert.get(name, ""))
        for name in ("category", "signature", "metadata")
    ).lower()
    if "ddos" in text:
        return "DDoS"
    if "dos" in text or "denial" in text:
        return "DoS"
    if "scan" in text or "recon" in text:
        return "Reconnaissance"
    if "brute" in text or "login" in text or "credential" in text:
        return "Brute Force"
    if "bot" in text or "trojan" in text or "malware" in text or "c2" in text:
        return "Botnet"
    if "web" in text or "sql" in text or "xss" in text:
        return "Web Attack"
    return "Other Attack"


def _confidence(event: dict[str, Any]) -> float:
    """Convert Suricata priority/severity into model-input confidence."""
    alert = event.get("alert", {})
    alert = alert if isinstance(alert, dict) else {}
    severity = int(_float(alert.get("severity"), 3.0))
    return {1: 0.95, 2: 0.85, 3: 0.75, 4: 0.65}.get(severity, 0.75)


def _packet_rate(event: dict[str, Any]) -> float:
    """Estimate packets per second from Suricata flow metadata."""
    flow = event.get("flow", {})
    flow = flow if isinstance(flow, dict) else {}
    packet_count = _float(flow.get("pkts_toserver")) + _float(
        flow.get("pkts_toclient")
    )
    start = _parse_time(flow.get("start"))
    end = _parse_time(event.get("timestamp"))
    duration = 1.0
    if start and end:
        duration = max((end - start).total_seconds(), 1.0)
    return round(max(packet_count / duration, 1.0), 4)


def _syn_count(event: dict[str, Any], attack_type: str) -> int:
    """Estimate whether the alert had SYN-like TCP evidence."""
    tcp = event.get("tcp", {})
    tcp = tcp if isinstance(tcp, dict) else {}
    flag_text = " ".join(str(value) for value in tcp.values()).lower()
    if "syn" in flag_text:
        return 1
    for value in tcp.values():
        text = str(value).lower().removeprefix("0x")
        try:
            if int(text, 16) & 0x02:
                return 1
        except ValueError:
            continue
    if attack_type in {"DDoS", "DoS", "Reconnaissance"}:
        return 5
    return 0


def suricata_event_to_soc_ready_line(
    event: dict[str, Any],
    *,
    asset_criticality: float = 70.0,
    dataset: str = "Suricata",
) -> str | None:
    """Convert one Suricata EVE alert event into one SOC_READY_IDS line."""
    if event.get("event_type") != "alert":
        return None
    attack_type = _attack_type(event)
    src_ip = event.get("src_ip") or event.get("srcip")
    dst_ip = event.get("dest_ip") or event.get("dst_ip") or event.get("dstip")
    dst_port = event.get("dest_port") or event.get("dst_port") or 0
    values = {
        "dataset": dataset,
        "attack_type": attack_type,
        "srcip": src_ip,
        "dstip": dst_ip,
        "dstport": dst_port,
        "confidence": f"{_confidence(event):.2f}",
        "asset_criticality": f"{asset_criticality:.0f}",
        "flow_packets_s": f"{_packet_rate(event):.4f}",
        "syn_count": str(_syn_count(event, attack_type)),
    }
    return (
        "SOC_READY_IDS|"
        f"dataset={_clean(values['dataset'])}|"
        f"attack_type={_clean(values['attack_type'])}|"
        f"srcip={_clean(values['srcip'])}|"
        f"dstip={_clean(values['dstip'])}|"
        f"dstport={_clean(values['dstport'], '0')}|"
        f"confidence={_clean(values['confidence'], '0.75')}|"
        f"asset_criticality={_clean(values['asset_criticality'], '70')}|"
        f"flow_packets_s={_clean(values['flow_packets_s'], '1')}|"
        f"syn_count={_clean(values['syn_count'], '0')}"
    )


def read_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    """Read current JSON lines from a file once."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def follow_json_lines(path: Path, *, from_start: bool = False) -> Iterable[dict[str, Any]]:
    """Follow a growing EVE JSON file and yield parsed objects."""
    position = 0
    while True:
        if not path.exists():
            time.sleep(2.0)
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            if not from_start:
                handle.seek(0, 2)
            elif position:
                handle.seek(position)
            while True:
                line = handle.readline()
                if not line:
                    position = handle.tell()
                    time.sleep(1.0)
                    if path.exists() and path.stat().st_size < position:
                        position = 0
                        break
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event


def append_soc_ready_lines(
    events: Iterable[dict[str, Any]],
    output_log: Path,
    *,
    asset_criticality: float,
) -> int:
    """Append converted Suricata alerts to the SOC-ready input log."""
    output_log.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_log.open("a", encoding="utf-8") as handle:
        for event in events:
            line = suricata_event_to_soc_ready_line(
                event,
                asset_criticality=asset_criticality,
            )
            if line is None:
                continue
            handle.write(line + "\n")
            handle.flush()
            count += 1
    return count


def main() -> int:
    """Run the Suricata-to-SOC-ready bridge."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eve-log", default="/var/log/suricata/eve.json")
    parser.add_argument(
        "--output-log", default="/var/ossec/logs/soc-ready-ids-input.log"
    )
    parser.add_argument("--asset-criticality", type=float, default=70.0)
    parser.add_argument("--from-start", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    eve_log = Path(args.eve_log)
    output_log = Path(args.output_log)
    events = (
        read_json_lines(eve_log)
        if args.once
        else follow_json_lines(eve_log, from_start=args.from_start)
    )
    count = append_soc_ready_lines(
        events,
        output_log,
        asset_criticality=args.asset_criticality,
    )
    if args.once:
        print(f"Converted {count} Suricata alert events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
