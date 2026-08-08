"""Tests for the Suricata live-source bridge."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path("wazuh/suricata_to_soc_ready.py").resolve()
    spec = importlib.util.spec_from_file_location("suricata_bridge", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suricata_alert_converts_to_soc_ready_line() -> None:
    module = _module()
    event = {
        "timestamp": "2026-08-08T02:00:00.000000+0000",
        "event_type": "alert",
        "src_ip": "203.0.113.10",
        "src_port": 80,
        "dest_ip": "192.168.64.20",
        "dest_port": 49152,
        "proto": "TCP",
        "flow": {
            "pkts_toserver": 5,
            "pkts_toclient": 7,
            "start": "2026-08-08T01:59:58.000000+0000",
        },
        "alert": {
            "signature": "GPL ATTACK_RESPONSE id check returned root",
            "category": "Potentially Bad Traffic",
            "severity": 2,
        },
    }

    line = module.suricata_event_to_soc_ready_line(event)

    assert line is not None
    assert line.startswith("SOC_READY_IDS|")
    assert "dataset=Suricata" in line
    assert "attack_type=Other Attack" in line
    assert "srcip=203.0.113.10" in line
    assert "dstip=192.168.64.20" in line
    assert "dstport=49152" in line
    assert "confidence=0.85" in line


def test_suricata_non_alert_is_ignored() -> None:
    module = _module()
    assert module.suricata_event_to_soc_ready_line({"event_type": "flow"}) is None
