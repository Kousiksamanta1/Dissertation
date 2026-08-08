"""Tests for the Wazuh active-response integration."""

from __future__ import annotations

import importlib.util
import io
import json
import xml.etree.ElementTree as element_tree
from pathlib import Path

import yaml


def _module():
    """Import the standalone active-response script."""
    path = Path("wazuh/active_response.py").resolve()
    spec = importlib.util.spec_from_file_location("test_active_response", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wazuh_xml_files_are_well_formed() -> None:
    """Decoder, rules, and ossec configuration should parse as XML."""
    for path in (
        "wazuh/custom_decoder.xml",
        "wazuh/custom_rules.xml",
        "wazuh/ossec.conf",
    ):
        assert element_tree.parse(path).getroot() is not None


def test_wazuh_message_mapping_and_active_response(tmp_path: Path) -> None:
    """A mock Wazuh alert should create SQLite and enriched JSON output."""
    module = _module()
    message = json.loads(
        Path("wazuh/mock_wazuh_alert.json").read_text(encoding="utf-8")
    )
    parsed = module.read_message(io.StringIO(json.dumps(message) + "\n"))
    flow = module.alert_to_flow(module.extract_wazuh_alert(parsed))

    project = Path.cwd()
    values = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    values["paths"]["model_dir"] = str(tmp_path / "models")
    values["paths"]["processed_data_dir"] = str(tmp_path / "processed")
    values["paths"]["shap_dir"] = str(tmp_path / "shap")
    values["paths"]["explanation_dir"] = str(tmp_path / "explanations")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(values), encoding="utf-8")

    event = module.process_message(
        parsed,
        project_root=project,
        config_path=config_path,
        db_path=tmp_path / "alerts.db",
        output_log=tmp_path / "enriched.json",
    )
    ignored = module.process_message(
        {"command": "delete"},
        project_root=project,
    )

    assert flow["src_ip"] == "10.0.2.208"
    assert event["integration"] == "soc-ready-ids"
    assert event["attack_type"] == "DDoS"
    assert (tmp_path / "alerts.db").exists()
    assert json.loads(
        (tmp_path / "enriched.json").read_text(encoding="utf-8")
    )["alert_id"] == "mock-wazuh-alert-001"
    assert ignored["status"] == "ignored"
