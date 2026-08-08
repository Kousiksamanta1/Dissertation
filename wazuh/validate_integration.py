"""Validate Wazuh XML and execute the active-response mock locally."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import xml.etree.ElementTree as element_tree
from pathlib import Path
from typing import Any


def load_active_response(path: Path) -> Any:
    """Import active_response.py without requiring wazuh to be a package."""
    specification = importlib.util.spec_from_file_location(
        "soc_ready_wazuh_active_response", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_xml(wazuh_dir: Path) -> list[str]:
    """Parse all supplied Wazuh XML files."""
    validated: list[str] = []
    for name in ("custom_decoder.xml", "custom_rules.xml", "ossec.conf"):
        element_tree.parse(wazuh_dir / name)
        validated.append(name)
    return validated


def run_mock(
    project_root: Path, dataset: str = "cicids2017"
) -> dict[str, Any]:
    """Process the bundled mock Wazuh alert into a temporary output log."""
    wazuh_dir = project_root / "wazuh"
    module = load_active_response(wazuh_dir / "active_response.py")
    message = json.loads(
        (wazuh_dir / "mock_wazuh_alert.json").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory(prefix="soc-ready-wazuh-") as directory:
        temporary = Path(directory)
        event = module.process_message(
            message,
            project_root=project_root,
            config_path=project_root / "config.yaml",
            db_path=temporary / "alerts.db",
            output_log=temporary / "enriched.json",
            dataset=dataset,
        )
        if event.get("integration") != "soc-ready-ids":
            raise RuntimeError("Mock active response did not return enrichment.")
        if not (temporary / "enriched.json").exists():
            raise RuntimeError("Mock active response did not write JSON output.")
        return event


def main() -> None:
    """Validate all integration files and print the mock enrichment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument(
        "--dataset",
        choices=("cicids2017", "bot-iot", "combined"),
        default="cicids2017",
    )
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    validated = validate_xml(project_root / "wazuh")
    event = run_mock(project_root, args.dataset)
    print(
        json.dumps(
            {"validated_xml": validated, "mock_event": event},
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
