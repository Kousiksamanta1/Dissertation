#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/soc-ready-ids}"
DATASET="${SOC_READY_IDS_DATASET:-combined}"
WAZUH_HOME="${WAZUH_HOME:-/var/ossec}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSSEC_CONF="${WAZUH_HOME}/etc/ossec.conf"
ACTIVE_RESPONSE_BIN="${WAZUH_HOME}/active-response/bin/active_response.py"
INPUT_LOG="${WAZUH_HOME}/logs/soc-ready-ids-input.log"
ENRICHED_LOG="${WAZUH_HOME}/logs/soc-ready-ids-enriched.json"
DB_PATH="${PROJECT_ROOT}/data/runtime/${DATASET}/alerts.db"

log() {
  printf '[soc-ready-ids] %s\n' "$*"
}

fail() {
  printf '[soc-ready-ids] ERROR: %s\n' "$*" >&2
  exit 1
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    fail "Run this script with sudo."
  fi
}

require_wazuh() {
  [[ -d "${WAZUH_HOME}" ]] || fail "Wazuh home not found at ${WAZUH_HOME}."
  [[ -f "${OSSEC_CONF}" ]] || fail "Wazuh manager config not found at ${OSSEC_CONF}."
  id wazuh >/dev/null 2>&1 || fail "Linux user 'wazuh' was not found."
}

copy_project() {
  [[ -f "${SOURCE_DIR}/config.yaml" ]] || fail "config.yaml not found in ${SOURCE_DIR}."
  [[ -d "${SOURCE_DIR}/src" ]] || fail "src directory not found in ${SOURCE_DIR}."

  mkdir -p "${PROJECT_ROOT}"
  if [[ "$(cd "${SOURCE_DIR}" && pwd)" != "$(cd "${PROJECT_ROOT}" && pwd)" ]]; then
    log "Copying project to ${PROJECT_ROOT}"
    rsync -a \
      --exclude .git \
      --exclude .venv \
      --exclude htmlcov \
      --exclude .pytest_cache \
      --exclude data/runtime \
      "${SOURCE_DIR}/" "${PROJECT_ROOT}/"
  else
    log "Project already running from ${PROJECT_ROOT}"
  fi
}

install_dependencies() {
  if [[ "${INSTALL_DEPS}" != "1" ]]; then
    log "Skipping Python dependency installation because INSTALL_DEPS=${INSTALL_DEPS}"
    return
  fi

  log "Installing Ubuntu packages"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    rsync

  log "Creating Python environment"
  python3 -m venv "${PROJECT_ROOT}/.venv"
  "${PROJECT_ROOT}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
  "${PROJECT_ROOT}/.venv/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements-wazuh.txt"
}

verify_artifacts() {
  log "Checking combined model artifacts"
  [[ "${DATASET}" == "combined" ]] || log "Dataset is ${DATASET}; combined is recommended for the full two-dataset project."
  [[ -f "${PROJECT_ROOT}/models/saved/${DATASET}/best_model.json" ]] || fail "Missing best_model.json for ${DATASET}."
  [[ -f "${PROJECT_ROOT}/models/saved/${DATASET}/xgboost_ids.joblib" ]] || fail "Missing xgboost_ids.joblib for ${DATASET}."
  [[ -f "${PROJECT_ROOT}/data/processed/${DATASET}/scaler.joblib" ]] || fail "Missing scaler.joblib for ${DATASET}."
  [[ -f "${PROJECT_ROOT}/data/processed/${DATASET}/feature_columns.json" ]] || fail "Missing feature_columns.json for ${DATASET}."
  [[ -f "${PROJECT_ROOT}/data/processed/${DATASET}/selected_feature_columns.json" ]] || fail "Missing selected_feature_columns.json for ${DATASET}."
}

install_wazuh_files() {
  log "Installing Wazuh decoder, rules, active response, and logs"
  install -o root -g wazuh -m 660 \
    "${PROJECT_ROOT}/wazuh/custom_decoder.xml" \
    "${WAZUH_HOME}/etc/decoders/soc_ready_ids_decoder.xml"
  install -o root -g wazuh -m 660 \
    "${PROJECT_ROOT}/wazuh/custom_rules.xml" \
    "${WAZUH_HOME}/etc/rules/soc_ready_ids_rules.xml"
  install -o root -g wazuh -m 750 \
    "${PROJECT_ROOT}/wazuh/active_response.py" \
    "${ACTIVE_RESPONSE_BIN}"
  sed -i "1c #!${PROJECT_ROOT}/.venv/bin/python" "${ACTIVE_RESPONSE_BIN}"

  touch "${INPUT_LOG}" "${ENRICHED_LOG}"
  chown root:wazuh "${INPUT_LOG}" "${ENRICHED_LOG}"
  chmod 660 "${INPUT_LOG}" "${ENRICHED_LOG}"

  mkdir -p \
    "${PROJECT_ROOT}/data/runtime/${DATASET}" \
    "${PROJECT_ROOT}/results/${DATASET}/shap" \
    "${PROJECT_ROOT}/results/${DATASET}/explanations"
  chown -R root:wazuh \
    "${PROJECT_ROOT}/data/runtime" \
    "${PROJECT_ROOT}/results/${DATASET}"
  chmod -R g+rwX \
    "${PROJECT_ROOT}/data/runtime" \
    "${PROJECT_ROOT}/results/${DATASET}"
}

patch_ossec_conf() {
  log "Updating ${OSSEC_CONF}"
  python3 - "${OSSEC_CONF}" "${PROJECT_ROOT}" "${DATASET}" "${INPUT_LOG}" "${ENRICHED_LOG}" "${DB_PATH}" <<'PY'
from __future__ import annotations

import copy
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

conf_path = Path(sys.argv[1])
project_root = sys.argv[2]
dataset = sys.argv[3]
input_log = sys.argv[4]
enriched_log = sys.argv[5]
db_path = sys.argv[6]

raw_text = conf_path.read_text(encoding="utf-8")


def child_text(element: ET.Element, name: str) -> str:
    child = element.find(name)
    return "" if child is None or child.text is None else child.text.strip()


def is_soc_ready_block(element: ET.Element) -> bool:
    tag = element.tag
    if tag == "localfile" and child_text(element, "location") in {input_log, enriched_log}:
        return True
    if tag == "command" and child_text(element, "name") == "soc-ready-ids-triage":
        return True
    if tag == "active-response" and child_text(element, "command") == "soc-ready-ids-triage":
        return True
    return False


def load_config_children(text: str) -> list[ET.Element]:
    try:
        parsed = ET.fromstring(text)
        if parsed.tag != "ossec_config":
            raise SystemExit(f"{conf_path} root element must be <ossec_config>.")
        return [copy.deepcopy(child) for child in list(parsed)]
    except ET.ParseError:
        # Earlier manual setup sometimes left multiple top-level blocks or
        # project blocks after </ossec_config>. Wazuh may tolerate fragments in
        # some cases, but ElementTree needs one root, so merge valid fragments.
        wrapper = ET.fromstring(f"<soc_ready_ids_wrapper>{text}</soc_ready_ids_wrapper>")
        children: list[ET.Element] = []
        for element in list(wrapper):
            if element.tag == "ossec_config":
                children.extend(copy.deepcopy(child) for child in list(element))
            else:
                children.append(copy.deepcopy(element))
        return children


root = ET.Element("ossec_config")
for child in load_config_children(raw_text):
    if not is_soc_ready_block(child):
        root.append(child)

for element in list(root):
    if is_soc_ready_block(element):
        root.remove(element)


def append_text(parent: ET.Element, name: str, text: str) -> ET.Element:
    child = ET.SubElement(parent, name)
    child.text = text
    return child


localfile = ET.SubElement(root, "localfile")
append_text(localfile, "location", input_log)
append_text(localfile, "log_format", "syslog")

localfile = ET.SubElement(root, "localfile")
append_text(localfile, "location", enriched_log)
append_text(localfile, "log_format", "json")

command = ET.SubElement(root, "command")
append_text(command, "name", "soc-ready-ids-triage")
append_text(command, "executable", "active_response.py")
append_text(
    command,
    "extra_args",
    f"--project-root={project_root} --config={project_root}/config.yaml "
    f"--dataset={dataset} --db={db_path} --output-log={enriched_log}",
)
append_text(command, "timeout_allowed", "no")

active_response = ET.SubElement(root, "active-response")
append_text(active_response, "disabled", "no")
append_text(active_response, "command", "soc-ready-ids-triage")
append_text(active_response, "location", "local")
append_text(active_response, "rules_group", "soc_ready_ids_raw")

backup = conf_path.with_suffix(conf_path.suffix + f".soc-ready-ids.{int(time.time())}.bak")
shutil.copy2(conf_path, backup)
tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
tree.write(conf_path, encoding="unicode", xml_declaration=False)
conf_path.write_text(conf_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
print(f"Backed up previous config to {backup}")
PY
}

validate_project_runtime() {
  log "Checking Python runtime and model loading"
  PYTHONPATH="${PROJECT_ROOT}/src" "${PROJECT_ROOT}/.venv/bin/python" - "${PROJECT_ROOT}" "${DATASET}" "${DB_PATH}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(sys.argv[1])
dataset = sys.argv[2]
db_path = Path(sys.argv[3])

from soc_ready_ids.config import load_config
from soc_ready_ids.triage.triage_pipeline import TriagePipeline

config = load_config(project_root / "config.yaml").for_dataset(dataset)
pipeline = TriagePipeline(config=config, db_path=db_path)
artifact = pipeline.model_artifact or {}
model_name = artifact.get("model_name")
if model_name != "xgboost_ids":
    raise SystemExit(f"Expected xgboost_ids model, loaded: {model_name!r}")
print(f"Loaded {model_name} with {len(pipeline.feature_columns)} selected features.")
PY
}

validate_and_restart_wazuh() {
  log "Validating Wazuh manager configuration"
  if [[ -x "${WAZUH_HOME}/bin/wazuh-analysisd" ]]; then
    "${WAZUH_HOME}/bin/wazuh-analysisd" -t
  fi

  log "Restarting Wazuh manager"
  systemctl restart wazuh-manager
  systemctl --no-pager --lines=20 status wazuh-manager
}

main() {
  require_root
  require_wazuh
  copy_project
  install_dependencies
  verify_artifacts
  install_wazuh_files
  patch_ossec_conf
  validate_project_runtime
  validate_and_restart_wazuh

  log "Deployment complete"
  log "Project root: ${PROJECT_ROOT}"
  log "Dataset/model: ${DATASET}"
  log "Raw input log: ${INPUT_LOG}"
  log "Enriched alert log: ${ENRICHED_LOG}"
  log "SQLite alert DB: ${DB_PATH}"
  log "Dashboard filter: rule.groups: \"soc_ready_ids_enriched\""
}

main "$@"
