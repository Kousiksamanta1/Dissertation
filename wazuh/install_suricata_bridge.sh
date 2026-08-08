#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/opt/soc-ready-ids}"
IFACE="${IFACE:-}"
ASSET_CRITICALITY="${ASSET_CRITICALITY:-70}"
INSTALL_SURICATA="${INSTALL_SURICATA:-1}"
EVE_LOG="${EVE_LOG:-/var/log/suricata/eve.json}"
OUTPUT_LOG="${OUTPUT_LOG:-/var/ossec/logs/soc-ready-ids-input.log}"
SERVICE_PATH="/etc/systemd/system/soc-ready-suricata-bridge.service"

log() {
  printf '[soc-ready-ids] %s\n' "$*"
}

fail() {
  printf '[soc-ready-ids] ERROR: %s\n' "$*" >&2
  exit 1
}

require_root() {
  [[ "${EUID}" -eq 0 ]] || fail "Run this script with sudo."
}

detect_interface() {
  if [[ -z "${IFACE}" ]]; then
    IFACE="$(ip -o -4 route show to default | awk '{print $5; exit}')"
  fi
  [[ -n "${IFACE}" ]] || fail "Could not detect the network interface. Set IFACE=enp0s1 and rerun."
  log "Using network interface: ${IFACE}"
}

install_suricata() {
  if [[ "${INSTALL_SURICATA}" != "1" ]]; then
    log "Skipping Suricata installation because INSTALL_SURICATA=${INSTALL_SURICATA}"
    return
  fi

  log "Installing Suricata"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    jq \
    software-properties-common
  if ! grep -R "oisf/suricata-stable" /etc/apt/sources.list /etc/apt/sources.list.d >/dev/null 2>&1; then
    add-apt-repository -y ppa:oisf/suricata-stable
    apt-get update
  fi
  DEBIAN_FRONTEND=noninteractive apt-get install -y suricata jq
}

set_default_value() {
  local key="$1"
  local value="$2"
  local file="/etc/default/suricata"
  touch "${file}"
  if grep -qE "^${key}=" "${file}"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "${file}"
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${file}"
  fi
}

configure_suricata() {
  log "Configuring Suricata service"
  cp -a /etc/default/suricata "/etc/default/suricata.soc-ready-ids.$(date +%s).bak" 2>/dev/null || true
  set_default_value "RUN" "yes"
  set_default_value "LISTENMODE" "af-packet"
  set_default_value "IFACE" "${IFACE}"

  if command -v suricata-update >/dev/null 2>&1; then
    log "Updating Suricata rules"
    suricata-update || log "suricata-update failed; continuing with installed rules"
  fi

  mkdir -p "$(dirname "${EVE_LOG}")" "$(dirname "${OUTPUT_LOG}")"
  touch "${EVE_LOG}" "${OUTPUT_LOG}"
  chmod 640 "${EVE_LOG}" || true
  chown root:wazuh "${OUTPUT_LOG}" 2>/dev/null || true
  chmod 660 "${OUTPUT_LOG}" 2>/dev/null || true

  systemctl enable --now suricata
  systemctl restart suricata
}

install_bridge_service() {
  [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]] || fail "Project virtualenv not found at ${PROJECT_ROOT}/.venv/bin/python. Run wazuh/deploy_project.sh first."
  [[ -f "${PROJECT_ROOT}/wazuh/suricata_to_soc_ready.py" ]] || fail "Bridge script not found. Run git pull origin main first."

  log "Installing bridge service"
  cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=SOC-ready IDS Suricata EVE bridge
After=suricata.service wazuh-manager.service
Wants=suricata.service

[Service]
Type=simple
User=root
Group=root
Environment=PYTHONUNBUFFERED=1
ExecStart=${PROJECT_ROOT}/.venv/bin/python ${PROJECT_ROOT}/wazuh/suricata_to_soc_ready.py --eve-log ${EVE_LOG} --output-log ${OUTPUT_LOG} --asset-criticality ${ASSET_CRITICALITY}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable --now soc-ready-suricata-bridge
  systemctl restart soc-ready-suricata-bridge
}

main() {
  require_root
  detect_interface
  install_suricata
  configure_suricata
  install_bridge_service

  log "Suricata live source is connected"
  log "Suricata EVE log: ${EVE_LOG}"
  log "SOC-ready input log: ${OUTPUT_LOG}"
  log "Check services with:"
  log "  systemctl status suricata --no-pager"
  log "  systemctl status soc-ready-suricata-bridge --no-pager"
}

main "$@"
