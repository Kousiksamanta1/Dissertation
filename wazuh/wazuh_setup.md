# Wazuh Installation and SOC-Ready IDS Integration

This guide deploys the research pipeline on a Linux Wazuh manager. It follows
the current Wazuh 4.14 quickstart and active-response conventions.

Official references:

- https://documentation.wazuh.com/current/quickstart.html
- https://documentation.wazuh.com/current/user-manual/capabilities/active-response/custom-active-response-scripts.html
- https://documentation.wazuh.com/current/user-manual/ruleset/decoders/custom.html
- https://documentation.wazuh.com/current/user-manual/ruleset/rules/custom.html

## 1. Install Wazuh

Use a supported Linux host sized according to the official quickstart:

```bash
curl -sO https://packages.wazuh.com/4.14/wazuh-install.sh
sudo bash ./wazuh-install.sh -a
```

Record the dashboard credentials printed by the installer. Confirm services:

```bash
sudo systemctl status wazuh-manager
sudo systemctl status wazuh-indexer
sudo systemctl status wazuh-dashboard
```

## 2. Deploy the Research Project

Deploy this repository at `/opt/soc-ready-ids`:

```bash
sudo mkdir -p /opt/soc-ready-ids
sudo rsync -a --exclude .venv ./ /opt/soc-ready-ids/
sudo python3.10 -m venv /opt/soc-ready-ids/.venv
sudo /opt/soc-ready-ids/.venv/bin/pip install -r /opt/soc-ready-ids/requirements.txt
```

Copy trained artifacts and processed transformers into the deployed project,
or run the unified real-data pipeline after placing both datasets on the host:

```bash
cd /opt/soc-ready-ids
sudo make unified PYTHON=.venv/bin/python
```

The supplied `wazuh/ossec.conf` selects `DATASET=combined`. For a
source-specific deployment, change it to `cicids2017` or `bot-iot`. Set
`SOC_READY_IDS_DATASET` to the same value when using an environment override.

## 3. Install Decoder and Rules

Wazuh reserves custom rule IDs `100000` through `120000`. This project uses
IDs `100100` through `100112`.

```bash
sudo cp wazuh/custom_decoder.xml /var/ossec/etc/decoders/soc_ready_ids_decoder.xml
sudo cp wazuh/custom_rules.xml /var/ossec/etc/rules/soc_ready_ids_rules.xml
sudo chown wazuh:wazuh /var/ossec/etc/decoders/soc_ready_ids_decoder.xml
sudo chown wazuh:wazuh /var/ossec/etc/rules/soc_ready_ids_rules.xml
sudo chmod 660 /var/ossec/etc/decoders/soc_ready_ids_decoder.xml
sudo chmod 660 /var/ossec/etc/rules/soc_ready_ids_rules.xml
```

Test decoding:

```bash
sudo /var/ossec/bin/wazuh-logtest
```

Paste:

```text
SOC_READY_IDS|dataset=CICIDS2017|attack_type=DDoS|srcip=10.0.0.5|dstip=192.168.10.50|dstport=80|confidence=0.95|asset_criticality=80|flow_packets_s=120|syn_count=5
```

The decoded event should contain `srcip`, `dstip`, `attack_type`, and the
remaining custom fields.

## 4. Install the Active Response

The script is stateless: it handles Wazuh `command: add`, runs the triage
pipeline, stores the result in SQLite, and appends enriched JSON to a monitored
log.

```bash
sudo cp wazuh/active_response.py /var/ossec/active-response/bin/active_response.py
sudo chown root:wazuh /var/ossec/active-response/bin/active_response.py
sudo chmod 750 /var/ossec/active-response/bin/active_response.py
```

Change the script shebang on the manager so it uses the project environment:

```bash
sudo sed -i '1c #!/opt/soc-ready-ids/.venv/bin/python3' \
  /var/ossec/active-response/bin/active_response.py
```

Merge the blocks in `wazuh/ossec.conf` into `/var/ossec/etc/ossec.conf`. Do
not add a second outer `<ossec_config>` around an existing configuration.

Create monitored logs:

```bash
sudo touch /var/ossec/logs/soc-ready-ids-input.log
sudo touch /var/ossec/logs/soc-ready-ids-enriched.json
sudo chown wazuh:wazuh /var/ossec/logs/soc-ready-ids-*.log
sudo chown wazuh:wazuh /var/ossec/logs/soc-ready-ids-enriched.json
sudo chmod 660 /var/ossec/logs/soc-ready-ids-input.log
sudo chmod 660 /var/ossec/logs/soc-ready-ids-enriched.json
```

Restart and verify:

```bash
sudo systemctl restart wazuh-manager
sudo systemctl status wazuh-manager
sudo tail -f /var/ossec/logs/ossec.log
```

## 5. End-to-End Test

Inject a raw IDS event:

```bash
echo 'SOC_READY_IDS|dataset=CICIDS2017|attack_type=DDoS|srcip=10.0.0.5|dstip=192.168.10.50|dstport=80|confidence=0.95|asset_criticality=80|flow_packets_s=120|syn_count=5' \
  | sudo tee -a /var/ossec/logs/soc-ready-ids-input.log
```

Then inspect:

```bash
sudo tail -n 1 /var/ossec/logs/soc-ready-ids-enriched.json | python3 -m json.tool
sudo sqlite3 /opt/soc-ready-ids/data/runtime/combined/alerts.db \
  'SELECT alert_id, attack_type, risk_score, risk_tier FROM alerts ORDER BY timestamp DESC LIMIT 5;'
```

The enriched event is collected by Wazuh's JSON decoder and indexed for the
dashboard. The active response does not mutate the original Wazuh alert; it
creates a correlated enrichment event, which is the supported observable
return path.

## 6. Local Validation Without Wazuh

From this repository:

```bash
source .venv/bin/activate
make wazuh DATASET=cicids2017
make wazuh DATASET=bot-iot
make wazuh DATASET=combined
```

This parses the XML and processes `wazuh/mock_wazuh_alert.json` through the
actual active-response code using temporary SQLite and output files.
