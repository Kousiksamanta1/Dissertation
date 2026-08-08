# SOC-Ready Explainable IDS Triage

End-to-end dissertation implementation for intrusion detection, explainability,
alert prioritisation, and Wazuh/OpenSearch SOC integration. CICIDS2017 is the
primary dataset and BoT-IoT is the secondary dataset. Evaluation uses dataset
ground truth only; there are no human participants.

Ethics approval: **Ref 77771/2026**.

## Implemented Phases

1. CICIDS2017 CSV/Parquet and BoT-IoT CSV loading, preprocessing, scaling,
   stratified splitting, mutual information, RFE, and top-20 features.
2. Random Forest with SMOTE, 50-trial Optuna XGBoost, and PyTorch autoencoder.
3. Global/local SHAP, LIME reports, and analyst-facing explanations.
4. Risk scoring, clustering, deduplication, storm detection, and SQLite.
5. Wazuh decoder, rules, active response, configuration, and dashboard guide.
6. IDS/triage metrics, ablation, explanation SCS, and publication figures.
7. Pytest unit/integration suite with a coverage gate of at least 95%.

No generated dataset fallback, Streamlit, or NSL-KDD components are used.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
make install
make doctor
```

The Makefile automatically uses `.venv/bin/python` when it exists, including
when the shell prompt still displays Conda `(base)`.

## Dataset Download And Locations

Download the datasets from their official providers, extract them, and place
the machine-learning files in these project directories:

- CICIDS2017: `CIC-IDS2017/`
- BoT-IoT: `BOT-IOT/`

These locations are configured in `config.yaml`. CICIDS2017 accepts CSV and
Parquet files. BoT-IoT accepts its CSV chunks and ignores `data_names.csv`.

Official sources:

- [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [BoT-IoT](https://research.unsw.edu.au/projects/bot-iot-dataset)

Expected local layout:

```text
CIC-IDS2017/
|-- Benign-Monday-no-metadata.parquet
|-- Bruteforce-Tuesday-no-metadata.parquet
|-- DoS-Wednesday-no-metadata.parquet
|-- WebAttacks-Thursday-no-metadata.parquet
|-- Infiltration-Thursday-no-metadata.parquet
|-- Botnet-Friday-no-metadata.parquet
|-- DDoS-Friday-no-metadata.parquet
`-- Portscan-Friday-no-metadata.parquet

BOT-IOT/
|-- data_names.csv
|-- data_1.csv
|-- data_2.csv
`-- ...
```

Use the provider terms when downloading. For CICIDS2017, download the
machine-learning CSV files from the CIC dataset page, or use the equivalent
Parquet files if already converted. For BoT-IoT, download the UNSW BoT-IoT CSV
chunks and keep the chunk filenames together with `data_names.csv`. After
placing the files, run:

```bash
make doctor
make data DATASET=cicids2017
make data DATASET=bot-iot
make data DATASET=combined
```

The original prompt listed `wazuh-api`; the implementation uses Wazuh active
response plus `requests`-based HTTP capability instead because there is no
installable PyPI package named `wazuh-api` available in this environment.

## Run One Unified Model

Train one simultaneous model from both datasets after common-label
harmonization:

```bash
make unified
```

This is equivalent to:

```bash
make pipeline DATASET=combined
```

The unified run allocates half of its configured rows to CICIDS2017 and half
to BoT-IoT. By default, `data.combined_feature_mode: union` trains one model
on the source-native CICIDS2017 and BoT-IoT schemas plus 15 derived shared flow
features. Label-derived fields such as `Label`, `attack`, `category`, and
`subcategory` are dropped before training. Labels are mapped to `BENIGN`,
`DDoS`, `DoS`, `Reconnaissance`, and `Other Attack`; dataset identity is kept
as evaluation metadata and is not supplied to the model.

Set `data.combined_feature_mode: shared` to reproduce the older stricter
15-feature shared-schema experiment. That version is easier to interpret but
does not model the heterogeneous CICIDS2017 `Other Attack` class as strongly as
the default source-native unified model.

## Run All Experiments

Run the two independent baselines, the unified model, and the comparison
report:

```bash
make all
```

This runs:

```bash
make pipeline DATASET=cicids2017
make pipeline DATASET=bot-iot
make pipeline DATASET=combined
make summary
```

The independent baselines use deterministic 100,000-row samples. The unified
run uses 100,000 rows total, split equally into 50,000 CICIDS2017 and 50,000
BoT-IoT rows, because this machine has 8 GB RAM. Sampling spans every source
file and preserves available CICIDS attack classes. The independent baselines
select the top 20 features; the unified model selects the top 80 source-native
features to keep five-class macro-F1 above 96% on the held-out combined split.
Override a cap with:

```bash
make pipeline DATASET=cicids2017 MAX_ROWS=250000
```

Setting a dataset's `data.max_rows` value to `null` requests all rows and can
require substantially more memory. The supplied BoT-IoT chunks are extremely
attack-heavy, so inspect the saved class distribution before interpreting
accuracy or false-positive metrics.

## Commands

```bash
make doctor
make pipeline DATASET=cicids2017
make pipeline DATASET=bot-iot
make pipeline DATASET=combined
make unified
make all
make explain DATASET=cicids2017
make quick-train DATASET=bot-iot
make summary
make wazuh DATASET=cicids2017
make wazuh DATASET=bot-iot
make wazuh DATASET=combined
make api DATASET=combined
make test
```

## Dataset-Isolated Outputs

Each run is preserved separately:

```text
data/processed/cicids2017/
data/processed/bot-iot/
data/processed/combined/
models/saved/cicids2017/
models/saved/bot-iot/
models/saved/combined/
results/cicids2017/
results/bot-iot/
results/combined/
reports/cicids2017/results_summary.md
reports/bot-iot/results_summary.md
reports/combined/results_summary.md
```

The cross-dataset comparison is generated at:

```text
reports/results_summary.md
```

Dataset-specific SQLite files are stored under `data/runtime/<dataset>/`.

## API

Start the API with the unified model:

```bash
make api DATASET=combined
```

Main endpoints:

- `GET /` - API status and endpoint list
- `GET /health` - lightweight health check
- `POST /predict`
- `POST /predict/batch`
- `GET /alerts`
- `GET /alert/<alert_id>`
- `POST /feedback`
- `GET /stats`

After starting the API, open `http://127.0.0.1:5000/` or run:

```bash
curl http://127.0.0.1:5000/health
curl http://127.0.0.1:5000/stats
```

## Wazuh

Deployment instructions are in
[`wazuh/wazuh_setup.md`](wazuh/wazuh_setup.md), with dashboard construction in
[`wazuh/opensearch_dashboard.md`](wazuh/opensearch_dashboard.md). Final
dissertation evidence to capture from a real deployment is listed in
[`reports/wazuh_opensearch_evidence_checklist.md`](reports/wazuh_opensearch_evidence_checklist.md).

For the full Wazuh deployment on the Ubuntu manager, use the combined
CICIDS2017 plus BoT-IoT model:

```bash
cd /opt/soc-ready-ids
sudo bash wazuh/deploy_project.sh
```

This installs the real model-backed active response and enriched alert
collection under Wazuh. It does not retrain the datasets or run experiments.
Dashboard fields such as `data.attack_type`, `data.risk_score`, and
`data.explanation_text` appear after Wazuh indexes enriched project alerts.

To connect a live IDS source after the dashboard is working:

```bash
cd /opt/soc-ready-ids
sudo bash wazuh/install_suricata_bridge.sh
```

This starts Suricata and a bridge that sends Suricata EVE alerts through the
same SOC-ready Wazuh enrichment path.

Official references:

- [Wazuh quickstart](https://documentation.wazuh.com/current/quickstart.html)
- [Custom active response](https://documentation.wazuh.com/current/user-manual/capabilities/active-response/custom-active-response-scripts.html)
- [Custom decoders](https://documentation.wazuh.com/current/user-manual/ruleset/decoders/custom.html)
- [Custom rules](https://documentation.wazuh.com/current/user-manual/ruleset/rules/custom.html)
- [Custom dashboards](https://documentation.wazuh.com/current/user-manual/wazuh-dashboard/creating-custom-dashboards.html)

## Structure

```text
.
|-- CIC-IDS2017/
|-- BOT-IOT/
|-- config.yaml
|-- Makefile
|-- requirements.txt
|-- data/{processed,runtime,external}/
|-- models/saved/<dataset>/
|-- notebooks/
|-- reports/<dataset>/
|-- results/<dataset>/
|-- src/soc_ready_ids/
|-- tests/
`-- wazuh/
```
