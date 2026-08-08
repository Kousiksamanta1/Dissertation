# Wazuh/OpenSearch Evidence Checklist

Use this checklist when capturing final dissertation evidence from a live
Wazuh/OpenSearch deployment. Local validation is covered by `make wazuh`; this
file is for screenshots or exported evidence from the actual SOC interface.

## Required Screenshots

1. Wazuh manager services running after restart.
2. `wazuh-logtest` decoding a `SOC_READY_IDS|...` event.
3. Enriched JSON event in `/var/ossec/logs/soc-ready-ids-enriched.json`.
4. OpenSearch Discover view filtered with `rule.groups: "soc_ready_ids_enriched"`.
5. Risk-sorted alert feed showing `risk_score`, `risk_tier`, `attack_type`,
   `explanation_text`, and `recommended_action`.
6. KPI cards: total alerts, suppressed percent, TP rate, and mean triage time.
7. Alert volume time-series.
8. Cluster scatter plot, if batch/API cluster coordinates are indexed.

## Exported Evidence

Save these artifacts with the dissertation appendix or viva material:

- Exported dashboard NDJSON, if dashboard export is permitted.
- One anonymised enriched alert JSON example.
- One SHAP waterfall PNG linked to the same alert.
- SQLite query output showing stored triage records.
- A short note identifying the dataset mode used for the deployment:
  `cicids2017`, `bot-iot`, or `combined`.

## Local Commands Already Covered

```bash
make wazuh DATASET=cicids2017
make wazuh DATASET=bot-iot
make wazuh DATASET=combined
```
