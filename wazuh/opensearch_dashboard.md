# Wazuh OpenSearch Dashboard Guide

The Wazuh dashboard is the SOC interface for this project. Follow the official
custom-dashboard workflow:

https://documentation.wazuh.com/current/user-manual/wazuh-dashboard/creating-custom-dashboards.html

## Data View

1. Open **Dashboard management > Dashboards Management > Data views**.
2. Use the Wazuh alerts data view, normally `wazuh-alerts-*`.
3. Set `timestamp` or `@timestamp` as the time field.
4. Filter the dashboard with:

```text
rule.groups: "soc_ready_ids_enriched"
```

Enriched events expose these useful fields:

```text
data.alert_id
data.attack_type
data.confidence
data.risk_score
data.risk_tier
data.cluster_id
data.is_suppressed
data.triage_duration_ms
data.explanation_text
data.recommended_action
data.shap_plot_path
data.src_ip
data.dst_ip
data.dst_port
```

Depending on the Wazuh version and JSON mapping, dynamic fields can appear
without the `data.` prefix. Confirm names in **Discover** before creating
visualizations.

## Saved Search: Risk-Sorted Alert Feed

1. Open **Discover** and apply the enriched-event filter.
2. Add timestamp, attack type, source, destination, risk score, risk tier,
   explanation, and recommended action columns.
3. Sort by `risk_score` descending.
4. Save as `SOC Ready - Risk Alert Feed`.

## Explanation Panel

Create a table visualization containing:

- `alert_id`
- `attack_type`
- `explanation_text`
- `recommended_action`
- `shap_plot_path`

Wazuh/OpenSearch cannot render a local filesystem SHAP PNG directly. Publish
the relevant `results/<dataset>/shap/` directory through an authenticated
internal web service or object store, then index that HTTPS URL as
`shap_plot_url` if inline images are required. Otherwise, retain the path as
evidence and show the explanation text in the dashboard.

## KPI Cards

Create metric visualizations:

1. **Total alerts**: document count.
2. **Suppressed percent**: filter `is_suppressed: true`, divide by total with a
   formula visualization.
3. **True-positive rate**: use ground-truth evaluation output for dissertation
   reporting; analyst feedback can provide an operational dashboard estimate.
4. **Mean triage time**: average `triage_duration_ms`.

## Alert Volume Time Series

Create a line or area chart:

- Horizontal axis: date histogram on timestamp.
- Vertical axis: document count.
- Break down by `attack_type` or `risk_tier`.

## Cluster Scatter Plot

The active-response path stores `cluster_id`; batch/API processing also stores
`cluster_x` and `cluster_y`. Create a scatter plot:

- X axis: `cluster_x`
- Y axis: `cluster_y`
- Color: `attack_type`
- Size: `risk_score`
- Filter: fields exist

Single-alert active responses have no batch context, so cluster coordinates are
zero until alerts are processed in a batch or by an external scheduled
clustering job.

## Dashboard Assembly

Create `SOC-Ready Explainable IDS Triage` and add:

1. KPI cards across the top.
2. Alert volume time series.
3. Risk-sorted saved search.
4. Explanation table.
5. Cluster scatter plot.

Set automatic refresh to 10 seconds for demonstrations. Use a larger interval
for production to reduce indexer load.
