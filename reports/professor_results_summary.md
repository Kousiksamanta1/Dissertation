# SOC-Ready Explainable IDS Triage - Professor Results Summary

Generated: 27 July 2026

## Project Status

The SOC-Ready Explainable IDS Triage project is functionally complete. The
system supports CICIDS2017, BoT-IoT, and a combined simultaneous dataset mode.
It includes preprocessing, feature selection, model training, explainability,
triage, Wazuh integration files, generated reports, notebooks, and tests.

Overall completion: 95%.

Remaining work: live Wazuh/OpenSearch deployment and evidence screenshots.

## Main Results

| Dataset | Best model | Accuracy | F1-macro | Binary attack F1 | Alert reduction | TP preservation | Explanation SCS |
|---|---|---:|---:|---:|---:|---:|---:|
| CICIDS2017 | Random Forest | 99.51% | 80.47% | 99.16% | 77% | 100% | 89.0% |
| BoT-IoT | Random Forest | 100.00% | 100.00% | 100.00% | 77% | 100% | 100.0% |
| Combined CICIDS2017 + BoT-IoT | XGBoost | 99.88% | 98.45% | 99.89% | 78% | 100% | 95.3% |

## Combined Dataset Outcome

The main simultaneous-dataset target was achieved.

- Combined model F1-macro: 98.45%
- Combined model accuracy: 99.88%
- Combined binary attack F1: 99.89%
- Combined rows: 100,000 total
- CICIDS2017 rows: 50,000
- BoT-IoT rows: 50,000
- Common taxonomy: BENIGN, DDoS, DoS, Other Attack, Reconnaissance

## Testing Outcome

- Tests passed: 48
- Test coverage: 96.09%
- Coverage gate: 95%

## Wazuh/SOC Integration

Wazuh integration files are complete and local validation passes:

- `custom_decoder.xml`
- `custom_rules.xml`
- `active_response.py`
- `ossec.conf`
- `opensearch_dashboard.md`

The remaining operational step is to deploy Wazuh/OpenSearch on an authorised
Linux VM or server and capture the final dashboard/evidence screenshots.

## Interpretation Note

The combined model is the primary reported result because it trains one unified
IDS model on both CICIDS2017 and BoT-IoT simultaneously. CICIDS2017 15-class
macro-F1 is lower because rare attack classes are difficult under the configured
100,000-row sample, but binary attack detection remains high at 99.16%.

