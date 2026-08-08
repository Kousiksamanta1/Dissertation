# Results Summary

This file is generated from the current pipeline artifacts. The active dataset
is **combined** with 100000
retained rows and 118 encoded features.

> Results use the configured real dataset sample. Record the row cap,
> sampling procedure, and class distribution when reporting findings.

## IDS Model Comparison

| model           |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_macro_ovr |   binary_attack_f1_macro |   binary_attack_roc_auc |
|:----------------|-----------:|------------------:|---------------:|-----------:|--------------------:|-------------------------:|------------------------:|
| xgboost_ids     |    0.99875 |          0.985561 |       0.983466 |   0.984508 |            0.99996  |                 0.998934 |                0.999988 |
| random_forest   |    0.99835 |          0.971933 |       0.989405 |   0.980162 |            0.999846 |                 0.998376 |                0.999957 |
| autoencoder_ids |    0.9537  |          0.95285  |       0.953183 |   0.953014 |            0.975507 |                 0.953014 |                0.975507 |

The selected downstream model is **xgboost_ids**, chosen by
`f1_macro` with score
**0.9845**.

## Triage Ground-Truth Evaluation

- Alerts before triage: 400
- Alerts after triage: 88
- Alert reduction rate: 78.0%
- True-positive preservation rate: 100.0%
- False-negative rate for unrepresented attacks: 0.0%
- Mean explanation length: 53.28 words
- Feature coverage: 1.0

Evaluation design: Held-out dataset flows repeated in controlled four-alert bursts.

## Explanation Quality

- Completeness: 1.0
- Actionability: 0.859
- Conciseness: 1.0
- Overall SCS: 0.953
- Evaluated explanations: 88


## Unified Model by Source Dataset

The same selected model was evaluated separately on held-out rows from each original dataset.

| source_dataset   |   test_rows |   accuracy |   f1_macro |
|:-----------------|------------:|-----------:|-----------:|
| bot-iot          |        9973 |   0.9999   |   0.999948 |
| cicids2017       |       10027 |   0.997606 |   0.969577 |


## Ablation Study

| scenario         |   alerts_before |   alerts_after |   alert_reduction_rate |   true_positive_preservation_rate |   false_negative_rate_suppressed_real_attacks |   mean_explanation_length |   feature_coverage |   mean_visible_risk |
|:-----------------|----------------:|---------------:|-----------------------:|----------------------------------:|----------------------------------------------:|--------------------------:|-------------------:|--------------------:|
| full_pipeline    |             400 |             88 |                  78    |                               100 |                                             0 |                     53.28 |                  1 |              75.568 |
| no_clustering    |             400 |            100 |                  75    |                               100 |                                             0 |                     53.13 |                  1 |              75.23  |
| no_deduplication |             400 |            146 |                  63.5  |                               100 |                                             0 |                     54.2  |                  1 |              74.123 |
| no_risk_scoring  |             400 |             89 |                  77.75 |                               100 |                                             0 |                     53.22 |                  1 |              50     |

## Generated Evidence

- Model metrics and prediction tables: `results/combined/metrics/`
- SHAP summary, importance, and waterfall plots: `results/combined/shap/`
- LIME representative HTML reports: `results/combined/lime/`
- Publication PNG/PDF figures: `results/combined/figures/`
- Wazuh integration and OpenSearch guide: `wazuh/`
