# Results Summary

This file is generated from the current pipeline artifacts. The active dataset
is **cicids2017** with 100000
retained rows and 69 encoded features.

> Results use the configured real dataset sample. Record the row cap,
> sampling procedure, and class distribution when reporting findings.

## IDS Model Comparison

| model           |   accuracy |   precision_macro |   recall_macro |   f1_macro |   roc_auc_macro_ovr |   binary_attack_f1_macro |   binary_attack_roc_auc |
|:----------------|-----------:|------------------:|---------------:|-----------:|--------------------:|-------------------------:|------------------------:|
| random_forest   |    0.99505 |          0.78295  |       0.838192 |   0.804665 |            0.979142 |                 0.99156  |                0.999335 |
| autoencoder_ids |    0.9143  |          0.798088 |       0.810812 |   0.804248 |            0.879503 |                 0.804248 |                0.879503 |
| xgboost_ids     |    0.99635 |          0.775639 |       0.699587 |   0.721303 |            0.999544 |                 0.992481 |                0.999878 |

The selected downstream model is **random_forest**, chosen by
`f1_macro` with score
**0.8047**.

## Triage Ground-Truth Evaluation

- Alerts before triage: 400
- Alerts after triage: 92
- Alert reduction rate: 77.0%
- True-positive preservation rate: 100.0%
- False-negative rate for unrepresented attacks: 0.0%
- Mean explanation length: 58.97 words
- Feature coverage: 1.0

Evaluation design: Held-out dataset flows repeated in controlled four-alert bursts.

## Explanation Quality

- Completeness: 1.0
- Actionability: 0.67
- Conciseness: 1.0
- Overall SCS: 0.89
- Evaluated explanations: 92



## Ablation Study

| scenario         |   alerts_before |   alerts_after |   alert_reduction_rate |   true_positive_preservation_rate |   false_negative_rate_suppressed_real_attacks |   mean_explanation_length |   feature_coverage |   mean_visible_risk |
|:-----------------|----------------:|---------------:|-----------------------:|----------------------------------:|----------------------------------------------:|--------------------------:|-------------------:|--------------------:|
| full_pipeline    |             400 |             92 |                     77 |                               100 |                                             0 |                     58.97 |                  1 |              67.48  |
| no_clustering    |             400 |            100 |                     75 |                               100 |                                             0 |                     58.98 |                  1 |              67.195 |
| no_deduplication |             400 |            152 |                     62 |                               100 |                                             0 |                     59.12 |                  1 |              67.336 |
| no_risk_scoring  |             400 |             92 |                     77 |                               100 |                                             0 |                     58.99 |                  1 |              50     |

## Generated Evidence

- Model metrics and prediction tables: `results/cicids2017/metrics/`
- SHAP summary, importance, and waterfall plots: `results/cicids2017/shap/`
- LIME representative HTML reports: `results/cicids2017/lime/`
- Publication PNG/PDF figures: `results/cicids2017/figures/`
- Wazuh integration and OpenSearch guide: `wazuh/`
