# Results Summary

This file is generated from the current pipeline artifacts. The active dataset
is **bot-iot** with 100000
retained rows and 32 encoded features.

> Results use the configured real dataset sample. Record the row cap,
> sampling procedure, and class distribution when reporting findings.

## IDS Model Comparison

| model           |   accuracy |   precision_macro |   recall_macro |    f1_macro |   roc_auc_macro_ovr |   binary_attack_f1_macro |   binary_attack_roc_auc |
|:----------------|-----------:|------------------:|---------------:|------------:|--------------------:|-------------------------:|------------------------:|
| random_forest   |    1       |          1        |       1        | 1           |            1        |              1           |                1        |
| xgboost_ids     |    1       |          1        |       1        | 1           |            1        |              1           |                1        |
| autoencoder_ids |    0.00035 |          0.416717 |       0.333458 | 0.000349978 |            0.239669 |              0.000349978 |                0.239669 |

The selected downstream model is **random_forest**, chosen by
`f1_macro` with score
**1.0000**.

## Triage Ground-Truth Evaluation

- Alerts before triage: 400
- Alerts after triage: 92
- Alert reduction rate: 77.0%
- True-positive preservation rate: 100.0%
- False-negative rate for unrepresented attacks: 0.0%
- Mean explanation length: 51.21 words
- Feature coverage: 1.0

Evaluation design: Held-out dataset flows repeated in controlled four-alert bursts.

## Explanation Quality

- Completeness: 1.0
- Actionability: 1.0
- Conciseness: 1.0
- Overall SCS: 1.0
- Evaluated explanations: 92



## Ablation Study

| scenario         |   alerts_before |   alerts_after |   alert_reduction_rate |   true_positive_preservation_rate |   false_negative_rate_suppressed_real_attacks |   mean_explanation_length |   feature_coverage |   mean_visible_risk |
|:-----------------|----------------:|---------------:|-----------------------:|----------------------------------:|----------------------------------------------:|--------------------------:|-------------------:|--------------------:|
| full_pipeline    |             400 |             92 |                  77    |                               100 |                                             0 |                     51.21 |                  1 |              81.346 |
| no_clustering    |             400 |            100 |                  75    |                               100 |                                             0 |                     51.14 |                  1 |              81.118 |
| no_deduplication |             400 |            147 |                  63.25 |                               100 |                                             0 |                     51.12 |                  1 |              81.434 |
| no_risk_scoring  |             400 |             93 |                  76.75 |                               100 |                                             0 |                     51.19 |                  1 |              50     |

## Generated Evidence

- Model metrics and prediction tables: `results/bot-iot/metrics/`
- SHAP summary, importance, and waterfall plots: `results/bot-iot/shap/`
- LIME representative HTML reports: `results/bot-iot/lime/`
- Publication PNG/PDF figures: `results/bot-iot/figures/`
- Wazuh integration and OpenSearch guide: `wazuh/`
