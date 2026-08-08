# Combined Real-Dataset Results

CICIDS2017 and BoT-IoT were evaluated independently, and the combined run trained one unified model on both source schemas with a common label taxonomy. Artifacts are isolated by run.

| dataset    |   retained_rows |   classes | best_model    |   accuracy |   f1_macro |   binary_attack_f1_macro |   alert_reduction_rate |   tp_preservation_rate |   explanation_scs |
|:-----------|----------------:|----------:|:--------------|-----------:|-----------:|-------------------------:|-----------------------:|-----------------------:|------------------:|
| cicids2017 |          100000 |        15 | random_forest |    0.99505 |   0.804665 |                 0.99156  |                     77 |                    100 |             0.89  |
| bot-iot    |          100000 |         4 | random_forest |    1       |   1        |                 1        |                     77 |                    100 |             1     |
| combined   |          100000 |         5 | xgboost_ids   |    0.99875 |   0.984508 |                 0.998934 |                     78 |                    100 |             0.953 |

> Compare results with care: the configured samples preserve each dataset's observed class distribution, and the supplied BoT-IoT files contain very few benign rows. The combined run uses a broad common label taxonomy and equal row allocation from each source.

## cicids2017

- Dataset report: `reports/cicids2017/results_summary.md`
- Processed data: `data/processed/cicids2017/`
- Models: `models/saved/cicids2017/`
- Results: `results/cicids2017/`
- Classes: BENIGN, Bot, DDoS, DoS GoldenEye, DoS Hulk, DoS Slowhttptest, DoS slowloris, FTP-Patator, Heartbleed, Infiltration, PortScan, SSH-Patator, Web Attack � Brute Force, Web Attack � Sql Injection, Web Attack � XSS
- Class distribution: BENIGN=87793, DDoS=7270, DoS Hulk=3650, DoS GoldenEye=232, FTP-Patator=207, PortScan=201, Bot=124, Web Attack � Brute Force=117, SSH-Patator=113, DoS slowloris=106, DoS Slowhttptest=102, Web Attack � XSS=64, Infiltration=10, Web Attack � Sql Injection=6, Heartbleed=5

## bot-iot

- Dataset report: `reports/bot-iot/results_summary.md`
- Processed data: `data/processed/bot-iot/`
- Models: `models/saved/bot-iot/`
- Results: `results/bot-iot/`
- Classes: BENIGN, DDoS, DoS, Reconnaissance
- Class distribution: DDoS=52689, DoS=44607, Reconnaissance=2689, BENIGN=15

## combined

- Dataset report: `reports/combined/results_summary.md`
- Processed data: `data/processed/combined/`
- Models: `models/saved/combined/`
- Results: `results/combined/`
- Classes: BENIGN, DDoS, DoS, Other Attack, Reconnaissance
- Class distribution: BENIGN=43883, DDoS=30003, DoS=24326, Reconnaissance=1428, Other Attack=360
- Source distribution: cicids2017=50000, bot-iot=50000
