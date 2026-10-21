# Evaluation Report: Legacy Wav2Vec 2.0 + MLP

- **Protocol**: `native_benchmark`
- **Manifest**: `manifest_asvspoof2019_eval.csv` (71237 samples)
- **Device**: `cpu`
- **Score Orientation**: `higher_is_fake`

## Overall Metrics

| Metric | Value |
|---|---|
| **EER** | **7.13%** |
| **ROC AUC** | **0.9815** |
| **PR AUC** | **0.9978** |
| **Accuracy** | 84.90% |
| **Precision** | 99.80% |
| **Recall (Spoof)** | 83.33% |
| **F1 Score** | 0.9083 |
| **Genuine False Positive Rate** | 1.47% |
| **Operating Threshold** | 0.9850 (validation_eer_selected (manifest_asvspoof2019_dev)) |
| **Throughput** | 174901.8 samples/sec |

## Confusion Matrix

| | Predicted Genuine | Predicted Fake |
|---|---|---|
| **Actual Genuine** | 7247 | 108 |
| **Actual Fake** | 10647 | 53235 |

## Per-Attack Breakdown

| Attack Type | Total Samples | Detected Spoofs | Recall | EER |
|---|---|---|---|---|
| `bonafide (-)` | 7355 | 0 | 0.0% | N/A |
| `A07` | 4914 | 4788 | 97.4% | 1.89% |
| `A08` | 4914 | 4843 | 98.6% | 1.44% |
| `A09` | 4914 | 4907 | 99.9% | 0.70% |
| `A10` | 4914 | 4812 | 97.9% | 1.79% |
| `A11` | 4914 | 4759 | 96.8% | 2.07% |
| `A12` | 4914 | 4868 | 99.1% | 1.31% |
| `A13` | 4914 | 4911 | 99.9% | 0.37% |
| `A14` | 4914 | 4891 | 99.5% | 0.93% |
| `A15` | 4914 | 4322 | 88.0% | 4.55% |
| `A16` | 4914 | 4334 | 88.2% | 5.29% |
| `A17` | 4914 | 1331 | 27.1% | 18.93% |
| `A18` | 4914 | 1873 | 38.1% | 13.42% |
| `A19` | 4914 | 2596 | 52.8% | 12.67% |
