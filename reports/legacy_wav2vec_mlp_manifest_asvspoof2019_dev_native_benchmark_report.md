# Evaluation Report: Legacy Wav2Vec 2.0 + MLP

- **Protocol**: `native_benchmark`
- **Manifest**: `manifest_asvspoof2019_dev.csv` (24844 samples)
- **Device**: `cpu`
- **Score Orientation**: `higher_is_fake`

## Overall Metrics

| Metric | Value |
|---|---|
| **EER** | **3.47%** |
| **ROC AUC** | **0.9947** |
| **PR AUC** | **0.9994** |
| **Accuracy** | 96.51% |
| **Precision** | 99.59% |
| **Recall (Spoof)** | 96.51% |
| **F1 Score** | 0.9803 |
| **Genuine False Positive Rate** | 3.45% |
| **Operating Threshold** | 0.9850 (model_calibrated_default) |
| **Throughput** | 165086.9 samples/sec |

## Confusion Matrix

| | Predicted Genuine | Predicted Fake |
|---|---|---|
| **Actual Genuine** | 2460 | 88 |
| **Actual Fake** | 778 | 21518 |

## Per-Attack Breakdown

| Attack Type | Total Samples | Detected Spoofs | Recall | EER |
|---|---|---|---|---|
| `bonafide (-)` | 2548 | 0 | 0.0% | N/A |
| `A01` | 3716 | 3711 | 99.9% | 0.33% |
| `A02` | 3716 | 3715 | 100.0% | 0.33% |
| `A03` | 3716 | 3707 | 99.8% | 0.81% |
| `A04` | 3716 | 3389 | 91.2% | 5.92% |
| `A05` | 3716 | 3714 | 99.9% | 1.07% |
| `A06` | 3716 | 3282 | 88.3% | 6.55% |
