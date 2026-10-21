# Evaluation Report: Legacy Wav2Vec 2.0 + MLP

- **Protocol**: `sliding_window`
- **Manifest**: `manifest_local_eval.csv` (6 samples)
- **Device**: `cpu`
- **Score Orientation**: `higher_is_fake`

## Overall Metrics

| Metric | Value |
|---|---|
| **EER** | **75.00%** |
| **ROC AUC** | **0.5000** |
| **PR AUC** | **0.4167** |
| **Accuracy** | 33.33% |
| **Precision** | 33.33% |
| **Recall (Spoof)** | 100.00% |
| **F1 Score** | 0.5000 |
| **Genuine False Positive Rate** | 100.00% |
| **Operating Threshold** | 0.9850 (model_calibrated_default) |
| **Throughput** | 0.7 samples/sec |

## Confusion Matrix

| | Predicted Genuine | Predicted Fake |
|---|---|---|
| **Actual Genuine** | 0 | 4 |
| **Actual Fake** | 0 | 2 |

## Per-Attack Breakdown

| Attack Type | Total Samples | Detected Spoofs | Recall | EER |
|---|---|---|---|---|
| `bonafide (-)` | 4 | 0 | 0.0% | N/A |
| `voice_clone` | 1 | 1 | 100.0% | 75.00% |
| `voice_clone_or_deepfake` | 1 | 1 | 100.0% | 75.00% |
