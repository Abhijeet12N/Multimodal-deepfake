# Audio Evaluation Protocol & Benchmark Documentation

## 1. Overview and Objectives

This document establishes the standardized, reproducible evaluation protocol for the audio deepfake detection pipeline upgrade, fulfilling the requirements of **Phase 2 — Build a Reproducible Evaluation Pipeline** from the system modernization plan.

The protocol enforces:
1. An immutable, application-wide label standard: `0 = genuine (bonafide)`, `1 = fake (spoof)`.
2. A unified 9-column dataset manifest schema.
3. Automated verification against cross-split data leakage (speaker overlap, derivative contamination, exact duplication).
4. Strict validation threshold selection (decision thresholds are selected on development data and applied blindly to evaluation data without label peeking).
5. Explicit distinction between **Native Benchmark Inference** and **Sliding-Window Recording Inference**.
6. Comprehensive metric reporting: Equal Error Rate (EER), ROC AUC, PR AUC, confusion matrices, genuine false positive rate (FPR), and per-attack breakdowns (`A01`–`A19`).

---

## 2. Dataset Manifest Specification

### Canonical Schema

All dataset manifests are stored in `data/manifests/` as CSV files with the following 9 columns:

| Column | Type | Description | Example |
|---|---|---|---|
| `sample_id` | String | Unique sample identifier | `LA_D_1047731` / `jay_shetty_vocals` |
| `path` | String | Relative or absolute path to audio file or feature tensor | `data/audio/sample.wav` / `features.npy` |
| `label` | Integer | Binary ground truth (`0 = genuine`, `1 = fake`) | `0` or `1` |
| `dataset` | String | Dataset name | `asvspoof2019_la` / `fakeavceleb` / `local_eval` |
| `split` | String | Data split partition | `train`, `dev`, `eval` |
| `speaker_id` | String | Anonymized or public speaker identity | `LA_0079` / `jay_shetty` |
| `source_id` | String | Origin recording identity for derivative tracking | `source_clip_001` |
| `attack_type` | String | Attack identifier (`-` for bonafide, `A01`–`A19`, or custom) | `A07` / `voice_clone` |
| `language` | String | ISO 639-1 language code | `en`, `hi`, `mr`, `uk` |

### Label Convention

```
0 = Genuine (Bonafide / Real speech)
1 = Fake (Spoof / Deepfake / Cloned speech)
```

Upstream datasets with differing conventions (e.g. ASVspoof `bonafide`/`spoof`, or inverted index conventions) are mapped explicitly at manifest creation time.

---

## 3. Data Leakage Controls

To ensure that reported metrics reflect true out-of-distribution generalization rather than memorization, `scripts/check_data_leakage.py` audits:

1. **Speaker Disjointness**:
   - Zero speaker overlap between `train`, `dev`, and `eval` splits.
   - Verified on ASVspoof 2019 partitions (Train: 20 speakers, Dev: 20 speakers, Eval: 67 speakers — completely disjoint).
2. **Derivative and Source Recording Isolation**:
   - If multiple audio slices, augmented versions, or vocal extractions originate from a shared `source_id`, all derivatives are constrained to the same split.
3. **Exact Duplication Screening**:
   - Check for duplicate sample IDs or identical audio waveform byte hashes across split boundaries.
4. **Pretrained Checkpoint Training Provenance**:
   - Documented training sets for all candidate backbones:
     - `facebook/wav2vec2-base`: LibriSpeech (clean English speech).
     - `wav2vec_best.pt`: ASVspoof 2019 LA Train.
     - `AASIST.pth` / `AASIST-L.pth`: ASVspoof 2019 LA Train.
     - `audio_model.pt`: FakeAVCeleb Train.
   - Evaluation splits (`dev`, `eval`) are held-out relative to training sets.
5. **Threshold Selection Hygiene**:
   - Operating decision thresholds are strictly calibrated on the validation (`dev`) split. Test/eval labels are never used for threshold selection, calibration, or model tuning.

---

## 4. Evaluation Protocols

The evaluation pipeline distinguishes between two operational protocols:

### Protocol A: Native Benchmark Inference (`native_benchmark`)
- Evaluates models under their upstream evaluation setting:
  - For `legacy_wav2vec_mlp`: evaluates on 512-dimensional CNN acoustic embeddings (`mean(dim=1)`) corresponding to official ASVspoof 2019 clips.
  - For `aasist`: evaluates on raw 64,600-sample waveforms (4.0375 seconds) with repetition/truncation padding.
- Measures raw algorithmic discrimination without windowing artifacts.

### Protocol B: Sliding-Window Recording Inference (`sliding_window`)
- Evaluates models under the production application pipeline:
  - Multi-format audio decoding (resampling to 16 kHz mono).
  - Sliding-window slicing (2.0s windows, 1.0s hop).
  - Voice Activity Detection (RMS energy gating, `min_rms_vad = 0.015`).
  - Window score aggregation (mean probability, top-k suspicious timestamps).
- Measures end-to-end performance on full-length recording files.

---

## 5. Baseline Evaluation Results (`legacy_wav2vec_mlp`)

Evaluated using `scripts/evaluate_audio.py` on the official ASVspoof 2019 LA partitions:

### 1. ASVspoof 2019 LA — Development Split (24,844 samples)

| Metric | Value | Notes |
|---|---|---|
| **Total Samples** | 24,844 | 2,548 bonafide, 22,296 spoof |
| **Equal Error Rate (EER)** | **3.47%** | Matches checkpoint metadata (`0.0347155`) |
| **Optimal EER Threshold** | **0.9850** | Used as calibrated decision threshold |
| **ROC AUC** | **0.9947** | High discrimination on known attacks |
| **PR AUC** | **0.9994** | Area under precision-recall curve |
| **Accuracy** | 96.51% | At threshold 0.9850 |
| **Precision** | 99.59% | Low false alarms on known distribution |
| **Recall (Spoof Detection)** | 96.51% | 21,518 / 22,296 detected |
| **Genuine False Positive Rate** | **3.45%** | 88 false alarms out of 2,548 genuine files |
| **Throughput** | 165,087 samples/sec | CPU batch evaluation (batch size = 256) |

### 2. ASVspoof 2019 LA — Evaluation Split (71,237 samples, Unseen Attacks A07–A19)

Threshold calibrated on dev split (`0.9850`) and applied blindly to evaluation split:

| Metric | Value | Notes |
|---|---|---|
| **Total Samples** | 71,237 | 7,355 bonafide, 63,882 spoof |
| **Equal Error Rate (EER)** | **7.13%** | Unseen attack generalization |
| **Operating Threshold** | **0.9850** | Selected on Dev set (no test label peeking) |
| **ROC AUC** | **0.9815** | Area under ROC curve |
| **PR AUC** | **0.9978** | Area under PR curve |
| **Accuracy** | 84.90% | At dev-calibrated threshold |
| **Precision** | 99.80% | Extremely high precision |
| **Recall (Spoof Detection)** | 83.33% | 53,235 / 63,882 detected |
| **Genuine False Positive Rate** | **1.47%** | Only 108 false alarms out of 7,355 genuine files |
| **Throughput** | 130,019 samples/sec | CPU batch evaluation |

### 3. Per-Attack Breakdown on Evaluation Split (Unseen Attacks)

| Attack Type | Method Description | Total Samples | Detected | Recall | Attack-Specific EER |
|---|---|---|---|---|---|
| `bonafide (-)` | Real human speech | 7,355 | 7,247 (TN) | — | FPR = 1.47% |
| `A07` | Vocoder (waveform concatenation) | 4,914 | 4,788 | **97.4%** | 1.89% |
| `A08` | Neural vocoder | 4,914 | 4,843 | **98.6%** | 1.44% |
| `A09` | Vocoder | 4,914 | 4,907 | **99.9%** | 0.70% |
| `A10` | Neural network TTS | 4,914 | 4,812 | **97.9%** | 1.79% |
| `A11` | Neural network TTS | 4,914 | 4,759 | **96.8%** | 2.07% |
| `A12` | Neural network TTS | 4,914 | 4,868 | **99.1%** | 1.31% |
| `A13` | VC (Voice Conversion) | 4,914 | 4,911 | **99.9%** | 0.37% |
| `A14` | VC (Voice Conversion) | 4,914 | 4,891 | **99.5%** | 0.93% |
| `A15` | VC (Voice Conversion) | 4,914 | 4,322 | **88.0%** | 4.55% |
| `A16` | Waveform concatenation | 4,914 | 4,334 | **88.2%** | 5.29% |
| `A17` | Direct waveform concatenation | 4,914 | 1,331 | **27.1%** | 18.93% |
| `A18` | Voice conversion | 4,914 | 1,873 | **38.1%** | 13.42% |
| `A19` | Voice conversion | 4,914 | 2,596 | **52.8%** | 12.67% |

> [!IMPORTANT]
> **Diagnostic Finding**: While `legacy_wav2vec_mlp` achieves high detection (>97%) on neural vocoders and typical TTS algorithms (`A07`–`A14`), its recall drops severely on waveform concatenation and specific voice conversion attacks (`A17`: 27.1%, `A18`: 38.1%, `A19`: 52.8%). This clearly documents the baseline vulnerabilities that the Phase 3 (AASIST) and Phase 4 (SSL-AASIST with RawBoost) candidates must address.

---

## 6. Pipeline Artifacts & Acceptance Gate

Phase 2 deliverables are committed and verifiable via command-line execution:

- **Manifest Generator**: `scripts/build_manifest.py`
  - Generates `data/manifests/manifest_asvspoof2019_{train,dev,eval}.csv`
  - Generates `data/manifests/manifest_local_eval.csv`
  - Generates `data/manifests/manifest_all.csv` (121,467 total records)
  - Generates `reports/dataset_summary.json`
- **Leakage Checker**: `scripts/check_data_leakage.py`
  - Verifies zero cross-split leakage and generates `reports/leakage_audit.json`
- **Evaluation Engine**: `scripts/evaluate_audio.py`
  - Evaluates models under both `native_benchmark` and `sliding_window`
  - Generates `reports/per_recording_predictions.csv`
  - Generates `reports/per_attack_metrics.csv`
  - Generates `reports/legacy_eval_report.json` and Markdown summaries
- **Automated Test Suite**: `tests/test_evaluation_pipeline.py` (13 tests, all passing)

### Reproduction Command

To reproduce the legacy baseline evaluation with one command:

```powershell
py -3.13 scripts/evaluate_audio.py --model legacy_wav2vec_mlp --manifest data/manifests/manifest_asvspoof2019_eval.csv --val_manifest data/manifests/manifest_asvspoof2019_dev.csv --protocol native_benchmark
```
