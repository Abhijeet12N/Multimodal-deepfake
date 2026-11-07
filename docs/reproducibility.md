# Reproducibility Guide & Workflow Commands

This guide provides step-by-step instructions to reproduce all audit, evaluation, calibration,
benchmarking, and test results in the audio deepfake detection system.

---

## 1. Environment Preparation

### Local Environment (Windows 11 / CPU / GPU)
```bash
# Clone or navigate to repository
cd "Multimodal-deepfake"

# Install dependencies
pip install -r requirements.txt
pip install pytest scikit-learn soundfile librosa
```

### Google Colab GPU Environment (for SSL-AASIST / XLSR-MamBo)
1. Open Google Colab and set Runtime -> Change runtime type -> **T4 GPU** or **A100 GPU**.
2. Mount Google Drive and navigate to repository root.
3. Install required backend packages:
```bash
pip install fairseq timm librosa soundfile
```

---

## 2. Verification & Automated Tests

To run the complete test suite:
```bash
python -m pytest tests/ -v
```

To run individual test domains:
```bash
# Legacy and Upstream numerical parity (< 1e-6 tolerance)
python -m pytest tests/test_legacy_parity.py tests/test_upstream_parity.py -v

# Score direction semantics & mappings
python -m pytest tests/test_score_direction.py -v

# Adaptive windowing & padding
python -m pytest tests/test_windowing.py -v

# Score aggregation & interval merging
python -m pytest tests/test_aggregation.py -v

# API response schemas & contract adherence
python -m pytest tests/test_api_contract.py -v

# Calibration & ECE quantification
python -m pytest tests/test_calibration.py -v

# Manifest building, data leakage & metrics
python -m pytest tests/test_evaluation_pipeline.py -v
```

---

## 3. Dataset Manifests & Data Leakage Audit

### 3.1 Build Manifest
To generate a standardized 9-column CSV dataset manifest from audio files:
```bash
python scripts/build_manifest.py \
    --input-dir data/audio_samples \
    --dataset local_eval \
    --output data/manifests/manifest_local_eval.csv
```

### 3.2 Audit Data Leakage
To ensure subject disjointness and eliminate near-duplicates:
```bash
python scripts/check_data_leakage.py \
    --manifests data/manifests/manifest_asvspoof2019_train.csv data/manifests/manifest_asvspoof2019_eval.csv \
    --output reports/leakage_audit.json
```

---

## 4. Model Evaluation & Benchmarks

### 4.1 Run Native Evaluation Protocol
Evaluates any registered detector on a test manifest, calculating EER, AUC, and per-attack metrics:
```bash
# Evaluate AASIST
python scripts/evaluate_audio.py \
    --manifest data/manifests/manifest_local_eval.csv \
    --model aasist \
    --output-dir reports/aasist_eval

# Evaluate Legacy Wav2Vec baseline
python scripts/evaluate_audio.py \
    --manifest data/manifests/manifest_local_eval.csv \
    --model legacy_wav2vec_mlp \
    --output-dir reports/legacy_eval
```

### 4.2 Benchmark Runtime & Memory
Measures model load latency, inference throughput, and memory consumption across 3s, 10s, and 30s recordings:
```bash
python scripts/benchmark_audio.py \
    --models legacy_wav2vec_mlp aasist \
    --device cpu \
    --durations 3.0 10.0 30.0 \
    --output reports/benchmark_audio.json
```

---

## 5. Score Calibration & Uncertainty Fitting

To fit Platt scaling or Isotonic regression on development-set scores:
```bash
python scripts/fit_calibration.py \
    --predictions-csv reports/per_recording_predictions.csv \
    --method platt \
    --model-id aasist \
    --score-orientation higher_is_bonafide \
    --operating-threshold 0.50 \
    --uncertain-lower 0.40 \
    --uncertain-upper 0.60 \
    --output configs/calibration/aasist_platt.json
```

---

## 6. System & Hardware Audit

To programmatically generate an environment and hardware capability audit:
```bash
python scripts/audit_environment.py --output docs/environment_audit.md
```
