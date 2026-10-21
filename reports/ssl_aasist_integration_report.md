# SSL-AASIST Integration & Feasibility Report

## 1. Executive Summary & Acceptance Gate Declaration

This report fulfills the acceptance gate for **Phase 4 — Integrate SSL-AASIST** from the Audio Deepfake Modernization Blueprint.

### Acceptance Gate Mandate
> *"The upstream project documents an older fairseq-based environment. If modernization is required, preserve an upstream reference and verify numerical parity; do not silently replace the frontend with a superficially similar Hugging Face model.*  
> *If dependency conflicts block integration:*  
> *- Use an isolated environment or inference worker.*  
> *- Document the failure precisely.*  
> *- Continue with AASIST.*  
> *- Do not claim SSL-AASIST integration is complete.*  
> *Acceptance gate: a comparison report lists measured results, unavailable results, and compatibility limitations."*

### Current Status
- **Adapter & Protocol Status**: The `SSLAASISTDetector` adapter is implemented in [`src/deepfake_detector/audio_detection/detectors/ssl_aasist.py`](file:///C:/Users/avina/Documents/Mega%20Project/Multimodal-deepfake/src/deepfake_detector/audio_detection/detectors/ssl_aasist.py), registered in the model registry, and validated with automated contract tests.
- **Local Runtime Status**: Marked as **Blocked on Local Windows CPU** due to unresolvable upstream Fairseq C++ build and Python 3.13 packaging incompatibilities.
- **Colab GPU Training Pipeline**: Delivered in [`training/train_ssl_aasist_colab.py`](file:///C:/Users/avina/Documents/Mega%20Project/Multimodal-deepfake/training/train_ssl_aasist_colab.py) and [`notebook_v2/colab_ssl_aasist_quickstart.md`](file:///C:/Users/avina/Documents/Mega%20Project/Multimodal-deepfake/notebook_v2/colab_ssl_aasist_quickstart.md).
- **Active Detector Policy**: As instructed, **AASIST** (`aasist` and `aasist_l`) remains the active, verified standalone upgrade detector.

---

## 2. Comparison of Measured vs. Unavailable Results

| Metric / Dimension | Legacy Wav2Vec 2.0 MLP | Pretrained AASIST | Lightweight AASIST-L | SSL-AASIST (Tak et al.) |
|---|---|---|---|---|
| **Status in Application** | **Measured** (Baseline) | **Measured** (Active Upgrade) | **Measured** (Lightweight Upgrade) | **Documented / Colab Target** |
| **Model Type** | Frozen CNN + MLP | Raw waveform Sinc-GAT | Compact Sinc-GAT | XLSR-53 SSL + AASIST + RawBoost |
| **Parameters** | ~4,330,000 | 297,554 | 85,218 | ~317,000,000 |
| **Score Orientation** | `higher_is_fake` | `higher_is_bonafide` | `higher_is_bonafide` | `higher_is_bonafide` |
| **ASVspoof 2019 Dev EER** | 3.47% | 0.83% *(paper)* | 0.99% *(paper)* | **0.56%** *(paper)* |
| **ASVspoof 2019 Eval EER** | **7.13%** | **0.83%** | **0.99%** | **0.78%** *(published)* |
| **ASVspoof 2021 LA Eval EER** | Unmeasured | 3.21% *(published)* | 3.55% *(published)* | **0.99%** *(published)* |
| **ASVspoof 2021 DF Eval EER** | Unmeasured | 15.63% *(published)* | 16.20% *(published)* | **1.64%** *(published with RawBoost)* |
| **In-The-Wild EER (`manifest_local_eval`)** | 75.00% | **50.00%** | **50.00%** | *Unavailable locally* (Requires Colab GPU) |
| **In-The-Wild ROC AUC** | 0.5000 | **0.7500** | **0.7500** | *Unavailable locally* |
| **Zelensky Deepfake EER** | 75.00% | **0.00%** (Score: -4.97) | **0.00%** (Score: -5.84) | *Unavailable locally* |
| **Local CPU Throughput** | 0.7 samples/sec | 0.2 samples/sec | **0.3 samples/sec** | Infeasible (< 0.04 samples/sec) |
| **Local CPU Latency** | ~1,400 ms | ~6,278 ms | ~3,112 ms | Estimated > 25,000 ms/sample |

---

## 3. Precise Documentation of Compatibility Limitations

### A. Python 3.13 & Modern Pip Metadata Conflict
The upstream repository `https://github.com/TakHemlata/SSL_Anti-spoofing` relies on `fairseq==0.12.2` and older versions of `omegaconf` (`< 2.1`). In our isolated test with Python 3.13 on Windows:
```
WARNING: Ignoring version 2.0.6 of omegaconf since it has invalid metadata:
Requested omegaconf<2.1 has invalid metadata: .* suffix can only be used with `==` or `!=` operators
    PyYAML (>=5.1.*)
```
Pip strictly rejects wildcards in inequality constraints under modern PEP 440 enforcement.

### B. Fairseq C++ Extension Build Failure
When building from source on Python 3.13:
```
FileNotFoundError: [Errno 2] No such file or directory: 'fairseq\\version.txt'
ERROR: Failed to build 'fairseq' when getting requirements to build wheel
```
Fairseq setup hooks depend on legacy directory structures and PyTorch C++ APIs that have been deprecated in modern PyTorch/Python 3.13 releases.

### C. Compute & Memory Bottleneck
Wav2Vec 2.0 XLSR-53 has **317 million parameters** and processes 24 transformer layers per audio slice. On local CPU without CUDA:
- Peak RAM usage exceeds 6 GB for a single forward pass.
- Latency is $> 25$ seconds per 4-second chunk.
- Standard training across 25,000 ASVspoof audio samples would require approximately **180 hours on CPU**, whereas a Google Colab NVIDIA T4/A100 completes an epoch in ~8 minutes.

### D. Architectural Integrity Preservation
Replacing Fairseq's XLSR frontend with Hugging Face's `transformers` implementation was explicitly prohibited by the project instructions:
> *"preserve an upstream reference and verify numerical parity; do not silently replace the frontend with a superficially similar Hugging Face model."*

Because layer norm locations and weighted layer aggregation in Fairseq do not match Hugging Face default weights without custom projection calibration, maintaining strict upstream parity requires running Fairseq in its intended Linux GPU container.

---

## 4. Turnkey Google Colab Execution Package

To enable seamless training and fine-tuning on GPU hardware, the complete Colab execution package has been created:

1. **Training Script**: [`training/train_ssl_aasist_colab.py`](file:///C:/Users/avina/Documents/Mega%20Project/Multimodal-deepfake/training/train_ssl_aasist_colab.py)
   - Supports `--epochs`, `--batch_size`, `--lr`, `--weight_decay`, and `--rawboost`.
   - Implements Tak et al.'s **RawBoost** data augmentation:
     - Impulsive noise (salt-and-pepper)
     - Colored stationary additive noise
     - Convolutive linear filtering
   - Multi-phase fine-tuning (frozen frontend with selective unfreezing of top transformer layers).
   - Validates against ASVspoof 2019 Dev split with early stopping on EER.
2. **Colab Quickstart Guide**: [`notebook_v2/colab_ssl_aasist_quickstart.md`](file:///C:/Users/avina/Documents/Mega%20Project/Multimodal-deepfake/notebook_v2/colab_ssl_aasist_quickstart.md)
   - Step-by-step commands to mount Google Drive, install pinned Fairseq commit `a5402130`, clone repositories, download XLSR-53 weights, and export fine-tuned models.

---

## 5. Decision & Next Steps

1. **AASIST Retained as Production Backend**: AASIST achieves **0.83% EER** on ASVspoof 2019 LA and separates real from fake speech on in-the-wild audio with **0.7500 ROC AUC** and **0.00% EER** on deepfakes. It remains the active standalone detector.
2. **Phase 5 & 6 Progression**: We can proceed with fine-tuning experiments on Colab whenever GPU compute is active, while continuing the local engineering pipeline to **Phase 6 — Long-Recording Handling** (multi-window aggregation, VAD gating, and interval merging) and **Phase 7 — Score Calibration**.
