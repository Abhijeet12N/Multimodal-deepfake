# Model Provenance & Checkpoint Audit

This document records the exact provenance, upstream sources, commit hashes, checksums,
and training datasets for all audio deepfake detectors supported in this repository.

---

## 1. Inventory Summary

| Model Identifier | Architecture | Checkpoint Size | SHA-256 Checksum | License | Upstream Origin |
|---|---|---|---|---|---|
| `legacy_wav2vec_mlp` | Wav2Vec 2.0 CNN + 2-layer MLP | 3.2 MB | `efba2b1fadcb602b928b49496d6bd0ce9786fc10f8d28f76a6de77dfd378159f` | MIT / Internal | `initial setup/deepfake_phase1` |
| `aasist` | RawNet2 Sinc-Conv + Heterogeneous Graph Attention | 1.28 MB | `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0` | Apache-2.0 | [clovaai/aasist](https://github.com/clovaai/aasist) |
| `aasist_l` | Lightweight AASIST (Reduced GAT filters) | 426 KB | `814331d04403848b5327299a917036a5faeb23c22ea36f7eb54f73315a6b73ba` | Apache-2.0 | [clovaai/aasist](https://github.com/clovaai/aasist) |
| `ssl_aasist` | Wav2Vec 2.0 XLSR-53 + AASIST backend | ~1.2 GB (Frontend + Backend) | Documented upstream | MIT | [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) |
| `xlsr_mambo` | Wav2Vec 2.0 XLSR-53 + Mamba State-Space Model | Pending upstream download | Pending upstream | Apache-2.0 | [saki-ciallo/XLSR-MamBo](https://github.com/saki-ciallo/XLSR-MamBo) |

---

## 2. Checkpoint Details

### 2.1 Legacy Wav2Vec 2.0 + MLP (`legacy_wav2vec_mlp`)
- **File Location:** `initial setup/deepfake_phase1/checkpoints/wav2vec_best.pt`
- **Frontend:** Frozen Facebook `wav2vec2-base-960h` feature extractor (7-layer CNN, kernel sizes [10, 3, 3, 3, 3, 2, 2], strides [5, 2, 2, 2, 2, 2, 2]).
- **Embedding:** 512-dimensional acoustic frame features mean-pooled over time.
- **Backend:** 2-layer MLP: Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 2).
- **Training Set:** FakeAVCeleb audio track splits (subject-disjoint split, 80/10/10).
- **Operating Semantics:**
  - `score_orientation`: `higher_is_fake`
  - Score index 0 = Bonafide, index 1 = Spoof.
  - Softmax probability of class 1.
  - Calibrated threshold: `0.984982` (selected at EER).

### 2.2 Pretrained AASIST (`aasist`)
- **File Location:** `aasist/models/weights/AASIST.pth`
- **Upstream Repository:** `https://github.com/clovaai/aasist`
- **Upstream Commit:** `a735c03...` (Official PyTorch release)
- **Architecture:**
  - SincNet convolutional frontend directly operating on raw 16 kHz waveforms.
  - 4 residual blocks with MaxFeatureMap (MFM) activation.
  - Heterogeneous Graph Attention Network (GAT) modeling spectral and temporal graph representations simultaneously.
- **Training Set:** ASVspoof 2019 Logical Access (LA) training partition.
- **Operating Semantics:**
  - `score_orientation`: `higher_is_bonafide`
  - Output: 2 logits `[spoof_logit, bonafide_logit]`.
  - Target score: `bonafide_logit` (evaluated directly in official ASVspoof benchmark).
  - Decision threshold: `0.0` (scores > 0.0 denote authentic speech).

### 2.3 Pretrained AASIST-L (`aasist_l`)
- **File Location:** `aasist/models/weights/AASIST-L.pth`
- **Upstream Repository:** `https://github.com/clovaai/aasist`
- **Architecture:** Lightweight AASIST variant with compressed graph projection channels (85k parameters vs 297k in AASIST).
- **Operating Semantics:** Identical to AASIST (`higher_is_bonafide`).

### 2.4 SSL-AASIST (`ssl_aasist`)
- **Upstream Repository:** `https://github.com/TakHemlata/SSL_Anti-spoofing`
- **Paper:** *SSL-Anti-spoofing: Self-Supervised Learning for Voice Anti-Spoofing* (Interspeech 2022).
- **Frontend:** Pretrained `xlsr_53_56k.pt` (Wav2Vec 2.0 multi-lingual model with 53 languages).
- **Backend:** AASIST graph neural network with RawBoost data augmentation.
- **Training Set:** ASVspoof 2019 LA train set with RawBoost perturbation (reverberation, linear/non-linear filtering).
- **Deployment Constraint:** Requires GPU and `fairseq`. Handled via Google Colab GPU notebook.

### 2.5 XLSR-MamBo (`xlsr_mambo`)
- **Upstream Repository:** `https://github.com/saki-ciallo/XLSR-MamBo`
- **Paper:** *XLSR-MamBo: Bidirectional State Space Models for Audio Deepfake Detection* (2024).
- **Status:** Optional challenger.
- **Hardware Prerequisite:** Linux / Colab with CUDA 12.8+, `flash_attn`, `mamba_ssm`.
