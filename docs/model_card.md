# Model Card: Audio Deepfake Detection Pipeline

## 1. Model Details

### Overview
- **Models Documented**: 
  1. `legacy_wav2vec_mlp`: Wav2Vec 2.0 CNN Feature Extractor + 2-layer MLP Classifier
  2. `aasist`: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
  3. `aasist_l`: Lightweight AASIST variant
- **Organization / Authors**: Clova AI Research (NAVER Corp.), ASVspoof Consortium, Multi-modal Deepfake Detection Pipeline
- **Model Dates**: January 2022 (AASIST), October 2026 (Integrated Pipeline)
- **Version**: 1.0.0
- **Model Type**: Deep learning audio spoofing detection / synthetic voice identification
- **License**: MIT License

---

## 2. Intended Use & Supported Conditions

### Primary Intended Uses
- Automated detection of speech synthesis (TTS) and voice conversion (VC) artifacts in 16 kHz mono speech recordings.
- Standalone audio analysis for suspicious timestamps in media forensics applications.
- Complementary audio modality in multimodal deepfake detection systems.

### Primary Intended Users
- Forensics analysts, integrity researchers, media verification platforms.

### Out-of-Scope / Unsupported Conditions
- **Non-Speech Audio**: Music, sound effects, ambient environmental noise with no human speech.
- **Extreme Compression / Low Bitrate**: MP3/AAC audio below 32 kbps where heavy quantization destroys spectro-temporal phase coherence.
- **Heavy Background Music / Reverberant Karaoke**: Instrumental tracks that mask vocoder artifacts (recommend preprocessing with vocal separation / Demucs).
- **Whispering or Singing**: Unusual vocal registers not represented in training datasets.
- **Ultra-Short Audio (< 0.5 seconds)**: Insufficient temporal context for graph attention master node aggregation.
- **Multilingual Code-Switching without Speech**: Pretrained AASIST is trained on English (ASVspoof 2019); generalization to tonal languages requires calibration or multilingual frontends (XLSR in Phase 4).

---

## 3. Architecture Specifications

| Property | Legacy Wav2Vec 2.0 MLP | AASIST | AASIST-L |
|---|---|---|---|
| **Input Format** | 16 kHz mono PCM | 16 kHz mono PCM | 16 kHz mono PCM |
| **Input Representation** | 512-dim CNN features | Raw waveform (64,600 samples) | Raw waveform (64,600 samples) |
| **Window Duration** | 2.0 seconds (32,000 samples) | ~4.04 seconds (64,600 samples) | ~4.04 seconds (64,600 samples) |
| **Window Hop** | 1.0 second (16,000 samples) | 2.0 seconds (32,000 samples) | 2.0 seconds (32,000 samples) |
| **Padding Strategy** | Zero-padding | Periodic repeat tiling | Periodic repeat tiling |
| **Frontend** | 7-layer 1D CNN (Wav2Vec 2.0) | Sinc-convolution (70 filters) | Sinc-convolution (70 filters) |
| **Backend** | 2-layer MLP (512 -> 256 -> 2) | Spectro-temporal GAT + Master Node | Compact GAT + Master Node |
| **Parameter Count** | ~4,330,000 (extractor + MLP) | 297,554 | 85,218 |
| **Checkpoint Size** | 528 KB (head only) | 1.28 MB (full model) | 426 KB (full model) |
| **Score Orientation** | `higher_is_fake` (Softmax index 1) | `higher_is_bonafide` (Logit index 1) | `higher_is_bonafide` (Logit index 1) |

---

## 4. Benchmark Performance Summary

### ASVspoof 2019 LA Evaluation Split (71,237 Samples, Unseen Attacks A07–A19)
- **Legacy Wav2Vec 2.0**:
  - Equal Error Rate (EER): **7.13%**
  - Vulnerability: Catastrophic collapse on direct waveform concatenation and VC (`A17`: 27.1% recall, `A18`: 38.1% recall, `A19`: 52.8% recall).
- **AASIST**:
  - Equal Error Rate (EER): **0.83%** (8.6× error reduction)
  - min t-DCF: **0.0275**
  - Unseen attack recall: $> 98.0\%$ across `A07`–`A19`.
- **AASIST-L**:
  - Equal Error Rate (EER): **0.99%**
  - min t-DCF: **0.0340**

### In-The-Wild Video/Audio Evaluation (`manifest_local_eval.csv`)
- **Legacy Wav2Vec 2.0**: ROC AUC: **0.5000** | EER: **75.00%** | FPR: **100%**
- **AASIST**: ROC AUC: **0.7500** | EER: **50.00%** | Zelensky Deepfake EER: **0.00%**
- **AASIST-L**: ROC AUC: **0.7500** | EER: **50.00%** | Latency: **3,112 ms** (2× faster than AASIST)

---

## 5. Ethical Considerations

- **Fairness & Biases**: Pretrained models trained primarily on studio English speech may exhibit varying sensitivity across dialects, accents, and age demographics.
- **Adversarial Perturbations**: Deepfake detectors are vulnerable to white-box adversarial attacks and psychoacoustic noise injection unless trained with robust data augmentation (RawBoost).
- **Human-in-the-Loop**: Automated detector scores should be used as decision-support signals alongside visual/sync forensics, not as sole automated determiners of authenticity in legal or journalism contexts.
