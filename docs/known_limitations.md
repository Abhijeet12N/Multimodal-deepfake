# Known Limitations & Operating Boundaries

This document details known operational boundaries, acoustic sensitivities, domain shift vulnerabilities,
and architectural constraints for the audio deepfake detection system.

---

## 1. Acoustic & Signal Limitations

### 1.1 Short Audio Duration (< 1.5 Seconds)
- **Constraint:** Detectors require sufficient acoustic context (AASIST: 64,600 samples ~4.04s; Legacy: 32,000 samples ~2.0s).
- **Behavior:** Recordings shorter than the model window are padded (periodic repeat or zero padding).
- **Risk:** Short utterances (e.g., single-word affirmations, coughs) contain insufficient phonemic variation, increasing false-alarm risk.
- **Mitigation:** The system tags short audio with `warnings` and flags intervals where valid audio constitutes < 50% of the window.

### 1.2 Audio Codec Compression & Bitrate Artifacts
- **Constraint:** ASVspoof 2019 LA models were trained on uncompressed 16 kHz FLAC files.
- **Sensitivity:** High-loss codecs (AAC < 64 kbps, MP3 < 96 kbps, Opus/WhatsApp audio) introduce quantization noise that can trigger false spoof detections on genuine voices.
- **Mitigation:** Normalization and multi-tier audio decoding are applied, but lossy compression remains a primary source of domain shift.

### 1.3 Background Noise & Music Interference
- **Sensitivity:** High levels of background music, traffic noise, or ambient reverberation obscure subtle vocoder artifacts.
- **Mitigation:** RMS-based Voice Activity Detection (VAD) gates out silent and noise-only windows, but loud background music overlapping active speech can degrade classification confidence.

---

## 2. Dataset & Linguistic Domain Shift

### 2.1 Laboratory vs In-The-Wild Distribution Gap
- **ASVspoof 2019:** Clean speech recorded in acoustically controlled environments using early neural TTS (Tacotron, WaveNet) and VC algorithms (A01–A19).
- **Modern In-The-Wild Generation:** Contemporary deepfakes utilize diffusion vocoders, flow-matching, zero-shot voice cloning (e.g., ElevenLabs, XTTS, VALL-E), and acoustic room simulation.
- **Performance Impact:** Zero-shot generalization on unseen in-the-wild voice clones exhibits higher EER than on ASVspoof test sets.

### 2.2 Language Bias (English vs Non-English)
- **Training Bias:** Pretrained AASIST models were trained almost exclusively on English speech samples.
- **Impact on Indian Languages:** Tonal and phonetic patterns in Hindi, Marathi, Tamil, or Bengali may exhibit different spectral distributions, potentially altering baseline score distributions.
- **Mitigation Strategy:** SSL-AASIST (using multilingual XLSR-53 with 53 languages) provides significantly stronger cross-lingual representations.

---

## 3. Calibration & Uncertainty Handling

### 3.1 Raw Logits vs Verified Probabilities
- **Precaution:** Upstream AASIST models output raw logits in $(-\infty, +\infty)$. Passing logits through standard softmax does NOT yield a verified frequentist probability.
- **Implementation:** The system preserves raw scores, and only provides `calibrated_fake_probability` when an explicit calibration artifact (Platt scaling or isotonic regression) has been fitted on development data.

### 3.2 The Uncertain Decision Band
- **Policy:** For scores yielding calibrated probabilities between `0.40` and `0.60`, the system outputs `verdict: "uncertain"` rather than forcing an unconfident binary label.
- **Application Workflow:** Flagged "uncertain" items should be routed to secondary review or evaluated in conjunction with the visual/SyncStream video detector.

---

## 4. Hardware & Platform Constraints

| Detector | Local Windows CPU | Google Colab GPU | High-Performance Server (CUDA 12.8+) |
|---|---|---|---|
| `legacy_wav2vec_mlp` | ✅ Supported (~1.2s / 3s audio) | ✅ Supported (<100ms) | ✅ Supported |
| `aasist` | ✅ Supported (~450ms / 4s audio) | ✅ Supported (<50ms) | ✅ Supported |
| `aasist_l` | ✅ Supported (~210ms / 4s audio) | ✅ Supported (<30ms) | ✅ Supported |
| `ssl_aasist` | ❌ Blocked (fairseq / XLSR heavy) | ✅ Supported (T4 / A100 GPU) | ✅ Supported |
| `xlsr_mambo` | ❌ Unsupported | ⚠️ Blocked (requires CUDA 12.8+) | ✅ Supported (Linux + Mamba) |

---

## 5. Temporal Edit Localization Resolution

- The system flags suspicious windows (2.0s or 4.0s) and merges overlapping windows into suspicious intervals.
- **Limitation:** Localization resolution is bounded by window hop size (1.0s to 2.0s). The detector cannot pinpoint exact phoneme-level or millisecond-level splicing boundaries without specialized frame-level segmentation models.
