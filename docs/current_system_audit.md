# Current System Audit & Discrepancy Analysis

**Audit Date**: October 20, 2026  
**Auditor**: Antigravity AI Assistant  
**Repository**: `Multimodal-deepfake`  
**Branch**: `feature/audio-pipeline-upgrade`  

---

## 1. Executive Summary & Acceptance Gate Findings

The primary acceptance gate for Phase 0 requires the existing application to run and requires the agent to **explain how the actual codebase differs from the technical report**.

### Key Differences Between Report Claims and Actual Code

| Component | Technical Report Claim | Codebase Implementation (Phase 1 / Actual Checkpoints) | Discrepancy Impact |
| :--- | :--- | :--- | :--- |
| **Window Length & Hop** | 3.0-second windows with 1.5-second hop | **2.0-second windows with 1.0-second hop** in `predict_wav2vec.py` / `wav2vec_best.pt`. (3.0s / 1.5s is coded in `training/train_audio.py`, but no trained checkpoint exists for it in `models/`). | The existing production checkpoint expects exactly 32,000 samples (2.0s), not 48,000 samples (3.0s). |
| **Wav2Vec Representation** | Mean-pooled 768-dimensional transformer embeddings | **512-dimensional CNN acoustic embeddings** (`backbone.feature_extractor(x).mean(dim=1)`) in `wav2vec_best.pt`. | Passing 768-dim embeddings into `wav2vec_best.pt` causes a tensor dimension mismatch (expects 512-d). |
| **Classifier Output & Scoring** | Single-score logit with sigmoid → [0, 1] fake probability | **Two-class output** `[real_logit, spoof_logit]` with Softmax. Spoof probability is index 1. | Orientation is `spoof_prob = softmax(logits)[1]`. Calibrated threshold is 0.984982 (EER on ASVspoof 2019 LA dev set). |
| **Window Gating / VAD** | Simple sliding window across full audio | **RMS Energy Voice Activity Gating** (`min_rms = 0.015`). Silence / noise windows are assigned score 0.0 and excluded from mean speech score. | Without VAD, long pauses distort recording-level aggregation. |
| **Local Checkpoint Availability** | Report implies `models/audio_model.pt` and `models/best_model.pt` exist | `models/` directory only contains a `README.md`. Actual weights are located in `initial setup/deepfake_phase1/checkpoints/wav2vec_best.pt` and `aasist/models/weights/`. | The Flask app returns 503 if checkpoints are not explicitly linked or configured via environment variables. |

---

## 2. Application Framework and Entry Points

### 2.1 Web Application
- **Framework**: Flask (`ui/app.py`).
- **Templates & Static Assets**: `ui/templates/index.html`, `ui/static/css/`, `ui/static/js/`.
- **Primary Routes**:
  - `GET /`: Interactive web dashboard.
  - `GET /api/status`: System health and model availability status.
  - `POST /api/analyze`: Multimodal video deepfake detection.
  - `POST /api/analyze-audio`: Standalone audio sliding-window detection.
  - `GET /samples/<filename>`: Serves sample media files.
  - `GET /uploads/<filename>`: Serves user-uploaded media files.

### 2.2 CLI and Script Entry Points
- `scripts/analyze_video.py`: Headless CLI tool to analyze video or audio files.
- `training/train_audio.py`: Script to train standalone audio classifier on FakeAVCeleb metadata.
- `training/train_multimodal.py`: Script for multi-stream multimodal training.
- `initial setup/deepfake_phase1/predict_wav2vec.py`: Phase 1 standalone Wav2Vec CLI tool.

---

## 3. Audio Decoding and Preprocessing

- **Decoding Strategy**:
  1. `soundfile.read` (primary, fast uncompressed / FLAC / WAV reader).
  2. `imageio_ffmpeg` pipe fallback for container formats.
  3. `librosa.load(..., sr=16000, mono=True)` (universal fallback).
- **Audio Specifications**:
  - Sample Rate: 16,000 Hz.
  - Channels: Mono (stereo averaged to 1 channel).
  - Normalization: Floating point `[-1.0, 1.0]`.
- **Preprocessing Implementation**:
  - In `src/deepfake_detector/preprocessing.py`: `load_audio_windows()` creates 3.0s windows with 1.5s hop.
  - In `initial setup/deepfake_phase1/predict_wav2vec.py`: `load_audio()` converts to 16kHz mono and windows into 32,000 samples (2.0s) with 16,000 samples (1.0s) hop.

---

## 4. Existing Model Classes & Checkpoint Loaders

### 4.1 Standalone Audio Models
1. **`DeepfakeClassifier`** (`initial setup/deepfake_phase1/wav2vec_model.py`):
   ```python
   nn.Sequential(
       nn.Linear(input_dim=512, hidden_dim=256),
       nn.BatchNorm1d(256),
       nn.ReLU(),
       nn.Dropout(0.3),
       nn.Linear(256, 2)
   )
   ```
   - Checkpoint: `initial setup/deepfake_phase1/checkpoints/wav2vec_best.pt`
   - SHA256: `efba2b1fadcb602b928b49496d6bd0ce9786fc10f8d28f76a6de77dfd378159f`
   - Input: 512-dim mean-pooled CNN acoustic features from `facebook/wav2vec2-base`.

2. **`AudioClassifier`** (`src/deepfake_detector/model.py`):
   ```python
   nn.Sequential(
       nn.LayerNorm(input_dim=768),
       nn.Linear(768, 256),
       nn.GELU(),
       nn.Dropout(0.25),
       nn.Linear(256, 64),
       nn.GELU(),
       nn.Dropout(0.15),
       nn.Linear(64, 1)
   )
   ```
   - Input: 768-dim mean-pooled transformer hidden state from `facebook/wav2vec2-base-960h`.

3. **`AASIST` & `AASIST-L`** (`aasist/models/AASIST.py`):
   - Checkpoint: `aasist/models/weights/AASIST.pth` (SHA256: `51d2d9cf...`, size: 1,281,532 bytes).
   - Checkpoint: `aasist/models/weights/AASIST-L.pth` (SHA256: `814331d0...`, size: 426,428 bytes).
   - Input: 64,600 raw waveform samples (~4.04s at 16kHz). Sinc-convolution filterbank + Spectro-Temporal Graph Attention.
   - Output: 2 logits: index 0 = spoof, index 1 = bonafide.

---

## 5. API Request and Response Schema

### `POST /api/analyze-audio`
- **Request**: Multipart form data with `file` (audio upload: WAV, MP3, M4A, FLAC, OGG, AAC).
- **Current Response Schema**:
  ```json
  {
    "ok": true,
    "filename": "speech.wav",
    "result": {
      "fake_probability": 0.9999,
      "label": "fake",
      "threshold": 0.5,
      "analyzed_seconds": 10.54,
      "windows_analyzed": 10,
      "top_windows": [
        {
          "start_seconds": 1.0,
          "end_seconds": 3.0,
          "fake_score": 1.0
        }
      ],
      "inference_seconds": 0.53
    }
  }
  ```

---

## 6. Video and SyncStream Dependencies

The multimodal video pipeline (`src/deepfake_detector/model.py:MultiTaskDetector`) fuses three independent feature streams:
1. **`ImageStream`**: Xception architecture (via `timm`) extracting 2048-dim features pooled over 8 frames (mean + std = 4096-dim) projected to 512-dim.
2. **`AudioStream`**: Frozen `facebook/wav2vec2-base-960h` encoder (768-dim projected to 512-dim).
3. **`SyncStream`**:
   - Visual: 3D CNN on 8 face lip crops (`112x112`).
   - Audio: 2D CNN on 64-band log-mel spectrogram (`64x301`).
   - Late fusion trunk (1536-dim) with 3 output heads: `video`, `audio`, and `any`.

---

## 7. Checkpoint Inventory and Verification Hashes

| Path | File Size | SHA256 Hash | Status |
| :--- | :--- | :--- | :--- |
| `initial setup/deepfake_phase1/checkpoints/wav2vec_best.pt` | 666,067 B | `efba2b1fadcb602b928b49496d6bd0ce9786fc10f8d28f76a6de77dfd378159f` | **Verified & Reproducible** |
| `initial setup/deepfake_phase1/checkpoints/wav2vec_final.pt` | 665,953 B | `889eb22ad317d7b0559f2ff0a3311e548232230691e8eaecf9e612ec9a572a12` | Backup final epoch |
| `aasist/models/weights/AASIST.pth` | 1,281,532 B | `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0` | Pretrained AASIST |
| `aasist/models/weights/AASIST-L.pth` | 426,428 B | `814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a` | Pretrained AASIST-L |
| `initial setup/deepfake_phase1/Audios/jay_shetty_vocals.wav` | 337,222 B | `cc0c1d4bff3e67530447a122ce7e3147f3027d98553205626f13255b74e4f460` | Input reference file |
