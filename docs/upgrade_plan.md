# Standalone Audio Deepfake Detection Upgrade Plan

## 1. Project Objective

Upgrade an existing multimodal deepfake-detection application with a stronger, reproducible standalone audio-detection pipeline.

**Core Rules:**
- Do not rewrite the entire application. Inspect the actual repository first, preserve working functionality, and implement changes incrementally.
- Success means demonstrably better audio detection under a documented evaluation protocol—not merely integrating a newer model.
- Preserve existing video and SyncStream functionality without breaking multimodal dependencies.

---

## 2. Target Model Strategy

Implement model adapters in the following prioritized order:

| Priority | Model Identifier | Purpose | Upstream Repository |
|---|---|---|---|
| **Required** | `legacy_wav2vec_mlp` | Preserve current baseline | Existing application |
| **Required** | `aasist` | Independent non-Wav2Vec anti-spoofing baseline | [clovaai/aasist](https://github.com/clovaai/aasist) |
| **Required** (Env Feasible) | `ssl_aasist` | Main upgrade candidate: XLSR frontend + RawBoost | [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) |
| **Optional** | `xlsr_mambo` | Newer research challenger (Mamba/FlashAttention) | [saki-ciallo/XLSR-MamBo](https://github.com/saki-ciallo/XLSR-MamBo) |

### Key Architectural Guidelines
1. Do not replace the production model before formal evaluation.
2. Do not implement all architectures at once; follow staged integration.
3. Distinguish original AASIST from SSL-AASIST clearly.
4. Record upstream commit hashes, download URLs, and checkpoint SHA256 checksums.

---

## 3. Implementation Phases

### Phase 0: Repository & Hardware Audit
- Identify entry points, preprocessing code, model classes, windowing, and existing tests.
- Audit runtime resources: Python version, PyTorch/CUDA, RAM, VRAM, and disk space.
- Reproduce one reference prediction and record input hash, checkpoint hash, window scores, and device info.
- Deliverables: `docs/current_system_audit.md`, `docs/environment_audit.md`, `configs/legacy_audio.yaml`, `tests/fixtures/legacy_prediction_reference.json`.

### Phase 1: Introduce AudioDetector Interface
- Abstract base class with `load()`, `requirements()`, `predict_batch()`, and `metadata()`.
- Registry pattern with dynamic discovery.
- Keep decoding, windowing, model inference, calibration, and aggregation separate.
- Store score orientation explicitly (`higher_is_fake` vs `higher_is_bonafide`).
- Acceptance Gate: Wrapping legacy model reproduces reference prediction within numerical tolerance.

### Phase 2: Shared Preprocessing
- Unified `load_audio` pipeline with fallback chain (`soundfile` -> `imageio_ffmpeg` -> `librosa`).
- Standardize on 16 kHz mono float32 normalized representations.
- Support container formats (MP4, MKV) and uncompressed formats (WAV, FLAC).

### Phase 3: Adaptive Windowing
- Configurable window length and hop per detector.
- Explicit final-window coverage with zero or periodic repeat padding.
- Voice Activity Detection (VAD) via RMS energy gating to avoid silence bias.

### Phase 4: Recording-Level Aggregation
- Provide configurable aggregation strategies: `mean`, `top_k`, `percentile`, `duration_weighted`.
- Merge overlapping or contiguous suspicious windows into continuous interval timestamps with peak and mean scores.

### Phase 5: Reproducible Evaluation Pipeline
- Standard manifest schema with 9 columns (`sample_id,path,label,dataset,split,speaker_id,source_id,attack_type,language`).
- Leakage controls: audit speaker overlap, derivative files, and duplicate samples.
- Compute EER, ROC AUC, PR AUC, confusion matrix, and per-attack breakdown.

### Phase 6: Upstream Parity & Pretrained Inference
- Pin upstream revisions, obtain verified checkpoint weights.
- Ensure numerical parity with upstream repositories on identical test files (< 1e-6 difference).

### Phase 7: Score Calibration & Uncertainty Quantification
- Separate calibration module: Platt scaling, isotonic regression, temperature scaling.
- Fit calibration strictly using development-set data.
- Quantify calibration error via Expected Calibration Error (ECE) and Brier score.
- Support an explicit uncertain region (`verdict: "uncertain"`).

### Phase 8: Safe Application Integration
- Single-point service `StandaloneAudioService` for sliding-window audio detection.
- Maintain backward-compatible API schemas and response structures.
- Display raw scores, calibrated fake probabilities, and suspicious timestamp intervals.

### Phase 9: Optional Challenger (XLSR-MamBo)
- Validate CUDA 12.8+ and FlashAttention/Mamba hardware dependencies before deployment.
- Maintain stub adapter with explicit runtime feasibility gating.

---

## 4. Promotion & Rollback Criteria

Promote a model to production status only if:
1. It achieves lower EER and higher AUC than the legacy baseline on primary evaluation.
2. Genuine-audio false-positive rate remains strictly within agreed limits (< 5%).
3. Improvements are verified against data leakage.
4. Latency and memory fit within target compute budget.
5. Automated rollback remains available via config backend selection.
