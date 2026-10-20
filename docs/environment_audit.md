# Environment and Runtime Audit

**Audit Date**: October 20, 2026  
**Auditor**: Antigravity AI Assistant  
**Repository**: `Multimodal-deepfake`  
**Execution Context**: Local Development (Windows 11) + Google Colab (Training Target)

---

## 1. Local Runtime Resources

| Resource | Value | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Home / Pro (Build 10.0.26300) | 64-bit AMD64 architecture |
| **Python Version** | Python 3.13.5 (tags/v3.13.5:6cb20a2) | Standard CPython 64-bit |
| **PyTorch Version** | 2.10.0+cpu | CPU-only build; CUDA extensions not bound |
| **Torchaudio Version**| Not installed (`ModuleNotFoundError`) | Audio I/O handled via `soundfile` 0.14.0 & `librosa` 1.0.0 |
| **Torchvision** | 0.21.0+cpu | Patched `_meta_registrations.py` for PyTorch 2.10 fake-op compatibility |
| **Transformers** | 5.2.0 | Offline mode enabled (`HF_HUB_OFFLINE=1`) |
| **timm** | 1.0.30 | Used by multimodal `ImageStream` (Xception) |
| **GPU Model & VRAM** | None available to PyTorch | `torch.cuda.is_available() == False` |
| **System RAM** | 15.4 GB total / ~1.9 GB available | High memory pressure; large batch extraction must be avoided locally |
| **Storage / Disk** | 393.05 GB total / 127.54 GB free | Adequate for code, checkpoints, and cached feature arrays |

---

## 2. Execution Strategy: Local vs. Google Colab

The user has explicitly confirmed: **"I will train model on colab, current system does not have gpu."**

### Local Development Scope
- Codebase refactoring and interface abstraction (`AudioDetector`).
- Unit and regression testing.
- Checkpoint hashing and schema validation.
- Pretrained CPU inference parity testing on sample recordings.
- Dataset manifest generation, leakage checking, and formatting.
- Flask UI and API endpoints testing.

### Google Colab Training & Benchmark Scope
- Model training (AASIST, SSL-AASIST, Fine-tuning).
- Heavy GPU inference and batch evaluation on ASVspoof 2019 / 2021 datasets.
- CUDA acceleration (mixed precision FP16/BF16, gradient accumulation).
- Upstream Fairseq / RawBoost compilation required by SSL-AASIST.

---

## 3. Target Model Feasibility & Dependency Matrix

| Model Identifier | Upstream Repository | Local CPU Feasibility | Colab GPU Feasibility | Dependency Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **`legacy_wav2vec_mlp`** | Existing application | **Yes** (tested: 0.53s / 10.5s audio) | **Yes** | Uses cached `facebook/wav2vec2-base` + local `wav2vec_best.pt` |
| **`aasist`** | [clovaai/aasist](https://github.com/clovaai/aasist) | **Yes** (pure PyTorch + SincNet) | **Yes** | Checkpoints `AASIST.pth` (1.28 MB) & `AASIST-L.pth` (426 KB) present in workspace |
| **`ssl_aasist`** | [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) | **Inference with caution** | **Yes (Primary target)** | XLSR-53 / Wav2Vec 2.0 XLS-R frontend; upstream documents Fairseq & Python 3.7/3.8 |
| **`xlsr_mambo`** *(Optional)* | [saki-ciallo/XLSR-MamBo](https://github.com/saki-ciallo/XLSR-MamBo) | **Blocked** (no CUDA/FlashAttention) | **Conditional** | Requires CUDA 12.8, PyTorch 2.9+, FlashAttention, Mamba-ssm, and Hydra |

---

## 4. Key Environmental Workarounds & Mitigations

1. **HuggingFace Offline Mode**:
   Local network calls to `huggingface.co` are blocked or timeout in the sandbox. `facebook/wav2vec2-base` is already cached locally at `~/.cache/huggingface/hub/models--facebook--wav2vec2-base`. Setting `local_files_only=True` and `HF_HUB_OFFLINE=1` allows immediate, reliable loading.
2. **PyTorch 2.10 Fake Operator Patch**:
   PyTorch 2.10 strict registration checks caused torchvision's `@torch.library.register_fake("torchvision::nms")` to raise a `RuntimeError` on Windows CPU. A safe wrapper was applied to allow `timm` and `torchvision` to import cleanly without native C++ compilation.
3. **Decoupled Audio Decoding**:
   Since `torchaudio` is absent, all audio decoding is standardized on `soundfile` with a robust fallback to `librosa` and FFmpeg.
