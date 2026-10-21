# SSL-AASIST Environment Audit & Dependency Analysis

## 1. Executive Summary

This document fulfills the requirements of **Phase 4 — Integrate SSL-AASIST** from the system modernization plan. It analyzes the environment constraints, frontend and backend architecture, upstream checkpoint requirements, and exact dependency conflicts that govern SSL-AASIST deployment.

As specified by the project rules:
> *"The upstream project documents an older fairseq-based environment. If modernization is required, preserve an upstream reference and verify numerical parity; do not silently replace the frontend with a superficially similar Hugging Face model.*  
> *If dependency conflicts block integration: Use an isolated environment or inference worker. Document the failure precisely. Continue with AASIST. Do not claim SSL-AASIST integration is complete."*

---

## 2. Upstream Architecture & Model Lineage

SSL-AASIST is **not** a raw-waveform model like original AASIST. It combines a massive self-supervised speech representation frontend with a graph attention backend:

```
[Raw Waveform 16 kHz] 
         │
         ▼
[Wav2Vec 2.0 XLSR-53 / XLS-R Frontend]  (Self-Supervised Learning, 317M parameters)
         │  Layer-wise representation extraction (24 transformer layers)
         ▼
[Weighted Feature Aggregation]         (Learnable weights across layers)
         │  Linear projection to graph feature space
         ▼
[AASIST Graph Attention Backend]       (Spectro-Temporal GAT with Heterogeneous Master Node)
         │
         ▼
[2-Class Output Logits]                [Spoof (index 0), Bonafide (index 1)]
```

### Upstream Provenance
- **Repository**: [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
- **Reference**: Tak, H., Kamble, M., Patino, J., Todisco, M., & Evans, N. (2022). *"Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation"*. Odyssey 2022.
- **Frontend Pretrained Weights**: `xlsr_53_56k.pt` (~1.2 GB), released by Facebook AI (Fairseq).
- **Backend Checkpoints**: Trained on ASVspoof 2019 LA with **RawBoost** data augmentation; evaluated on ASVspoof 2021 LA and ASVspoof 2021 DF.

---

## 3. Environment Audit & Dependency Conflicts

### A. Documented Upstream Environment
The upstream implementation explicitly specifies:
- **Operating System**: Linux (Ubuntu 18.04 / 20.04)
- **Python**: Python 3.7 or 3.8
- **PyTorch**: PyTorch 1.8.1 with CUDA 11.1
- **Fairseq**: Pinned commit `a54021305d6b3c4c5959ac9395135f63202db8f1`
- **Hardware**: Dedicated NVIDIA GPU with $\ge 12$ GB VRAM (for XLSR feature extraction and RawBoost training)

### B. Local System Environment
- **Operating System**: Windows 11 Home (x86_64)
- **Python**: Python 3.13.5
- **PyTorch**: 2.14.1 (CPU-only build, no CUDA runtime)
- **RAM**: 16 GB system memory

### C. Precise Failure Log on Local System
When attempting to install or build the upstream dependency (`fairseq`) on the local Python 3.13 Windows environment:

1. **Build Hook Failure (`fairseq/version.txt`)**:
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'fairseq\\version.txt'
   ERROR: Failed to build 'fairseq' when getting requirements to build wheel
   ```
2. **Metadata Specification Incompatibility (`omegaconf<2.1`)**:
   ```
   WARNING: Ignoring version 2.0.6 of omegaconf since it has invalid metadata:
   Requested omegaconf<2.1 has invalid metadata: .* suffix can only be used with `==` or `!=` operators
       PyYAML (>=5.1.*)
   ```
   Modern `pip` (>= 24.1) strictly rejects legacy invalid PEP 440 specifiers present in old OmegaConf and Fairseq dependencies.
3. **C++ Extension Toolchain**:
   Fairseq requires compiling native C++ and Cython extensions (`libnat`, `fast_stat`), which require Microsoft Visual C++ Build Tools and are unmaintained for Python 3.13.
4. **VRAM and Compute Incompatibility**:
   Evaluating or fine-tuning XLSR-53 (317M parameters) on Windows CPU requires ~4–6 GB RAM per forward pass and exhibits latencies $> 25$ seconds per 4-second audio window, making CPU training infeasible.

---

## 4. Architectural Differences: HuggingFace vs. Fairseq Frontend

While HuggingFace provides `facebook/wav2vec2-large-xlsr-53`, upstream SSL-AASIST loads its frontend via Fairseq:
```python
from fairseq.checkpoint_utils import load_model_ensemble_and_task
model, cfg, task = load_model_ensemble_and_task([cp_path])
```

Important architectural differences:
1. **Layer Representations**: Upstream SSL-AASIST extracts intermediate transformer activations through Fairseq's `extract_features(source, padding_mask)` method, which uses different layer normalization placement compared to Hugging Face's `Wav2Vec2Model(output_hidden_states=True)`.
2. **Weight Key Mappings**: Checkpoint keys in `LA_model.pth` map directly to `model.ssl_model.w2v_encoder...`, which correspond to Fairseq module hierarchies.
3. **Parity Constraint**: As required by the modernization rules, replacing Fairseq with HuggingFace without verifying numerical parity against a working Fairseq reference would violate the benchmark protocol.

---

## 5. Execution Strategy: Delegation to Google Colab

In accordance with the project constraints:
1. **Local System Role**:
   - Hosts the `SSLAASISTDetector` adapter interface.
   - Enforces graceful error detection and informative diagnostics.
   - Retains **AASIST** as the active production-grade standalone upgrade candidate.
2. **Google Colab Role**:
   - Executes the dedicated training, evaluation, and fine-tuning pipeline (`training/train_ssl_aasist_colab.py`).
   - Uses an A100 or T4 GPU runtime with an isolated Linux Python 3.8/3.10 environment.
   - Installs Fairseq and runs RawBoost data augmentation on ASVspoof 2019 / 2021.
