"""Adapter stub for XLSR-MamBo (saki-ciallo/XLSR-MamBo).

This is ``adapter_id = "xlsr_mambo"`` — an **optional** newer research
challenger that is NOT implemented yet.

Architecture:
    Raw 16 kHz waveform
      → XLSR (Wav2Vec 2.0 / XLS-R) frontend (self-supervised)
      → Hybrid Mamba-Attention backbone
      → Binary output

**This model still uses the Wav2Vec / XLSR family** as its SSL frontend.

Upstream:
    Repository : https://github.com/saki-ciallo/XLSR-MamBo
    Paper      : "XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone
                  for Audio Deepfake Detection" (ACL 2026 Findings)
    Pretrained : Available on Google Drive (Table 3 models)

Environment constraints (from upstream README):
    - CUDA 12.8, Python 3.12, PyTorch 2.9.0
    - Requires: Flash Attention, Mamba (state-spaces), Hydra (goombalab)
    - These are heavy CUDA-compiled dependencies — Docker recommended
    - This CANNOT share a Python environment with SSL-AASIST (Python 3.7)

Status: STUB — not implemented.  To be attempted only after AASIST and
SSL-AASIST are evaluated.
"""
from __future__ import annotations

import torch

from .base import AdapterInfo, AudioModelAdapter, _register


@_register
class XLSRMamBoAdapter(AudioModelAdapter):
    """Stub adapter for XLSR-MamBo (not yet implemented)."""

    @classmethod
    def info(cls) -> AdapterInfo:
        return AdapterInfo(
            adapter_id="xlsr_mambo",
            display_name="XLSR-MamBo (Hybrid Mamba-Attention) [NOT IMPLEMENTED]",
            upstream_repo="https://github.com/saki-ciallo/XLSR-MamBo",
            upstream_commit="",
            license="Not explicitly stated",
            sample_rate=16_000,
            uses_ssl_frontend=True,
            checkpoint_url=(
                "https://drive.google.com/drive/folders/"
                "108a-gFEGhQU2_p16MvD2Rk7qeSyHEBpw"
            ),
            checkpoint_sha256="",
            notes=(
                "Optional research challenger.  Uses XLSR (Wav2Vec 2.0 "
                "family) as SSL frontend — NOT a raw-waveform model.  "
                "Requires CUDA 12.8, Python 3.12, PyTorch 2.9, "
                "Flash Attention, Mamba, and Hydra.  Cannot share a "
                "Python environment with SSL-AASIST (Python 3.7) or "
                "the main application (PyTorch 2.1+).  "
                "Docker strongly recommended."
            ),
        )

    def load(self, checkpoint_path: str, device: torch.device) -> None:
        raise NotImplementedError(
            "XLSR-MamBo adapter is not yet implemented.  "
            "Evaluate AASIST and SSL-AASIST first before attempting "
            "this optional architecture."
        )

    @torch.no_grad()
    def score(self, waveform: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("XLSR-MamBo adapter is not yet implemented.")
