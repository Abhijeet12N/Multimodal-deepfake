"""XLSR-MamBo (XLSR Frontend + Mamba State-Space Model) Detector Adapter.

Upstream repository: https://github.com/saki-ciallo/XLSR-MamBo
Reference: "XLSR-MamBo: Bidirectional State Space Models for Audio Deepfake Detection", 2024.

Architecture:
- Frontend: Wav2Vec 2.0 / XLSR-53 self-supervised representations
- Backend: Bidirectional Mamba State-Space Models (SSM)
- Upstream requirements: CUDA 12.8, Python 3.12, PyTorch 2.9+, FlashAttention, Mamba-ssm, causal-conv1d, Hydra
- Status: Optional challenger. Blocked on Windows CPU; requires dedicated CUDA environment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..interfaces import AudioDetector


class XLSRMamBoDetector(AudioDetector):
    """Audio detector adapter for XLSR-MamBo.

    This is an optional research challenger that uses Mamba state-space models
    and FlashAttention. It requires specialized CUDA hardware and kernel modules.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.checkpoint_path: str = ""
        self.checkpoint_sha256: str = ""
        self.threshold: float = 0.0

        if checkpoint_path is not None:
            self.load(str(checkpoint_path), device=device)

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"XLSR-MamBo checkpoint not found: {path}")

        self.checkpoint_path = str(path)
        with open(path, "rb") as f:
            self.checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Check required dependencies
        try:
            import mamba_ssm  # type: ignore
        except ImportError:
            raise NotImplementedError(
                "XLSR-MamBo requires 'mamba_ssm', 'causal_conv1d', and CUDA GPU support. "
                "As documented in the upgrade plan, XLSR-MamBo is an optional challenger "
                "requiring CUDA 12.8 and cannot run in a CPU-only environment."
            )

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model = ckpt.get("model", ckpt)
        if isinstance(self.model, nn.Module):
            self.model.eval()

    def requirements(self) -> dict[str, Any]:
        return {
            "sample_rate": 16000,
            "input_length_samples": 64600,
            "input_length_seconds": 64600 / 16000,
            "hop_samples": 32000,
            "hop_seconds": 2.0,
            "padding_mode": "repeat",
            "raw_score_semantics": "two_class_logits",
            "score_orientation": "higher_is_bonafide",
            "class_index_mapping": {0: "spoof", 1: "bonafide"},
            "calibrated_threshold": self.threshold,
            "environment_constraints": "Requires CUDA 12.8, FlashAttention, Mamba-ssm, causal-conv1d",
            "status": "optional_challenger",
        }

    @torch.no_grad()
    def predict_batch(
        self,
        waveforms: torch.Tensor | np.ndarray,
        lengths: list[int] | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(
                "XLSR-MamBo model is not loaded. Dependencies (mamba-ssm) not available."
            )

        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms)

        waveforms = waveforms.float().to(self.device)
        raw_logits = self.model(waveforms)
        bonafide_logits = raw_logits[:, 1].cpu().numpy()
        spoof_logits = raw_logits[:, 0].cpu().numpy()
        fake_scores = spoof_logits - bonafide_logits

        target_len = self.requirements()["input_length_samples"]
        if lengths is None:
            resolved_lengths = [target_len] * waveforms.size(0)
        else:
            resolved_lengths = list(lengths)

        return {
            "raw_scores": raw_logits.cpu().numpy(),
            "bonafide_scores": bonafide_logits,
            "fake_scores": fake_scores,
            "score_orientation": "higher_is_bonafide",
            "lengths": resolved_lengths,
            "model_metadata": self.metadata(),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": "xlsr_mambo",
            "display_name": "XLSR-MamBo (State-Space Model Challenger)",
            "upstream_repo": "https://github.com/saki-ciallo/XLSR-MamBo",
            "upstream_commit": "main",
            "license": "Apache-2.0",
            "uses_ssl_frontend": True,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "score_orientation": "higher_is_bonafide",
            "status": "optional_unsupported_on_cpu",
        }
