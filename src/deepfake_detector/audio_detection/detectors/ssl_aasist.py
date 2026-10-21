"""SSL-AASIST (Self-Supervised Wav2Vec 2.0 / XLSR + AASIST) Detector Adapter.

Upstream repository: https://github.com/TakHemlata/SSL_Anti-spoofing
Reference: Tak et al., "End-to-End Anti-Spoofing with Self-Supervised
Representations and RawBoost Data Augmentation", Odyssey 2022.

Architecture:
- Frontend: Wav2Vec 2.0 XLSR-53 / XLS-R self-supervised acoustic representation (317M params)
- Backend: AASIST spectro-temporal graph attention network with master node
- Training Augmentation: RawBoost (impulsive noise, colored noise, linear filtering)
- 2-class output: [spoof (index 0), bonafide (index 1)]
- Native input length: 64,600 samples (~4.04s at 16 kHz)
- Score orientation: UPSTREAM EVALUATION USES BONAFIDE LOGIT (higher_is_bonafide)
- Primary Execution Environment: Google Colab (CUDA GPU + Fairseq)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..interfaces import AudioDetector


class SSLAASISTDetector(AudioDetector):
    """Audio detector adapter for SSL-AASIST (XLSR frontend + AASIST backend).

    This model serves as the primary fine-tuning upgrade candidate.
    Full upstream execution requires fairseq and a CUDA GPU. In environments
    without fairseq (such as local Windows CPU), environment checks provide
    clear diagnostic messages while preserving system stability.
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

    @staticmethod
    def is_environment_available() -> tuple[bool, str]:
        """Check whether the current Python runtime satisfies SSL-AASIST dependencies."""
        try:
            import fairseq  # type: ignore # noqa: F401
            return True, "fairseq is available."
        except ImportError as e:
            return False, (
                f"Missing dependency: {e}. SSL-AASIST requires fairseq, which requires "
                "Linux and Python <= 3.10 with C++ toolchain. Use the Google Colab training "
                "environment for SSL-AASIST execution."
            )

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SSL-AASIST checkpoint not found: {path}")

        self.checkpoint_path = str(path)
        with open(path, "rb") as f:
            self.checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

        available, reason = self.is_environment_available()
        if not available:
            raise RuntimeError(
                f"Cannot load SSL-AASIST in this environment: {reason}\n"
                "To run SSL-AASIST, launch the Google Colab notebook documented in "
                "`notebook_v2/colab_ssl_aasist_quickstart.md` or `training/train_ssl_aasist_colab.py`."
            )

        # In a Colab / Fairseq environment: load upstream Model
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model = ckpt["model"]
        elif isinstance(ckpt, nn.Module):
            self.model = ckpt
        else:
            raise ValueError(f"Unrecognized SSL-AASIST checkpoint structure in {path}")

        self.model.to(self.device).eval()

    def requirements(self) -> dict[str, Any]:
        return {
            "sample_rate": 16000,
            "input_length_samples": 64600,  # ~4.0375 seconds native input
            "input_length_seconds": 64600 / 16000,
            "hop_samples": 32000,           # ~2.0 second default hop
            "hop_seconds": 2.0,
            "padding_mode": "repeat",        # Upstream repeats waveform periodically
            "raw_score_semantics": "two_class_logits",
            "score_orientation": "higher_is_bonafide",  # Upstream evaluates output[:, 1]
            "class_index_mapping": {0: "spoof", 1: "bonafide"},
            "calibrated_threshold": self.threshold,
            "min_gpu_vram_gb": 12.0,
            "environment": "Google Colab (CUDA GPU + Fairseq)",
        }

    @torch.no_grad()
    def predict_batch(
        self,
        waveforms: torch.Tensor | np.ndarray,
        lengths: list[int] | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            available, reason = self.is_environment_available()
            raise RuntimeError(
                f"SSL-AASIST model is not loaded. {reason}\n"
                "Please run on Google Colab with GPU acceleration."
            )

        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms)

        waveforms = waveforms.float().to(self.device)
        batch_size = waveforms.size(0)
        target_len = self.requirements()["input_length_samples"]

        # Ensure correct input length via repeat padding
        if waveforms.size(1) != target_len:
            from .aasist import AASISTDetector
            waveforms = AASISTDetector.pad_waveform(waveforms, target_len).to(self.device)

        raw_logits = self.model(waveforms)
        if isinstance(raw_logits, tuple):
            raw_logits = raw_logits[0]

        bonafide_logits = raw_logits[:, 1].cpu().numpy()
        spoof_logits = raw_logits[:, 0].cpu().numpy()
        fake_scores = spoof_logits - bonafide_logits

        if lengths is None:
            resolved_lengths = [target_len] * batch_size
        elif isinstance(lengths, torch.Tensor):
            resolved_lengths = lengths.cpu().tolist()
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
            "model_id": "ssl_aasist",
            "display_name": "SSL-AASIST (XLSR-53 + AASIST Backend)",
            "upstream_repo": "https://github.com/TakHemlata/SSL_Anti-spoofing",
            "upstream_commit": "a54021305d6b3c4c5959ac9395135f63202db8f1",
            "license": "MIT (Academic Research)",
            "uses_ssl_frontend": True,
            "ssl_backbone": "facebook/wav2vec2-large-xlsr-53 (Fairseq)",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "score_orientation": "higher_is_bonafide",
            "execution_target": "Google Colab",
        }
