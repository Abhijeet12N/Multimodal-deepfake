"""SSL-AASIST (Self-Supervised Wav2Vec 2.0 / XLSR + AASIST) Detector Adapter.

Upstream repository: https://github.com/TakHemlata/SSL_Anti-spoofing
Reference: Tak et al., "End-to-End Anti-Spoofing with Self-Supervised
Representations and RawBoost Data Augmentation", 2022.

Architecture:
- Frontend: Wav2Vec 2.0 XLSR-53 / XLS-R self-supervised acoustic representation
- Backend: AASIST spectro-temporal graph attention network
- Training: RawBoost data augmentation (impulsive noise, colored noise, linear filtering)
- 2-class output: [spoof (index 0), bonafide (index 1)]
- Native input length: 64,600 samples (~4.04s at 16 kHz)
- Score orientation: UPSTREAM EVALUATION USES BONAFIDE LOGIT (higher_is_bonafide)
- Training Target: Google Colab (due to Fairseq, RawBoost, and CUDA GPU requirements)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ..interfaces import AudioDetector


class SSLAASISTDetector(AudioDetector):
    """Audio detector adapter for SSL-AASIST (XLSR + AASIST backend).

    This model serves as the primary upgrade candidate over the legacy Wav2Vec baseline.
    Full training and large-scale evaluation require Google Colab GPU due to upstream
    Fairseq dependencies.
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
            raise FileNotFoundError(f"SSL-AASIST checkpoint not found: {path}")

        self.checkpoint_path = str(path)
        with open(path, "rb") as f:
            self.checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Check for Fairseq / XLSR upstream environment
        try:
            import fairseq  # type: ignore
        except ImportError:
            raise ImportError(
                "SSL-AASIST upstream backend requires 'fairseq'. "
                "As documented in the upgrade plan, SSL-AASIST training and evaluation "
                "should be run in the target Google Colab environment with Fairseq installed."
            )

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model = ckpt.get("model", ckpt)
        if isinstance(self.model, nn.Module):
            self.model.eval()

    def requirements(self) -> dict[str, Any]:
        return {
            "sample_rate": 16000,
            "input_length_samples": 64600,  # ~4.0375 seconds native input
            "input_length_seconds": 64600 / 16000,
            "hop_samples": 32000,           # ~2.0 second default hop
            "hop_seconds": 2.0,
            "padding_mode": "repeat",
            "raw_score_semantics": "two_class_logits",
            "score_orientation": "higher_is_bonafide",  # Upstream evaluates output[:, 1]
            "class_index_mapping": {0: "spoof", 1: "bonafide"},
            "calibrated_threshold": self.threshold,
            "environment_constraints": "Requires Fairseq, CUDA GPU, and Google Colab runtime",
        }

    @torch.no_grad()
    def predict_batch(
        self,
        waveforms: torch.Tensor | np.ndarray,
        lengths: list[int] | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError(
                "SSL-AASIST model is not loaded. Ensure fairseq is installed and a checkpoint is provided. "
                "For training/evaluation, execute on Google Colab."
            )

        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms)

        waveforms = waveforms.float().to(self.device)
        batch_size = waveforms.size(0)
        target_len = self.requirements()["input_length_samples"]

        # Predict
        raw_logits = self.model(waveforms)
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
            "display_name": "SSL-AASIST (XLSR + AASIST Backend)",
            "upstream_repo": "https://github.com/TakHemlata/SSL_Anti-spoofing",
            "upstream_commit": "upstream-main",
            "license": "MIT",
            "uses_ssl_frontend": True,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "score_orientation": "higher_is_bonafide",
            "training_pipeline": "Google Colab (Fairseq + RawBoost)",
        }
