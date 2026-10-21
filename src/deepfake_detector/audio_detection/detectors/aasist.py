"""AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) Detector.

Upstream repository: https://github.com/clovaai/aasist
Reference: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated
Spectro-Temporal Graph Attention Networks", ICASSP 2022.

Architecture:
- Raw waveform input (no Wav2Vec / XLSR frontend)
- Sinc-convolution filterbank
- Spectro-temporal graph attention with heterogeneous master node
- 2-class output: [spoof (index 0), bonafide (index 1)]
- Native input length: 64,600 samples (~4.04s at 16 kHz)
- Native padding: periodic repeat tiling
- Score orientation: UPSTREAM EVALUATION USES BONAFIDE LOGIT (higher_is_bonafide)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import AudioDetector

# Ensure upstream aasist directory is importable
AASIST_UPSTREAM_DIR = Path(__file__).resolve().parents[4] / "aasist"
if AASIST_UPSTREAM_DIR.is_dir() and str(AASIST_UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(AASIST_UPSTREAM_DIR))


def _load_upstream_aasist_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    """Import and instantiate the official AASIST Model class."""
    try:
        from models.AASIST import Model
        return Model(config).to(device)
    except ImportError:
        # If aasist repo is moved, find it relative to current working directory
        cwd_aasist = Path.cwd().parent / "aasist"
        if cwd_aasist.is_dir() and str(cwd_aasist) not in sys.path:
            sys.path.insert(0, str(cwd_aasist))
        from models.AASIST import Model
        return Model(config).to(device)


class AASISTDetector(AudioDetector):
    """Audio detector adapter for the official AASIST and AASIST-L models."""

    DEFAULT_CONFIG = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    }

    AASIST_L_CONFIG = {
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    }

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        is_lightweight: bool = False,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.is_lightweight = is_lightweight
        self.checkpoint_path: str = ""
        self.checkpoint_sha256: str = ""
        self.threshold: float = 0.0  # Logit threshold for bonafide decision

        if checkpoint_path is not None:
            self.load(str(checkpoint_path), device=device)

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"AASIST checkpoint not found: {path}")

        self.checkpoint_path = str(path)
        with open(path, "rb") as f:
            self.checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

        if "aasist-l" in path.name.lower():
            self.is_lightweight = True

        cfg = self.AASIST_L_CONFIG if self.is_lightweight else self.DEFAULT_CONFIG
        self.model = _load_upstream_aasist_model(cfg, self.device)

        state_dict = torch.load(str(path), map_location=self.device, weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            # Switch variant if state dict key shapes indicate AASIST vs AASIST-L
            self.is_lightweight = not self.is_lightweight
            cfg = self.AASIST_L_CONFIG if self.is_lightweight else self.DEFAULT_CONFIG
            self.model = _load_upstream_aasist_model(cfg, self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

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
        }

    @staticmethod
    def pad_waveform(x: torch.Tensor, max_len: int = 64600) -> torch.Tensor:
        """Periodic repeat padding matching upstream AASIST data_utils.py."""
        batch_padded = []
        for row in x:
            length = row.size(0)
            if length >= max_len:
                batch_padded.append(row[:max_len])
            else:
                repeats = int(math.ceil(max_len / max(1, length)))
                padded = row.repeat(repeats)[:max_len]
                batch_padded.append(padded)
        return torch.stack(batch_padded)

    @torch.no_grad()
    def predict_batch(
        self,
        waveforms: torch.Tensor | np.ndarray,
        lengths: list[int] | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("AASIST model is not loaded. Call load() first.")

        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms)

        waveforms = waveforms.float().to(self.device)
        batch_size = waveforms.size(0)
        target_len = self.requirements()["input_length_samples"]

        if waveforms.size(1) != target_len:
            waveforms = self.pad_waveform(waveforms, target_len).to(self.device)

        _, raw_logits = self.model(waveforms)  # shape: (batch_size, 2)
        bonafide_logits = raw_logits[:, 1].cpu().numpy()  # Upstream evaluated score

        # Explicit synthetic / fake score: logit difference (spoof_logit - bonafide_logit)
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
        name = "AASIST-L" if self.is_lightweight else "AASIST"
        return {
            "model_id": "aasist_l" if self.is_lightweight else "aasist",
            "display_name": f"Official {name} Anti-Spoofing Baseline",
            "upstream_repo": "https://github.com/clovaai/aasist",
            "upstream_commit": "official-main",
            "license": "MIT (NAVER Corp.)",
            "uses_ssl_frontend": False,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "score_orientation": "higher_is_bonafide",
        }
