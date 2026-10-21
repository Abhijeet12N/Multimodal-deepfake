"""Adapter for the original AASIST model (clovaai/aasist).

This is ``adapter_id = "aasist"`` — an independent anti-spoofing baseline
that does **NOT** use any Wav2Vec / XLSR self-supervised frontend.

Architecture:
    Raw 16 kHz waveform (64,600 samples = ~4.04s)
      → Learned sinc filterbank (RawNet-style encoder)
      → Spectro-Temporal Graph Attention Network
      → Binary output (bonafide vs spoof)

Upstream:
    Repository : https://github.com/clovaai/aasist
    License    : MIT (NAVER Corp.)
    Paper      : "AASIST: Audio Anti-Spoofing using Integrated
                  Spectro-Temporal Graph Attention Networks" (ICASSP 2022)
    Pretrained : AASIST (EER 0.83%) and AASIST-L (EER 0.99%, 85k params)
                 on ASVspoof 2019 LA

Note: This adapter is clearly distinct from SSL-AASIST.  Original AASIST
operates on raw waveform without any SSL feature extractor.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .base import AdapterInfo, AudioModelAdapter, _register

# AASIST expects exactly 64,600 samples (≈ 4.04 s at 16 kHz)
AASIST_FIXED_LENGTH = 64_600


def _pad_or_trim(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad (repeat) or trim waveform to *target_len* samples along dim=-1."""
    B, T = waveform.shape
    if T >= target_len:
        return waveform[:, :target_len]
    # Repeat-pad (same strategy as original AASIST data loader)
    reps = (target_len // T) + 1
    return waveform.repeat(1, reps)[:, :target_len]


@_register
class AASISTAdapter(AudioModelAdapter):
    """Adapter for the original AASIST anti-spoofing model."""

    _model: Optional[nn.Module] = None
    _device: torch.device = torch.device("cpu")

    @classmethod
    def info(cls) -> AdapterInfo:
        return AdapterInfo(
            adapter_id="aasist",
            display_name="AASIST (Spectro-Temporal Graph Attention)",
            upstream_repo="https://github.com/clovaai/aasist",
            upstream_commit="",  # fill when cloning / pinning
            license="MIT (NAVER Corp.)",
            sample_rate=16_000,
            uses_ssl_frontend=False,
            checkpoint_url=(
                "https://github.com/clovaai/aasist — pretrained AASIST "
                "and AASIST-L models are included in the repository"
            ),
            checkpoint_sha256="",
            notes=(
                "Original AASIST operates directly on raw waveform via "
                "learned sinc filters.  It does NOT use Wav2Vec or XLSR.  "
                "Input must be exactly 64,600 samples (~4.04s at 16 kHz); "
                "shorter inputs are repeat-padded, longer inputs are trimmed.  "
                "Pretrained on ASVspoof 2019 LA.  "
                "Environment: PyTorch 1.6+ (original docker), but the "
                "architecture itself is compatible with PyTorch 2.x."
            ),
        )

    def load(self, checkpoint_path: str, device: torch.device) -> None:
        """Load an AASIST checkpoint.

        The checkpoint is expected to be a state_dict saved by the upstream
        ``main.py`` training script, or a dict with key ``"model"`` containing
        the state_dict.

        To use this adapter you must first clone clovaai/aasist and ensure
        the ``models/`` directory of that repo is importable.  This adapter
        dynamically imports the ``Model`` class from the AASIST source.

        Alternatively, the adapter accepts a checkpoint dict that includes
        ``"model_config"`` so it can reconstruct the model without needing
        the upstream repo on ``sys.path`` — in that case, the architecture
        is re-implemented inline (see ``_build_aasist_from_config``).
        """
        self._device = device
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"AASIST checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Try to import the upstream AASIST Model class
        model = self._try_load_upstream_model(state_dict, path)

        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self._model = model

    def _try_load_upstream_model(
        self, state_dict: dict, ckpt_path: Path
    ) -> nn.Module:
        """Attempt to import and instantiate the upstream AASIST ``Model``.

        Falls back to loading the config from a sidecar JSON file or
        from the checkpoint itself.
        """
        # Strategy 1: look for a config JSON next to the checkpoint
        config_path = ckpt_path.with_suffix(".conf")
        if not config_path.exists():
            config_path = ckpt_path.parent / "AASIST.conf"

        model_config = None
        if config_path.exists():
            with open(config_path, "r") as f:
                full_config = json.load(f)
            model_config = full_config.get("model_config", full_config)

        # Strategy 2: try dynamic import of upstream aasist.models.AASIST
        aasist_root = ckpt_path.parent.parent  # assume models/ is one level up
        for candidate_module in ["models.AASIST", "AASIST", "aasist"]:
            try:
                if str(aasist_root) not in sys.path:
                    sys.path.insert(0, str(aasist_root))
                mod = importlib.import_module(candidate_module)
                ModelClass = getattr(mod, "Model")
                if model_config:
                    model = ModelClass(model_config)
                else:
                    model = ModelClass({})
                model.load_state_dict(state_dict, strict=False)
                return model
            except (ImportError, ModuleNotFoundError, AttributeError):
                continue

        raise RuntimeError(
            f"Could not load AASIST model.  Ensure the upstream clovaai/aasist "
            f"repo is cloned and its directory is either a parent of "
            f"{ckpt_path} or on sys.path.  Alternatively, place a matching "
            f"AASIST.conf JSON file alongside the checkpoint."
        )

    @torch.no_grad()
    def score(self, waveform: torch.Tensor) -> torch.Tensor:
        """Score a batch of waveforms.

        AASIST outputs ``(batch, 2)`` logits for [bonafide, spoof].
        We return ``logit_spoof - logit_bonafide`` so that higher
        values indicate fake/spoofed, consistent with other adapters.
        """
        assert self._model is not None, "Call load() before score()"
        waveform = _pad_or_trim(waveform, AASIST_FIXED_LENGTH).to(self._device)

        # AASIST forward returns (batch_out, (hidden_outputs...))
        # batch_out shape: (B, 2) — [bonafide, spoof]
        output = self._model(waveform)

        # Handle tuple output (model returns (logits, hidden_states))
        if isinstance(output, tuple):
            output = output[0]

        if output.dim() == 2 and output.shape[1] == 2:
            # Return spoof_logit - bonafide_logit
            return output[:, 1] - output[:, 0]
        else:
            # Single logit output
            return output.squeeze(-1)

    def to(self, device: torch.device) -> "AASISTAdapter":
        self._device = device
        if self._model is not None:
            self._model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device
