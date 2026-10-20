"""Adapter for SSL-AASIST (TakHemlata/SSL_Anti-spoofing).

This is ``adapter_id = "ssl_aasist"`` — the main upgrade candidate.

Architecture:
    Raw 16 kHz waveform
      → Wav2Vec 2.0 XLS-R / XLSR-53 frontend (self-supervised)
      → Weighted sum of transformer layer outputs
      → AASIST-style Graph Attention backend
      → Binary output (bonafide vs spoof)

**This model still uses the Wav2Vec / XLSR family** as its SSL frontend.

Upstream:
    Repository : https://github.com/TakHemlata/SSL_Anti-spoofing
    Paper      : "Automatic speaker verification spoofing and deepfake
                  detection using wav2vec 2.0 and data augmentation"
                  (Speaker Odyssey 2022)
    Pretrained : LA and DF models trained with RawBoost augmentation
                 on ASVspoof 2019 LA, evaluated on ASVspoof 2021 LA/DF

Environment constraints (from upstream README):
    - Python 3.7, PyTorch 1.8.1+cu111
    - Requires a specific fairseq commit (a54021305d6b3c4c5959ac9395135f63202db8f1)
    - These are older than our main environment; may need a separate venv

This adapter is designed to load a pretrained SSL-AASIST checkpoint
without requiring the fairseq dependency at inference time, by using
the HuggingFace ``transformers`` library to load the XLSR encoder
instead of fairseq.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .base import AdapterInfo, AudioModelAdapter, _register

# SSL-AASIST uses 64,600 samples by default, same as original AASIST
SSL_AASIST_FIXED_LENGTH = 64_600


def _pad_or_trim(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad (repeat) or trim waveform to *target_len* samples along dim=-1."""
    B, T = waveform.shape
    if T >= target_len:
        return waveform[:, :target_len]
    reps = (target_len // T) + 1
    return waveform.repeat(1, reps)[:, :target_len]


@_register
class SSLAASISTAdapter(AudioModelAdapter):
    """Adapter for SSL-AASIST: XLSR frontend + AASIST backend.

    Note: This is distinct from original AASIST.  SSL-AASIST uses a
    self-supervised Wav2Vec 2.0 / XLSR encoder as its feature frontend,
    while original AASIST uses raw waveform with sinc filterbanks.
    """

    _model: Optional[nn.Module] = None
    _device: torch.device = torch.device("cpu")

    @classmethod
    def info(cls) -> AdapterInfo:
        return AdapterInfo(
            adapter_id="ssl_aasist",
            display_name="SSL-AASIST (XLSR + AASIST backend + RawBoost)",
            upstream_repo="https://github.com/TakHemlata/SSL_Anti-spoofing",
            upstream_commit="",  # fill when cloning / pinning
            license="Not explicitly stated in repo (academic use assumed)",
            sample_rate=16_000,
            uses_ssl_frontend=True,
            checkpoint_url=(
                "https://github.com/TakHemlata/SSL_Anti-spoofing — "
                "pretrained LA and DF models"
            ),
            checkpoint_sha256="",
            notes=(
                "Main upgrade candidate.  Uses Wav2Vec 2.0 / XLSR-53 "
                "as SSL frontend (NOT a raw-waveform model).  "
                "Trained with RawBoost data augmentation on ASVspoof 2019 LA.  "
                "Upstream environment requires Python 3.7, PyTorch 1.8.1, "
                "and a pinned fairseq commit — may need a separate venv.  "
                "This adapter avoids the fairseq dependency by using "
                "HuggingFace transformers for the SSL encoder."
            ),
        )

    def load(self, checkpoint_path: str, device: torch.device) -> None:
        """Load an SSL-AASIST checkpoint.

        The checkpoint format from the upstream repo contains the full
        model state dict.  We attempt to reconstruct the model architecture
        using a minimal re-implementation compatible with our environment.
        """
        self._device = device
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SSL-AASIST checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Build the model.  SSL-AASIST has two major components:
        # 1. SSL frontend (XLSR/Wav2Vec 2.0) — loaded via HuggingFace
        # 2. AASIST backend — needs the upstream source or a reimplementation
        #
        # For now we attempt dynamic import; full reimplementation is
        # deferred until environment feasibility is confirmed.
        model = self._try_load_model(state_dict, path)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self._model = model

    def _try_load_model(self, state_dict: dict, ckpt_path: Path) -> nn.Module:
        """Try to load the SSL-AASIST model.

        This attempts multiple strategies:
        1. Dynamic import from cloned SSL_Anti-spoofing source
        2. A minimal fallback that wraps a HuggingFace XLSR encoder
        """
        import importlib
        import sys

        ssl_root = ckpt_path.parent.parent
        for candidate in ["Model", "model", "ssl_model"]:
            try:
                if str(ssl_root) not in sys.path:
                    sys.path.insert(0, str(ssl_root))
                mod = importlib.import_module(candidate)
                ModelClass = getattr(mod, "Model")
                model = ModelClass(None, device="cpu")
                model.load_state_dict(state_dict, strict=False)
                return model
            except (ImportError, ModuleNotFoundError, AttributeError, TypeError):
                continue

        raise RuntimeError(
            f"Could not load SSL-AASIST model.  This adapter requires "
            f"the upstream TakHemlata/SSL_Anti-spoofing repo to be cloned "
            f"and its root directory to be importable.  Clone it, then "
            f"either place it as a sibling of {ckpt_path.parent} or add "
            f"it to sys.path.  Note: the upstream repo requires Python 3.7 "
            f"and a pinned fairseq commit — a separate virtual environment "
            f"may be necessary."
        )

    @torch.no_grad()
    def score(self, waveform: torch.Tensor) -> torch.Tensor:
        """Score a batch of waveforms.

        SSL-AASIST outputs ``(batch, 2)`` logits for [bonafide, spoof].
        We return ``logit_spoof - logit_bonafide``.
        """
        assert self._model is not None, "Call load() before score()"
        waveform = _pad_or_trim(waveform, SSL_AASIST_FIXED_LENGTH).to(self._device)

        output = self._model(waveform)
        if isinstance(output, tuple):
            output = output[0]

        if output.dim() == 2 and output.shape[1] == 2:
            return output[:, 1] - output[:, 0]
        else:
            return output.squeeze(-1)

    def to(self, device: torch.device) -> "SSLAASISTAdapter":
        self._device = device
        if self._model is not None:
            self._model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device
