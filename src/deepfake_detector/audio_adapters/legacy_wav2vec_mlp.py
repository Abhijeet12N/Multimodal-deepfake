"""Adapter wrapping the existing Wav2Vec 2.0 + MLP baseline.

This is ``adapter_id = "legacy_wav2vec_mlp"`` and exists solely to give the
current production model a slot in the adapter framework so it can be
compared side-by-side with newer backends.

Architecture (unchanged):
    Raw 16 kHz waveform
      → Frozen Wav2Vec 2.0 base (facebook/wav2vec2-base-960h)
      → Mean-pool over time → 768-dim embedding
      → LayerNorm → Linear(768,256) → GELU → Dropout(0.25)
                  → Linear(256,64)  → GELU → Dropout(0.15)
                  → Linear(64,1)    → logit

Upstream: This application itself (no external repo).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import Wav2Vec2Model

from .base import AdapterInfo, AudioModelAdapter, _register


@_register
class LegacyWav2VecMLP(AudioModelAdapter):
    """Preserves the existing Wav2Vec 2.0 + MLP baseline as an adapter."""

    _encoder: Optional[Wav2Vec2Model] = None
    _classifier: Optional[torch.nn.Module] = None
    _device: torch.device = torch.device("cpu")
    _sample_rate: int = 16_000

    @classmethod
    def info(cls) -> AdapterInfo:
        return AdapterInfo(
            adapter_id="legacy_wav2vec_mlp",
            display_name="Legacy Wav2Vec 2.0 + MLP (baseline)",
            upstream_repo="https://github.com/Abhijeet12N/Multimodal-deepfake",
            upstream_commit="HEAD",
            license="MIT",
            sample_rate=16_000,
            uses_ssl_frontend=True,
            checkpoint_url="models/audio_model.pt",
            checkpoint_sha256="",
            notes=(
                "This is the existing production model shipped with the "
                "application.  The Wav2Vec 2.0 encoder is loaded from "
                "HuggingFace on first use; the classifier head is loaded "
                "from the checkpoint."
            ),
        )

    def load(self, checkpoint_path: str, device: torch.device) -> None:
        from ..model import AudioClassifier  # avoid circular import

        self._device = device
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")

        model_name = ckpt.get("wav2vec_model", "facebook/wav2vec2-base-960h")
        self._sample_rate = int(ckpt.get("sample_rate", 16_000))

        self._encoder = Wav2Vec2Model.from_pretrained(model_name)
        self._encoder.to(device).eval()
        for p in self._encoder.parameters():
            p.requires_grad = False

        input_dim = int(ckpt.get("input_dim", 768))
        self._classifier = AudioClassifier(input_dim).to(device)
        self._classifier.load_state_dict(ckpt["classifier_state_dict"])
        self._classifier.eval()

    @torch.no_grad()
    def score(self, waveform: torch.Tensor) -> torch.Tensor:
        assert self._encoder is not None, "Call load() before score()"
        waveform = waveform.to(self._device)
        if self._device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                emb = self._encoder(waveform).last_hidden_state.mean(dim=1)
        else:
            emb = self._encoder(waveform).last_hidden_state.mean(dim=1)
        logits = self._classifier(emb.float())
        return logits  # raw logits — caller applies sigmoid if needed

    def to(self, device: torch.device) -> "LegacyWav2VecMLP":
        self._device = device
        if self._encoder is not None:
            self._encoder.to(device)
        if self._classifier is not None:
            self._classifier.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device
