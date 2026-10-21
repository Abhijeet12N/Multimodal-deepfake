"""Abstract base class and registry for audio detection model adapters.

Every adapter must:
- accept raw waveform at a documented sample rate
- return per-utterance logit(s) or probability
- declare its upstream provenance via ``AdapterInfo``
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class AdapterInfo:
    """Immutable provenance record for an adapter.

    Attributes
    ----------
    adapter_id : str
        Short machine-readable identifier (e.g. ``legacy_wav2vec_mlp``).
    display_name : str
        Human-readable name for reports and logs.
    upstream_repo : str
        URL of the upstream Git repository.
    upstream_commit : str
        Pinned commit hash of the upstream source.
    license : str
        SPDX identifier or short description of the upstream license.
    sample_rate : int
        Expected input sample rate in Hz.
    uses_ssl_frontend : bool
        ``True`` if the model uses a Wav2Vec / XLSR / HuBERT self-supervised
        frontend; ``False`` for raw-waveform models like original AASIST.
    checkpoint_url : str
        URL or path where the pretrained checkpoint can be obtained.
    checkpoint_sha256 : str
        SHA-256 hex digest of the checkpoint file (for reproducibility).
    notes : str
        Any additional notes about environment constraints or caveats.
    """

    adapter_id: str
    display_name: str
    upstream_repo: str
    upstream_commit: str = ""
    license: str = "unknown"
    sample_rate: int = 16_000
    uses_ssl_frontend: bool = False
    checkpoint_url: str = ""
    checkpoint_sha256: str = ""
    notes: str = ""


class AudioModelAdapter(abc.ABC):
    """Uniform interface every audio anti-spoofing adapter must implement."""

    @classmethod
    @abc.abstractmethod
    def info(cls) -> AdapterInfo:
        """Return immutable provenance metadata."""

    @abc.abstractmethod
    def load(self, checkpoint_path: str, device: torch.device) -> None:
        """Load model weights from *checkpoint_path* onto *device*.

        After this call, :meth:`score` must be functional.
        """

    @abc.abstractmethod
    @torch.no_grad()
    def score(self, waveform: torch.Tensor) -> torch.Tensor:
        """Score a batch of waveforms.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape ``(batch, samples)`` at :pyattr:`info().sample_rate`.

        Returns
        -------
        torch.Tensor
            Shape ``(batch,)`` — higher values indicate **fake / spoofed**.
        """

    def to(self, device: torch.device) -> "AudioModelAdapter":
        """Move internal model state to *device* (optional override)."""
        return self

    @property
    def device(self) -> torch.device:
        """Return the device the model is currently on."""
        return torch.device("cpu")


# ── Global registry ──────────────────────────────────────────────────────────

ADAPTER_REGISTRY: Dict[str, type] = {}


def _register(cls: type) -> type:
    """Class decorator – registers an adapter under its ``info().adapter_id``."""
    adapter_id = cls.info().adapter_id
    if adapter_id in ADAPTER_REGISTRY:
        raise ValueError(f"Duplicate adapter id: {adapter_id!r}")
    ADAPTER_REGISTRY[adapter_id] = cls
    return cls


def get_adapter(adapter_id: str) -> type:
    """Look up an adapter class by its *adapter_id*.

    Raises ``KeyError`` with a helpful message listing known IDs.
    """
    try:
        return ADAPTER_REGISTRY[adapter_id]
    except KeyError:
        known = ", ".join(sorted(ADAPTER_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown audio adapter {adapter_id!r}. Known adapters: {known}"
        ) from None
