"""Detector interface for standalone audio anti-spoofing and deepfake detection.

Defines the contract that every model adapter must satisfy:
- Explicit requirements (sample rate, native length, padding mode, score semantics)
- Batch prediction with explicit raw scores and lengths
- Metadata and upstream provenance tracking
- Strict separation of raw scores from calibrated probabilities
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch


class AudioDetector(abc.ABC):
    """Abstract interface for all audio deepfake / anti-spoofing detectors."""

    @abc.abstractmethod
    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        """Load detector weights from *checkpoint_path* onto *device*.

        Parameters
        ----------
        checkpoint_path : str
            Path to the local checkpoint file.
        device : str
            Torch device string, e.g. 'cpu', 'cuda', 'cuda:0'.
        """

    @abc.abstractmethod
    def requirements(self) -> dict[str, Any]:
        """Declare sample rate, native input length, padding, and score semantics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'sample_rate': int (e.g. 16000)
            - 'input_length_samples': int (e.g. 32000 for legacy, 64600 for AASIST)
            - 'input_length_seconds': float
            - 'padding_mode': str ('repeat', 'zero', 'reflect')
            - 'raw_score_semantics': str ('logits', 'probability', 'two_class_logits')
            - 'score_orientation': str ('higher_is_fake' or 'higher_is_bonafide')
            - 'class_index_mapping': dict[int, str]
        """

    @abc.abstractmethod
    def predict_batch(
        self,
        waveforms: torch.Tensor | np.ndarray,
        lengths: list[int] | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute raw detection scores for a batch of audio waveforms.

        Parameters
        ----------
        waveforms : torch.Tensor | np.ndarray
            Input tensor of shape (batch_size, num_samples).
        lengths : list[int] | torch.Tensor | None
            Actual unpadded sample count for each item in the batch.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'raw_scores': np.ndarray of shape (batch_size,) or (batch_size, num_classes)
            - 'score_orientation': str ('higher_is_fake' or 'higher_is_bonafide')
            - 'lengths': list[int]
            - 'model_metadata': dict[str, Any]
        """

    @abc.abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return model metadata, upstream provenance, and licensing information.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'model_id': str
            - 'display_name': str
            - 'upstream_repo': str
            - 'upstream_commit': str
            - 'license': str
            - 'uses_ssl_frontend': bool
            - 'checkpoint_path': str
            - 'checkpoint_sha256': str
        """
