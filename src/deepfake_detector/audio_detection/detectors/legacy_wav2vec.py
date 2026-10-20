"""Legacy Wav2Vec 2.0 + MLP Audio Detector.

Wraps the existing production model (facebook/wav2vec2-base CNN acoustic encoder
with DeepfakeClassifier MLP head, trained on ASVspoof 2019 LA).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from ..interfaces import AudioDetector


class DeepfakeClassifier(nn.Module):
    """Legacy 2-class MLP classifier operating on 512-dim acoustic representations."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class LegacyWav2VecDetector(AudioDetector):
    """Audio detector wrapping the legacy Wav2Vec 2.0 CNN + MLP baseline."""

    def __init__(self, checkpoint_path: str | Path | None = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.backbone: Optional[Wav2Vec2Model] = None
        self.processor: Optional[Wav2Vec2Processor] = None
        self.classifier: Optional[DeepfakeClassifier] = None
        self.checkpoint_path: str = ""
        self.checkpoint_sha256: str = ""
        self.threshold: float = 0.9849822521209717  # dev_threshold from wav2vec_best.pt

        if checkpoint_path is not None:
            self.load(str(checkpoint_path), device=device)

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Legacy checkpoint not found: {path}")

        self.checkpoint_path = str(path)
        with open(path, "rb") as f:
            self.checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Load Wav2Vec2 processor and backbone (preferring offline cache)
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base", local_files_only=True
            )
            self.backbone = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base", local_files_only=True
            ).to(self.device).eval()
        except Exception:
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device).eval()

        for p in self.backbone.parameters():
            p.requires_grad = False

        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        input_dim = int(ckpt.get("input_dim", 512))
        hidden_dim = int(ckpt.get("hidden_dim", 256))
        dropout = float(ckpt.get("dropout", 0.3))

        self.classifier = DeepfakeClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device).eval()

        state_dict = ckpt.get("model_state_dict", ckpt)
        self.classifier.load_state_dict(state_dict)

        if "dev_threshold" in ckpt:
            self.threshold = float(ckpt["dev_threshold"])

    def requirements(self) -> dict[str, Any]:
        return {
            "sample_rate": 16000,
            "input_length_samples": 32000,  # 2.0 seconds native window
            "input_length_seconds": 2.0,
            "hop_samples": 16000,           # 1.0 second default hop
            "hop_seconds": 1.0,
            "padding_mode": "zero",
            "raw_score_semantics": "two_class_logits",
            "score_orientation": "higher_is_fake",
            "class_index_mapping": {0: "bonafide", 1: "spoof"},
            "calibrated_threshold": self.threshold,
            "min_rms_vad": 0.015,
        }

    @torch.no_grad()
    def predict_batch(
        self,
        waveforms: torch.Tensor | np.ndarray,
        lengths: list[int] | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.backbone is None or self.classifier is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        if isinstance(waveforms, np.ndarray):
            waveforms = torch.from_numpy(waveforms)

        waveforms = waveforms.float().to(self.device)
        batch_size, num_samples = waveforms.shape
        target_len = self.requirements()["input_length_samples"]

        # Ensure correct window size
        if num_samples < target_len:
            pad_amount = target_len - num_samples
            waveforms = torch.nn.functional.pad(waveforms, (0, pad_amount), value=0.0)
        elif num_samples > target_len:
            waveforms = waveforms[:, :target_len]

        # Extract 512-dim CNN acoustic representations
        cnn_features = self.backbone.feature_extractor(waveforms)
        cnn_features = cnn_features.transpose(1, 2)
        embeddings = cnn_features.mean(dim=1)  # (batch_size, 512)

        raw_logits = self.classifier(embeddings)  # (batch_size, 2)
        probs = torch.softmax(raw_logits, dim=-1)
        spoof_probs = probs[:, 1].cpu().numpy()  # index 1 = spoof / fake

        if lengths is None:
            resolved_lengths = [num_samples] * batch_size
        elif isinstance(lengths, torch.Tensor):
            resolved_lengths = lengths.cpu().tolist()
        else:
            resolved_lengths = list(lengths)

        return {
            "raw_scores": raw_logits.cpu().numpy(),
            "fake_probabilities": spoof_probs,
            "score_orientation": "higher_is_fake",
            "lengths": resolved_lengths,
            "model_metadata": self.metadata(),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": "legacy_wav2vec_mlp",
            "display_name": "Legacy Wav2Vec 2.0 + MLP",
            "upstream_repo": "https://github.com/facebookresearch/fairseq",
            "upstream_commit": "official-wav2vec2-base",
            "license": "Apache-2.0 / MIT",
            "uses_ssl_frontend": True,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "threshold": self.threshold,
        }
