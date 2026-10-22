"""Standalone sliding-window audio deepfake predictor."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Model

from .config import DEVICE
from .model import AudioClassifier
from .preprocessing import load_audio_windows


class StandaloneAudioPredictor:
    """Predictor for audio deepfakes using audio_model.pt checkpoints with sliding-window analysis."""

    def __init__(self, checkpoint_path: str | Path, device: torch.device | str | None = None):
        self.device = device or DEVICE
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Audio checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.sample_rate = int(checkpoint.get("sample_rate", 16000))
        self.audio_seconds = float(checkpoint.get("audio_seconds", 3.0))
        self.window_hop_seconds = float(checkpoint.get("window_hop_seconds", self.audio_seconds))
        self.threshold = float(checkpoint.get("threshold", 0.5))

        if checkpoint.get("pipeline_version") != 2 or checkpoint.get("aggregation") != "mean_window_probability":
            raise ValueError(
                "This audio checkpoint was trained for single-window scoring. "
                "Retrain it with the full-recording pipeline in training/train_audio.py."
            )

        model_name = checkpoint.get("wav2vec_model", "facebook/wav2vec2-base-960h")
        self.encoder = Wav2Vec2Model.from_pretrained(model_name).to(self.device).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = AudioClassifier(int(checkpoint.get("input_dim", 768))).to(self.device)
        self.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.classifier.eval()

    @torch.no_grad()
    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        """Run sliding-window audio analysis over an entire audio recording."""
        start = time.time()
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, starts, decoded_seconds = load_audio_windows(
            path,
            sample_rate=self.sample_rate,
            audio_seconds=self.audio_seconds,
            window_hop_seconds=self.window_hop_seconds,
        )

        sample_count = int(self.sample_rate * self.audio_seconds)
        window_probabilities: list[float] = []
        batch_size = 8

        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset: offset + batch_size]
            windows = []
            for begin in batch_starts:
                window = waveform[begin: begin + sample_count]
                if len(window) < sample_count:
                    window = np.pad(window, (0, sample_count - len(window)))
                windows.append(window.astype(np.float32, copy=False))

            signal = torch.from_numpy(np.stack(windows)).to(self.device)
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    embedding = self.encoder(signal).last_hidden_state.mean(dim=1)
            else:
                embedding = self.encoder(signal).last_hidden_state.mean(dim=1)

            probabilities = torch.sigmoid(self.classifier(embedding.float())).cpu().numpy()
            window_probabilities.extend(float(value) for value in probabilities)

        probability = float(np.mean(window_probabilities)) if window_probabilities else 0.0
        peak_score = float(np.max(window_probabilities)) if window_probabilities else 0.0

        top_indices = sorted(
            range(len(window_probabilities)),
            key=lambda index: window_probabilities[index],
            reverse=True,
        )[:min(5, len(window_probabilities))]

        top_windows = [
            {
                "start_seconds": round(starts[index] / self.sample_rate, 2),
                "end_seconds": round(min(starts[index] / self.sample_rate + self.audio_seconds, decoded_seconds), 2),
                "fake_score": round(window_probabilities[index], 4),
            }
            for index in top_indices
        ]

        # Forensic splicing & partial manipulation detection:
        # If an audio clip has real speech spliced with synthetic speech, the arithmetic mean
        # can get diluted below the threshold. We check for localized high-confidence synthetic clusters.
        fake_windows = [p for p in window_probabilities if p >= self.threshold]
        strong_fake_windows = [p for p in window_probabilities if p >= 0.85]
        fake_window_ratio = len(fake_windows) / len(window_probabilities) if window_probabilities else 0.0

        is_spliced = False
        if probability >= self.threshold:
            label = "fake"
            verdict_detail = "Full-recording average exceeds synthetic threshold."
        elif len(strong_fake_windows) >= 2 or (len(fake_windows) >= 2 and fake_window_ratio >= 0.20) or (len(window_probabilities) <= 2 and peak_score >= self.threshold):
            label = "fake"
            is_spliced = True
            verdict_detail = f"Spliced / partial manipulation detected across {len(fake_windows)} window(s) (peak: {round(peak_score * 100, 1)}%)."
        elif peak_score >= 0.90:
            label = "fake"
            is_spliced = True
            verdict_detail = f"High-confidence synthetic audio segment detected (peak: {round(peak_score * 100, 1)}%)."
        else:
            label = "real"
            verdict_detail = "All analyzed audio windows are consistent with authentic human speech."

        return {
            "fake_probability": probability,
            "peak_probability": peak_score,
            "label": label,
            "is_spliced": is_spliced,
            "verdict_detail": verdict_detail,
            "threshold": self.threshold,
            "analyzed_seconds": round(decoded_seconds, 2),
            "windows_analyzed": len(window_probabilities),
            "fake_windows_count": len(fake_windows),
            "top_windows": top_windows,
            "inference_seconds": round(time.time() - start, 2),
        }

