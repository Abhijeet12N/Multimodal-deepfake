"""Window generation and slicing module.

Extracts model-aware overlapping windows, computes exact timestamps, handles VAD gating,
and tracks valid vs padded sample lengths without confusing padded lengths with real audio duration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np


@dataclass
class AudioWindow:
    """Represents a single analysis window extracted from a recording."""

    window_index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    valid_length_samples: int
    total_padded_samples: int
    rms_energy: float
    is_speech: bool
    data: np.ndarray


class AudioWindowGenerator:
    """Generates model-aware sliding windows with explicit padding and timestamp tracking."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_samples: int = 32000,
        hop_samples: int = 16000,
        padding_mode: str = "zero",  # "zero", "repeat"
        min_rms_vad: float = 0.015,
        min_duration_seconds: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self.padding_mode = padding_mode
        self.min_rms_vad = min_rms_vad
        self.min_duration_seconds = min_duration_seconds

    def pad_chunk(self, chunk: np.ndarray, target_length: int) -> np.ndarray:
        """Apply model-specific padding policy."""
        if len(chunk) >= target_length:
            return chunk[:target_length]
        if self.padding_mode == "repeat":
            repeats = int(math.ceil(target_length / max(1, len(chunk))))
            return np.tile(chunk, repeats)[:target_length].astype(np.float32, copy=False)
        else:  # "zero"
            pad_amount = target_length - len(chunk)
            return np.pad(chunk, (0, pad_amount), mode="constant", constant_values=0.0).astype(np.float32, copy=False)

    def generate_windows(self, waveform: np.ndarray) -> list[AudioWindow]:
        """Slice recording into windows with exact timestamps and energy gating."""
        total_samples = len(waveform)
        if total_samples == 0:
            return []

        # Short audio handling: if audio is shorter than one window
        if total_samples <= self.window_samples:
            rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
            padded = self.pad_chunk(waveform, self.window_samples)
            is_speech = rms >= self.min_rms_vad
            return [
                AudioWindow(
                    window_index=0,
                    start_sample=0,
                    end_sample=total_samples,
                    start_seconds=0.0,
                    end_seconds=round(total_samples / self.sample_rate, 4),
                    valid_length_samples=total_samples,
                    total_padded_samples=self.window_samples,
                    rms_energy=round(rms, 6),
                    is_speech=is_speech,
                    data=padded,
                )
            ]

        windows: list[AudioWindow] = []
        start = 0
        idx = 0

        while start + self.window_samples <= total_samples:
            chunk = waveform[start : start + self.window_samples]
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            t_start = round(start / self.sample_rate, 4)
            t_end = round((start + self.window_samples) / self.sample_rate, 4)

            windows.append(
                AudioWindow(
                    window_index=idx,
                    start_sample=start,
                    end_sample=start + self.window_samples,
                    start_seconds=t_start,
                    end_seconds=t_end,
                    valid_length_samples=self.window_samples,
                    total_padded_samples=self.window_samples,
                    rms_energy=round(rms, 6),
                    is_speech=rms >= self.min_rms_vad,
                    data=chunk.astype(np.float32, copy=False),
                )
            )
            idx += 1
            start += self.hop_samples

        # Trailing partial window coverage: ensure tail section is not silently skipped
        if total_samples - start >= int(self.min_duration_seconds * self.sample_rate):
            chunk = waveform[start:]
            valid_len = len(chunk)
            rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
            t_start = round(start / self.sample_rate, 4)
            t_end = round(total_samples / self.sample_rate, 4)
            padded = self.pad_chunk(chunk, self.window_samples)

            windows.append(
                AudioWindow(
                    window_index=idx,
                    start_sample=start,
                    end_sample=total_samples,
                    start_seconds=t_start,
                    end_seconds=t_end,
                    valid_length_samples=valid_len,
                    total_padded_samples=self.window_samples,
                    rms_energy=round(rms, 6),
                    is_speech=rms >= self.min_rms_vad,
                    data=padded,
                )
            )

        return windows
