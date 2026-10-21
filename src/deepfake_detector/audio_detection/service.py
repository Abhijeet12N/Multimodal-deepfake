"""High-level standalone audio deepfake detection service.

Coordinates decoding, windowing, batch prediction, score orientation handling,
aggregation, and interval merging. Produces structured output schema.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .interfaces import AudioDetector
from .preprocessing import load_audio
from .windowing import AudioWindowGenerator
from .aggregation import RecordingScoreAggregator, merge_suspicious_intervals


class StandaloneAudioService:
    """Service providing end-to-end audio deepfake detection using any AudioDetector backend."""

    def __init__(
        self,
        detector: AudioDetector,
        batch_size: int = 8,
        aggregation_strategy: str = "mean",
        vad_enabled: bool = True,
    ):
        self.detector = detector
        self.batch_size = batch_size
        self.reqs = detector.requirements()
        self.meta = detector.metadata()
        self.vad_enabled = vad_enabled

        self.window_generator = AudioWindowGenerator(
            sample_rate=self.reqs.get("sample_rate", 16000),
            window_samples=self.reqs.get("input_length_samples", 32000),
            hop_samples=self.reqs.get("hop_samples", 16000),
            padding_mode=self.reqs.get("padding_mode", "zero"),
            min_rms_vad=self.reqs.get("min_rms_vad", 0.015),
        )

        self.aggregator = RecordingScoreAggregator(strategy=aggregation_strategy)
        self.threshold = float(self.reqs.get("calibrated_threshold", 0.5))

    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        """Analyze an audio file and produce structured detection results."""
        t0 = time.time()
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, duration = load_audio(path, target_sr=self.reqs["sample_rate"])
        windows = self.window_generator.generate_windows(waveform)

        if not windows:
            return {
                "status": "error",
                "verdict": "error",
                "model_id": self.meta.get("model_id"),
                "model_version": self.meta.get("checkpoint_sha256", "unknown")[:10],
                "raw_recording_score": None,
                "calibrated_fake_probability": None,
                "calibration_version": None,
                "duration_seconds": duration,
                "speech_duration_seconds": 0.0,
                "analysis_window_seconds": self.reqs.get("input_length_seconds"),
                "hop_seconds": self.reqs.get("hop_seconds"),
                "suspicious_intervals": [],
                "warnings": ["Audio recording is empty or unreadable."],
            }

        # Filter windows for inference
        active_windows = [w for w in windows if w.is_speech] if self.vad_enabled else windows
        if not active_windows:
            active_windows = windows  # Fallback to all windows if entire recording is quiet

        speech_duration = sum(w.valid_length_samples for w in active_windows) / self.reqs["sample_rate"]

        # Run batched inference
        all_window_results: list[dict[str, Any]] = []
        fake_window_scores: list[float] = []
        bonafide_window_scores: list[float] = []

        for b_start in range(0, len(active_windows), self.batch_size):
            b_windows = active_windows[b_start : b_start + self.batch_size]
            b_waves = np.stack([w.data for w in b_windows])
            b_lengths = [w.valid_length_samples for w in b_windows]

            pred_res = self.detector.predict_batch(b_waves, lengths=b_lengths)

            # Determine score orientation
            orientation = pred_res.get("score_orientation", self.reqs.get("score_orientation", "higher_is_fake"))

            for idx, win in enumerate(b_windows):
                bonafide_score = None
                if orientation == "higher_is_bonafide":
                    bonafide_score = float(pred_res["bonafide_scores"][idx])
                    fake_score = float(pred_res["fake_scores"][idx])
                    bonafide_window_scores.append(bonafide_score)
                    # Higher fake_score means fake, lower bonafide_score means fake
                    is_flagged = bonafide_score < self.threshold
                else:  # higher_is_fake
                    if "fake_probabilities" in pred_res:
                        fake_score = float(pred_res["fake_probabilities"][idx])
                    else:
                        fake_score = float(pred_res["raw_scores"][idx][1] if pred_res["raw_scores"].ndim > 1 else pred_res["raw_scores"][idx])
                    is_flagged = fake_score >= self.threshold

                fake_window_scores.append(fake_score)
                win_dict = {
                    "start_seconds": win.start_seconds,
                    "end_seconds": win.end_seconds,
                    "fake_score": round(fake_score, 4),
                    "is_fake": is_flagged,
                    "rms_energy": win.rms_energy,
                    "is_speech": win.is_speech,
                }
                if bonafide_score is not None:
                    win_dict["bonafide_score"] = round(bonafide_score, 4)
                all_window_results.append(win_dict)

        # Recording-level aggregation
        rec_score = self.aggregator.aggregate(fake_window_scores)
        if orientation == "higher_is_bonafide":
            # For bonafide orientation, bonafide_rec is evaluated against threshold
            bonafide_rec = float(np.mean(bonafide_window_scores)) if bonafide_window_scores else 0.0
            is_fake_verdict = bonafide_rec < self.threshold
            calibrated_prob = None  # Do not invent uncalibrated probabilities for logits
        else:
            is_fake_verdict = rec_score >= self.threshold
            calibrated_prob = round(rec_score, 4)

        # Flagged windows for interval merging
        flagged = [w for w in all_window_results if w["is_fake"]]
        suspicious_intervals = merge_suspicious_intervals(flagged)

        elapsed = round(time.time() - t0, 3)
        warnings: list[str] = []
        if len(active_windows) < len(windows):
            skipped = len(windows) - len(active_windows)
            warnings.append(f"{skipped} low-energy window(s) were gated out as non-speech.")

        return {
            "status": "ok",
            "verdict": "fake" if is_fake_verdict else "real",
            "model_id": self.meta.get("model_id"),
            "model_version": self.meta.get("checkpoint_sha256", "unknown")[:10],
            "raw_recording_score": round(bonafide_rec if orientation == "higher_is_bonafide" else rec_score, 6),
            "calibrated_fake_probability": calibrated_prob,
            "calibration_version": None,
            "threshold": self.threshold,
            "score_orientation": orientation,
            "duration_seconds": round(duration, 3),
            "speech_duration_seconds": round(speech_duration, 3),
            "analysis_window_seconds": self.reqs.get("input_length_seconds"),
            "hop_seconds": self.reqs.get("hop_seconds"),
            "total_windows": len(windows),
            "evaluated_windows": len(active_windows),
            "suspicious_intervals": suspicious_intervals,
            "top_windows": sorted(all_window_results, key=lambda w: w["fake_score"], reverse=True)[:5],
            "windows": all_window_results,
            "inference_seconds": elapsed,
            "warnings": warnings,
        }
