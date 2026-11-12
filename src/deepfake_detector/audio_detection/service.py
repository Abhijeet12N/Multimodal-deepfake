"""High-level standalone audio deepfake detection service.

Coordinates decoding, windowing, batch prediction, score orientation handling,
aggregation, score calibration, uncertainty quantification, and interval merging.
Produces structured output schema.
"""
from __future__ import annotations

import time
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .interfaces import AudioDetector
from .preprocessing import load_audio
from .windowing import AudioWindowGenerator
from .aggregation import RecordingScoreAggregator, merge_suspicious_intervals
from .calibration import ScoreCalibrator


class StandaloneAudioService:
    """Service providing end-to-end audio deepfake detection using any AudioDetector backend."""

    def __init__(
        self,
        detector: AudioDetector,
        batch_size: int = 8,
        aggregation_strategy: str = "mean",
        vad_enabled: bool = True,
        calibrator: Optional[ScoreCalibrator] = None,
        uncertain_band: Optional[Tuple[float, float]] = None,
    ):
        self.detector = detector
        self.batch_size = batch_size
        self.reqs = detector.requirements()
        self.meta = detector.metadata()
        self.vad_enabled = vad_enabled
        self.calibrator = calibrator
        self.uncertain_band = uncertain_band

        self.window_generator = AudioWindowGenerator(
            sample_rate=self.reqs.get("sample_rate", 16000),
            window_samples=self.reqs.get("input_length_samples", 32000),
            hop_samples=self.reqs.get("hop_samples", 16000),
            padding_mode=self.reqs.get("padding_mode", "zero"),
            min_rms_vad=self.reqs.get("min_rms_vad", 0.015),
        )

        self.aggregator = RecordingScoreAggregator(strategy=aggregation_strategy)
        self.threshold = float(self.reqs.get("calibrated_threshold", 0.5))

    def pipeline_signature(self) -> str:
        """Bind recording calibration to weights, preprocessing and aggregation."""
        pipeline = {
            "model_id": self.meta.get("model_id"),
            "checkpoint_sha256": self.meta.get("checkpoint_sha256"),
            "preprocessing_version": self.meta.get("preprocessing_version", "upstream_v1"),
            "requirements": self.reqs,
            "aggregation": vars(self.aggregator),
            "vad_enabled": self.vad_enabled,
            "decoder_version": "soundfile_soxr_hq_ffmpeg_v1",
        }
        return hashlib.sha256(json.dumps(pipeline, sort_keys=True).encode()).hexdigest()

    def accepts_calibrator(self, candidate: ScoreCalibrator) -> bool:
        provenance = candidate.fit_data
        return bool(
            candidate.fitted
            and candidate.model_id == self.meta.get("model_id")
            and candidate.score_orientation == self.reqs.get("score_orientation")
            and provenance.get("split") in {"dev", "val"}
            and int(provenance.get("sample_count", 0)) >= 1000
            and provenance.get("protocol") == "studio_sliding_window"
            and provenance.get("score_level") == "recording"
            and provenance.get("pipeline_signature") == self.pipeline_signature()
        )

    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        """Analyze an audio file and produce structured detection results."""
        t0 = time.time()
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        waveform, duration = load_audio(path, target_sr=self.reqs["sample_rate"])
        decoded_at = time.perf_counter()
        decode_seconds = time.time() - t0
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
        inference_started = time.perf_counter()

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
                    # A display-only two-class softmax score. This is not a
                    # calibrated probability and does not drive the verdict.
                    "display_fake_score": round(
                        fake_score if orientation == "higher_is_fake"
                        else float(1.0 / (1.0 + np.exp(-np.clip(fake_score, -60.0, 60.0)))),
                        4,
                    ),
                    "is_fake": is_flagged,
                    "rms_energy": win.rms_energy,
                    "is_speech": win.is_speech,
                }
                if bonafide_score is not None:
                    win_dict["bonafide_score"] = round(bonafide_score, 4)
                all_window_results.append(win_dict)

        model_seconds = time.perf_counter() - inference_started
        window_seconds = inference_started - decoded_at
        # Recording-level aggregation
        rec_score = self.aggregator.aggregate(fake_window_scores)
        if orientation == "higher_is_bonafide":
            bonafide_rec = float(np.mean(bonafide_window_scores)) if bonafide_window_scores else 0.0
            raw_rec = bonafide_rec
        else:
            raw_rec = rec_score

        # Recording calibration must be fitted on scores produced by this exact
        # windowing/VAD/aggregation pipeline. It does not calibrate window flags.
        if self.calibrator is not None and self.calibrator.fitted:
            calibrated_prob = float(self.calibrator.predict_proba(raw_rec))
            calibration_version = self.calibrator.version
            lower, upper = self.uncertain_band or self.calibrator.uncertain_band or (None, None)
            if lower is not None and lower <= calibrated_prob <= upper:
                verdict = "uncertain"
            else:
                verdict = "fake" if calibrated_prob >= self.calibrator.operating_threshold else "real"
        else:
            calibration_version = None
            if orientation == "higher_is_bonafide":
                is_fake_verdict = raw_rec < self.threshold
                calibrated_prob = None  # Do not invent uncalibrated probabilities for logits
            else:
                is_fake_verdict = raw_rec >= self.threshold
                calibrated_prob = None
            verdict = "fake" if is_fake_verdict else "real"

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
            "verdict": verdict,
            "model_id": self.meta.get("model_id"),
            "model_version": self.meta.get("checkpoint_sha256", "unknown")[:10],
            "raw_recording_score": round(raw_rec, 6),
            "calibrated_fake_probability": round(calibrated_prob, 6) if calibrated_prob is not None else None,
            "calibration_version": calibration_version,
            "threshold": (
                self.calibrator.operating_threshold
                if self.calibrator is not None and self.calibrator.fitted
                else self.threshold
            ),
            "model_threshold": self.threshold,
            "window_threshold": self.threshold,
            "pipeline_signature": self.pipeline_signature(),
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
            "timings": {
                "decode_seconds": round(decode_seconds, 6),
                "window_seconds": round(window_seconds, 6),
                "model_seconds": round(model_seconds, 6),
            },
            "warnings": warnings,
        }
