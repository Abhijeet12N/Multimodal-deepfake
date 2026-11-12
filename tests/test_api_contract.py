"""API schema and contract tests for audio detection interfaces and services.

Verifies that all registered detectors and StandaloneAudioService adhere strictly
to the required request/response schemas specified in the Model Upgradation Plan.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from deepfake_detector.audio_detection import (
    AASISTDetector,
    AudioDetector,
    LegacyWav2VecDetector,
    SSLAASISTDetector,
    StandaloneAudioService,
    XLSRMamBoDetector,
    get_detector,
    list_detectors,
)
from deepfake_detector.audio_detection.calibration import ScoreCalibrator

REQUIRED_PREDICT_SCHEMA_KEYS = {
    "status",
    "verdict",
    "model_id",
    "model_version",
    "raw_recording_score",
    "calibrated_fake_probability",
    "calibration_version",
    "threshold",
    "model_threshold",
    "window_threshold",
    "pipeline_signature",
    "timings",
    "score_orientation",
    "duration_seconds",
    "speech_duration_seconds",
    "analysis_window_seconds",
    "hop_seconds",
    "total_windows",
    "evaluated_windows",
    "suspicious_intervals",
    "top_windows",
    "windows",
    "inference_seconds",
    "warnings",
}

REQUIRED_REQUIREMENTS_KEYS = {
    "sample_rate",
    "input_length_samples",
    "score_orientation",
}


class MockDetector(AudioDetector):
    """Mock detector to test service API contracts without external weights."""

    def __init__(self, score_orientation: str = "higher_is_fake", fake_val: float = 0.8):
        self._orientation = score_orientation
        self._fake_val = fake_val

    def load(self, checkpoint_path: str, device: str = "cpu") -> None:
        pass

    def requirements(self) -> dict:
        return {
            "sample_rate": 16000,
            "input_length_samples": 32000,
            "input_length_seconds": 2.0,
            "hop_samples": 16000,
            "hop_seconds": 1.0,
            "padding_mode": "zero",
            "score_orientation": self._orientation,
            "calibrated_threshold": 0.5,
            "min_rms_vad": 0.01,
        }

    def predict_batch(self, waveforms: np.ndarray, lengths: list[int] | None = None) -> dict:
        b = len(waveforms)
        if self._orientation == "higher_is_bonafide":
            bonafide = np.full(b, -self._fake_val, dtype=np.float32)
            fake = np.full(b, self._fake_val, dtype=np.float32)
            return {
                "bonafide_scores": bonafide,
                "fake_scores": fake,
                "score_orientation": "higher_is_bonafide",
            }
        else:
            raw = np.full(b, self._fake_val, dtype=np.float32)
            return {
                "raw_scores": raw,
                "fake_probabilities": raw,
                "score_orientation": "higher_is_fake",
            }

    def metadata(self) -> dict:
        return {
            "model_id": "mock_detector",
            "architecture": "mock",
            "score_orientation": self._orientation,
            "checkpoint_sha256": "mock_hash_1234567890",
        }


class TestDetectorInterfaceContract:
    def test_all_registered_detectors_implement_interface(self):
        registered = list_detectors()
        assert "legacy_wav2vec_mlp" in registered
        assert "aasist" in registered
        assert "ssl_aasist" in registered
        assert "xlsr_mambo" in registered

    def test_mock_detector_requirements_schema(self):
        det = MockDetector()
        reqs = det.requirements()
        assert REQUIRED_REQUIREMENTS_KEYS.issubset(set(reqs.keys()))
        assert reqs["sample_rate"] == 16000
        assert reqs["score_orientation"] in ("higher_is_fake", "higher_is_bonafide")

    def test_mock_detector_predict_batch_schema(self):
        det = MockDetector()
        waveforms = np.zeros((2, 32000), dtype=np.float32)
        res = det.predict_batch(waveforms)
        assert isinstance(res, dict)
        assert "score_orientation" in res


class TestServiceAPIContract:
    @pytest.fixture
    def test_audio_file(self, tmp_path):
        audio_path = tmp_path / "contract_test.wav"
        t = np.linspace(0, 4.0, 16000 * 4)
        samples = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        sf.write(str(audio_path), samples, 16000)
        return audio_path

    def test_successful_predict_contract_schema(self, test_audio_file):
        detector = MockDetector(score_orientation="higher_is_fake", fake_val=0.75)
        service = StandaloneAudioService(detector, batch_size=2, vad_enabled=False)

        res = service.predict(test_audio_file)

        assert set(res.keys()) == REQUIRED_PREDICT_SCHEMA_KEYS
        assert res["status"] == "ok"
        assert res["verdict"] in ("fake", "real", "uncertain")
        assert res["model_id"] == "mock_detector"
        assert res["total_windows"] > 0
        assert isinstance(res["suspicious_intervals"], list)
        assert isinstance(res["warnings"], list)
        assert isinstance(res["windows"], list)

    def test_error_on_missing_file(self):
        detector = MockDetector()
        service = StandaloneAudioService(detector)
        with pytest.raises(FileNotFoundError):
            service.predict("non_existent_audio_file.wav")

    def test_uncalibrated_bonafide_contract(self, test_audio_file):
        """When detector uses raw logits (bonafide), calibrated_fake_probability must remain None unless calibrated."""
        detector = MockDetector(score_orientation="higher_is_bonafide", fake_val=1.2)
        service = StandaloneAudioService(detector, batch_size=2, vad_enabled=False)

        res = service.predict(test_audio_file)
        assert res["score_orientation"] == "higher_is_bonafide"
        assert res["calibrated_fake_probability"] is None
        assert res["calibration_version"] is None
        assert res["raw_recording_score"] is not None

    def test_calibrated_uncertain_contract(self, test_audio_file):
        """When calibrator with uncertain region is provided, verdict must become 'uncertain' when probability falls in band."""
        detector = MockDetector(score_orientation="higher_is_fake", fake_val=0.5)

        cal = ScoreCalibrator(
            method="platt",
            operating_threshold=0.5,
            uncertain_band=(0.4, 0.6),
            score_orientation="higher_is_fake",
        )
        # Mock fit: a=1.0, b=0.0 -> prob = 1 / (1 + exp(-0.5)) = 0.622, or let's fit synthetic
        scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        labels = np.array([0, 0, 0, 1, 1])
        cal.fit(scores, labels)

        service = StandaloneAudioService(
            detector,
            calibrator=cal,
            uncertain_band=(0.35, 0.65),
            vad_enabled=False,
        )

        res = service.predict(test_audio_file)
        assert res["calibration_version"] == "1.0.0"
        assert res["calibrated_fake_probability"] is not None
        assert 0.0 <= res["calibrated_fake_probability"] <= 1.0
        assert res["verdict"] in ("fake", "real", "uncertain")
