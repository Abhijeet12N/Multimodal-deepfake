"""Regression checks for training parity, decoding and recording calibration."""
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import Wav2Vec2FeatureExtractor

from deepfake_detector.audio_detection.calibration import ScoreCalibrator
from deepfake_detector.audio_detection.detectors.legacy_wav2vec import LegacyWav2VecDetector
from deepfake_detector.audio_detection.preprocessing import load_audio
from deepfake_detector.audio_detection.service import StandaloneAudioService


class CapturingCNN(torch.nn.Module):
    def forward(self, waveform):
        self.waveform = waveform.clone()
        return waveform[:, None, :]


class SmallClassifier(torch.nn.Module):
    def forward(self, embeddings):
        return torch.cat([-embeddings, embeddings], dim=1)


@pytest.mark.parametrize("length", [8000, 32000, 40000])
def test_phase1_cnn_receives_exact_training_processor_output(length):
    detector = LegacyWav2VecDetector()
    detector.processor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=False,
    )
    cnn = CapturingCNN()
    detector.backbone = SimpleNamespace(feature_extractor=cnn)
    detector.classifier = SmallClassifier()
    rng = np.random.default_rng(42)
    # Low amplitude and DC offsets expose the missing-normalization regression.
    waves = rng.normal(0.2, 0.001, (2, length)).astype(np.float32)
    fixed = np.pad(waves, ((0, 0), (0, max(0, 32000 - length))))[:, :32000]
    expected = detector.processor(list(fixed), sampling_rate=16000,
                                  return_tensors="pt", padding=True).input_values
    detector.predict_batch(waves)
    torch.testing.assert_close(cnn.waveform, expected, rtol=0, atol=0)
    assert not np.allclose(fixed, expected.numpy())


def test_decode_resampling_does_not_enter_librosa_jit(tmp_path, monkeypatch):
    path = tmp_path / "stereo.wav"
    wave = np.sin(2 * np.pi * 440 * np.arange(4410) / 44100).astype(np.float32)
    sf.write(path, np.stack([wave, wave], axis=1), 44100, subtype="FLOAT")
    # Accessing librosa.load would trigger its lazy imports; soundfile should
    # handle this input without touching librosa at all.
    from deepfake_detector.audio_detection import preprocessing
    monkeypatch.setattr(preprocessing, "librosa", SimpleNamespace())
    decoded, duration = load_audio(path)
    assert decoded.shape == (1600,)
    assert decoded.dtype == np.float32
    assert duration == pytest.approx(0.1)


class SequenceDetector:
    def metadata(self):
        return {"model_id": "test", "checkpoint_sha256": "test-checkpoint"}

    def requirements(self):
        return {"sample_rate": 16000, "input_length_samples": 32000,
                "hop_samples": 32000, "padding_mode": "zero",
                "score_orientation": "higher_is_fake", "calibrated_threshold": 0.8}

    def predict_batch(self, waveforms, lengths=None):
        return {"fake_probabilities": np.array([0.1, 0.9]),
                "score_orientation": "higher_is_fake"}


def test_calibration_applies_to_recording_aggregate_and_preserves_window_flags(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.full(64000, 0.1, dtype=np.float32), 16000)
    cal = ScoreCalibrator(model_id="test")
    cal.fitted, cal.params = True, {"a": 8.0, "b": -3.0}
    service = StandaloneAudioService(SequenceDetector(), calibrator=cal)
    result = service.predict(path)
    expected = cal.predict_proba(0.5)
    assert result["calibrated_fake_probability"] == pytest.approx(expected, abs=1e-6)
    assert abs(expected - np.mean(cal.predict_proba([0.1, 0.9]))) > 0.05
    assert [w["is_fake"] for w in result["windows"]] == [False, True]
    assert result["window_threshold"] == 0.8
    assert result["threshold"] == 0.5


def test_studio_rejects_native_calibration_and_changed_pipeline():
    service = StandaloneAudioService(SequenceDetector())
    cal = ScoreCalibrator(model_id="test")
    cal.fitted, cal.params = True, {"a": 1.0, "b": 0.0}
    cal.fit_data = {"split": "dev", "sample_count": 24844, "protocol": "native_benchmark"}
    assert not service.accepts_calibrator(cal)
    cal.fit_data.update(protocol="studio_sliding_window", score_level="recording",
                        pipeline_signature=service.pipeline_signature())
    assert service.accepts_calibrator(cal)
    cal.fit_data["pipeline_signature"] = "different-weights-or-preprocessing"
    assert not service.accepts_calibrator(cal)


def test_uncalibrated_probability_is_not_presented_as_calibrated(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, np.full(64000, 0.1, dtype=np.float32), 16000)
    result = StandaloneAudioService(SequenceDetector()).predict(path)
    assert result["calibrated_fake_probability"] is None
    assert result["calibration_version"] is None
