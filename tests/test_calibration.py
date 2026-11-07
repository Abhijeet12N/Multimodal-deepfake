"""Unit tests for ScoreCalibrator, ECE calculation, and uncertainty bands."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from deepfake_detector.audio_detection.calibration import (
    ScoreCalibrator,
    compute_ece,
    compute_reliability_curve,
)


class TestECEComputation:
    def test_perfect_calibration_low_ece(self):
        probs = np.array([0.1, 0.1, 0.9, 0.9])
        labels = np.array([0, 0, 1, 1])
        ece = compute_ece(probs, labels, n_bins=10)
        assert ece < 0.15

    def test_empty_inputs(self):
        assert compute_ece(np.array([]), np.array([])) == 0.0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_ece(np.array([0.5]), np.array([0, 1]))

    def test_reliability_curve_output_schema(self):
        probs = np.linspace(0.1, 0.9, 20)
        labels = np.array([0] * 10 + [1] * 10)
        res = compute_reliability_curve(probs, labels, n_bins=5)
        assert "bin_centers" in res
        assert "bin_accuracies" in res
        assert "bin_confidences" in res
        assert "bin_counts" in res
        assert "ece" in res
        assert len(res["bin_centers"]) == 5


class TestScoreCalibrator:
    @pytest.fixture
    def synthetic_data(self):
        np.random.seed(42)
        n = 100
        # Bonafide scores: higher is bonafide (mean +2.0)
        bonafide_scores = np.random.normal(loc=2.0, scale=1.0, size=n // 2)
        # Fake scores: lower bonafide scores (mean -2.0)
        fake_scores = np.random.normal(loc=-2.0, scale=1.0, size=n // 2)

        scores = np.concatenate([bonafide_scores, fake_scores])
        labels = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
        return scores, labels

    def test_platt_scaling_fit_and_predict(self, synthetic_data):
        scores, labels = synthetic_data
        cal = ScoreCalibrator(
            method="platt",
            score_orientation="higher_is_bonafide",
            model_id="test_model",
        )
        assert not cal.fitted
        with pytest.raises(RuntimeError):
            cal.predict_proba(0.0)

        cal.fit(scores, labels)
        assert cal.fitted
        assert "a" in cal.params
        assert "b" in cal.params

        # Lower bonafide score should produce higher fake probability
        fake_prob = cal.predict_proba(-3.0)
        real_prob = cal.predict_proba(3.0)
        assert fake_prob > real_prob
        assert 0.0 <= fake_prob <= 1.0
        assert 0.0 <= real_prob <= 1.0

    def test_isotonic_scaling_fit_and_predict(self, synthetic_data):
        scores, labels = synthetic_data
        cal = ScoreCalibrator(
            method="isotonic",
            score_orientation="higher_is_bonafide",
        )
        cal.fit(scores, labels)
        assert cal.fitted
        assert "x_thresholds" in cal.params

        p_fake = cal.predict_proba(-4.0)
        p_real = cal.predict_proba(4.0)
        assert p_fake >= p_real

    def test_temperature_scaling_fit(self, synthetic_data):
        scores, labels = synthetic_data
        cal = ScoreCalibrator(
            method="temperature",
            score_orientation="higher_is_bonafide",
        )
        cal.fit(scores, labels)
        assert cal.fitted
        assert "temperature" in cal.params
        assert cal.params["temperature"] > 0

    def test_uncertainty_verdict(self):
        cal = ScoreCalibrator(
            method="platt",
            operating_threshold=0.5,
            uncertain_band=(0.40, 0.60),
        )
        # Mock fitted params: a=1.0, b=0.0 -> score 0.0 gives prob 0.5
        cal.fitted = True
        cal.params = {"a": 1.0, "b": 0.0}

        # score=0.0 -> prob=0.5 -> within [0.4, 0.6] -> uncertain
        assert cal.predict_verdict(0.0) == "uncertain"
        # score=5.0 -> prob ~ 0.993 > 0.60 -> fake
        assert cal.predict_verdict(5.0) == "fake"
        # score=-5.0 -> prob ~ 0.007 < 0.40 -> real
        assert cal.predict_verdict(-5.0) == "real"

    def test_save_and_load_roundtrip(self, synthetic_data, tmp_path):
        scores, labels = synthetic_data
        cal = ScoreCalibrator(
            method="platt",
            model_id="aasist_v1",
            version="2.1.0",
            score_orientation="higher_is_bonafide",
            operating_threshold=0.55,
            uncertain_band=(0.42, 0.58),
        )
        cal.fit(scores, labels)

        save_path = tmp_path / "calibration.json"
        cal.save(save_path)
        assert save_path.is_file()

        loaded_cal = ScoreCalibrator.load(save_path)
        assert loaded_cal.model_id == "aasist_v1"
        assert loaded_cal.version == "2.1.0"
        assert loaded_cal.method == "platt"
        assert loaded_cal.score_orientation == "higher_is_bonafide"
        assert loaded_cal.operating_threshold == 0.55
        assert loaded_cal.uncertain_band == (0.42, 0.58)

        # Check identical probability outputs
        test_inputs = np.array([-3.0, 0.0, 2.5])
        orig_preds = cal.predict_proba(test_inputs)
        loaded_preds = loaded_cal.predict_proba(test_inputs)
        np.testing.assert_allclose(orig_preds, loaded_preds, atol=1e-6)
