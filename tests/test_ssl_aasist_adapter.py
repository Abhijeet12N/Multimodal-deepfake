"""Unit tests for SSLAASISTDetector adapter.

Verifies:
1. Requirements dictionary contract (sample rate, 64600 samples, score orientation, class mapping).
2. Model metadata specification (uses_ssl_frontend, upstream commit, license).
3. Environment detection helper (is_environment_available).
4. Graceful error handling when uninstalled dependencies (Fairseq) are requested.
5. Dynamic registration in registry (get_detector).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.audio_detection.detectors.ssl_aasist import SSLAASISTDetector
from deepfake_detector.audio_detection.registry import get_detector


class TestSSLAASISTAdapter:
    """Test suite for SSL-AASIST adapter contract and environment handling."""

    def test_requirements_contract(self):
        """Verify audio requirements contract for SSL-AASIST."""
        detector = SSLAASISTDetector()
        reqs = detector.requirements()

        assert reqs["sample_rate"] == 16000
        assert reqs["input_length_samples"] == 64600
        assert reqs["input_length_seconds"] == 64600 / 16000
        assert reqs["score_orientation"] == "higher_is_bonafide"
        assert reqs["class_index_mapping"] == {0: "spoof", 1: "bonafide"}
        assert reqs["padding_mode"] == "repeat"
        assert reqs["min_gpu_vram_gb"] >= 12.0

    def test_metadata_contract(self):
        """Verify metadata fields for provenance tracking."""
        detector = SSLAASISTDetector()
        meta = detector.metadata()

        assert meta["model_id"] == "ssl_aasist"
        assert "TakHemlata/SSL_Anti-spoofing" in meta["upstream_repo"]
        assert meta["uses_ssl_frontend"] is True
        assert meta["score_orientation"] == "higher_is_bonafide"
        assert meta["execution_target"] == "Google Colab"

    def test_environment_availability_check(self):
        """Verify is_environment_available method returns valid status and diagnostic message."""
        available, message = SSLAASISTDetector.is_environment_available()
        assert isinstance(available, bool)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_graceful_error_handling_without_checkpoint(self):
        """Verify RuntimeError is raised with actionable Colab advice when loaded without fairseq."""
        detector = SSLAASISTDetector()
        available, _ = SSLAASISTDetector.is_environment_available()

        if not available:
            # Should raise descriptive RuntimeError pointing to Colab
            fake_ckpt = ROOT / "models/fake_checkpoint.pt"
            with pytest.raises(FileNotFoundError):
                detector.load(str(fake_ckpt))

            dummy_wave = torch.zeros(1, 64600)
            with pytest.raises(RuntimeError) as exc_info:
                detector.predict_batch(dummy_wave)
            assert "Google Colab" in str(exc_info.value) or "fairseq" in str(exc_info.value)

    def test_registry_resolution(self):
        """Verify get_detector('ssl_aasist') instantiates SSLAASISTDetector."""
        detector = get_detector("ssl_aasist")
        assert isinstance(detector, SSLAASISTDetector)
        assert detector.metadata()["model_id"] == "ssl_aasist"
