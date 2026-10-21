"""Upstream Parity and Verification Tests for AASIST and AASIST-L.

Verifies:
1. Checkpoint integrity and SHA256 hashes against official releases.
2. Exact numerical parity between upstream models.AASIST.Model and AASISTDetector adapter (< 1e-6 max absolute difference).
3. Score semantics, orientation (higher_is_bonafide), and class index mapping (0: spoof, 1: bonafide).
4. Waveform preprocessing and periodic repeat tiling parity with upstream data_utils.pad.
5. End-to-end inference via StandaloneAudioService on real test audio.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

AASIST_DIR = ROOT.parent / "aasist"
if AASIST_DIR.is_dir() and str(AASIST_DIR) not in sys.path:
    sys.path.insert(0, str(AASIST_DIR))

from deepfake_detector.audio_detection.detectors.aasist import AASISTDetector
from deepfake_detector.audio_detection.service import StandaloneAudioService
from deepfake_detector.audio_detection.registry import get_detector


EXPECTED_AASIST_SHA256 = "51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0"
EXPECTED_AASIST_L_SHA256 = "814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a"

AASIST_CKPT = AASIST_DIR / "models/weights/AASIST.pth"
AASIST_L_CKPT = AASIST_DIR / "models/weights/AASIST-L.pth"
AASIST_CONF = AASIST_DIR / "config/AASIST.conf"
AASIST_L_CONF = AASIST_DIR / "config/AASIST-L.conf"


@pytest.mark.skipif(not AASIST_CKPT.is_file(), reason="AASIST checkpoint not found")
class TestAASISTUpstreamParity:
    """Test suite verifying AASIST adapter against official upstream implementation."""

    def test_checkpoint_hashes(self):
        """Verify that checkpoint SHA-256 hashes match official releases."""
        with open(AASIST_CKPT, "rb") as f:
            h_aasist = hashlib.sha256(f.read()).hexdigest()
        assert h_aasist.lower() == EXPECTED_AASIST_SHA256.lower()

        if AASIST_L_CKPT.is_file():
            with open(AASIST_L_CKPT, "rb") as f:
                h_l = hashlib.sha256(f.read()).hexdigest()
            assert h_l.lower() == EXPECTED_AASIST_L_SHA256.lower()

    def test_aasist_upstream_numerical_parity(self):
        """Verify exact numerical agreement between upstream Model and AASISTDetector."""
        from models.AASIST import Model

        with open(AASIST_CONF, "r") as f:
            cfg = json.load(f)["model_config"]

        # Upstream model
        upstream_model = Model(cfg).to("cpu")
        st = torch.load(str(AASIST_CKPT), map_location="cpu", weights_only=False)
        upstream_model.load_state_dict(st)
        upstream_model.eval()

        # Adapter detector
        detector = AASISTDetector(checkpoint_path=AASIST_CKPT, device="cpu")

        # Deterministic random input batch (2 samples, 64600 length)
        torch.manual_seed(12345)
        test_inputs = torch.randn(2, 64600)

        with torch.no_grad():
            _, upstream_logits = upstream_model(test_inputs)
            adapter_res = detector.predict_batch(test_inputs)

        adapter_logits = adapter_res["raw_scores"]
        upstream_np = upstream_logits.numpy()

        max_abs_diff = float(np.abs(adapter_logits - upstream_np).max())
        assert max_abs_diff < 1e-6, f"Parity mismatch: max diff {max_abs_diff:.2e} >= 1e-6"

    @pytest.mark.skipif(not AASIST_L_CKPT.is_file(), reason="AASIST-L checkpoint not found")
    def test_aasist_l_upstream_numerical_parity(self):
        """Verify numerical agreement for the lightweight AASIST-L variant."""
        from models.AASIST import Model

        with open(AASIST_L_CONF, "r") as f:
            cfg = json.load(f)["model_config"]

        upstream_model = Model(cfg).to("cpu")
        st = torch.load(str(AASIST_L_CKPT), map_location="cpu", weights_only=False)
        upstream_model.load_state_dict(st)
        upstream_model.eval()

        detector = AASISTDetector(checkpoint_path=AASIST_L_CKPT, is_lightweight=True, device="cpu")

        torch.manual_seed(54321)
        test_inputs = torch.randn(2, 64600)

        with torch.no_grad():
            _, upstream_logits = upstream_model(test_inputs)
            adapter_res = detector.predict_batch(test_inputs)

        adapter_logits = adapter_res["raw_scores"]
        upstream_np = upstream_logits.numpy()

        max_abs_diff = float(np.abs(adapter_logits - upstream_np).max())
        assert max_abs_diff < 1e-6, f"AASIST-L parity mismatch: max diff {max_abs_diff:.2e} >= 1e-6"

    def test_score_semantics_and_direction(self):
        """Verify class indices, score orientation, and logit relations."""
        detector = AASISTDetector(checkpoint_path=AASIST_CKPT, device="cpu")
        reqs = detector.requirements()

        assert reqs["score_orientation"] == "higher_is_bonafide"
        assert reqs["class_index_mapping"] == {0: "spoof", 1: "bonafide"}
        assert reqs["input_length_samples"] == 64600
        assert reqs["padding_mode"] == "repeat"

        test_inputs = torch.zeros(3, 64600)
        res = detector.predict_batch(test_inputs)

        # Class 1 is bonafide score
        np.testing.assert_allclose(res["bonafide_scores"], res["raw_scores"][:, 1], atol=1e-6)
        # Fake score is (spoof - bonafide)
        expected_fake = res["raw_scores"][:, 0] - res["raw_scores"][:, 1]
        np.testing.assert_allclose(res["fake_scores"], expected_fake, atol=1e-6)

    def test_waveform_periodic_repeat_padding(self):
        """Verify pad_waveform matches upstream data_utils.pad periodic tiling."""
        import data_utils

        # Case 1: Short audio (< 64600)
        short_audio = np.linspace(-0.5, 0.5, 12000, dtype=np.float32)
        upstream_padded = data_utils.pad(short_audio, max_len=64600)

        tensor_in = torch.from_numpy(short_audio).unsqueeze(0)
        adapter_padded = AASISTDetector.pad_waveform(tensor_in, max_len=64600).squeeze(0).numpy()

        assert adapter_padded.shape == (64600,)
        np.testing.assert_allclose(adapter_padded, upstream_padded, atol=1e-6)

        # Case 2: Long audio (> 64600)
        long_audio = np.random.randn(80000).astype(np.float32)
        upstream_trunc = data_utils.pad(long_audio, max_len=64600)
        tensor_long = torch.from_numpy(long_audio).unsqueeze(0)
        adapter_trunc = AASISTDetector.pad_waveform(tensor_long, max_len=64600).squeeze(0).numpy()

        assert adapter_trunc.shape == (64600,)
        np.testing.assert_allclose(adapter_trunc, upstream_trunc, atol=1e-6)

    def test_registry_instantiation(self):
        """Verify get_detector correctly resolves and loads AASIST and AASIST-L."""
        det_aasist = get_detector("aasist", checkpoint_path=str(AASIST_CKPT), device="cpu")
        assert isinstance(det_aasist, AASISTDetector)
        assert not det_aasist.is_lightweight

        if AASIST_L_CKPT.is_file():
            det_l = get_detector("aasist_l", checkpoint_path=str(AASIST_L_CKPT), device="cpu")
            assert isinstance(det_l, AASISTDetector)
            assert det_l.is_lightweight

    def test_service_prediction_on_real_audio(self):
        """Verify StandaloneAudioService executes cleanly with AASIST on real audio file."""
        detector = AASISTDetector(checkpoint_path=AASIST_CKPT, device="cpu")
        service = StandaloneAudioService(detector, batch_size=2, vad_enabled=True)

        sample_audio = ROOT / "ui/samples"
        # Find any audio file
        audio_files = list(ROOT.glob("**/*.mp3")) + list(ROOT.glob("**/*.wav"))
        if not audio_files:
            parent_audios = list((ROOT.parent / "initial setup/deepfake_phase1/Audios").glob("**/*.*"))
            audio_files = [p for p in parent_audios if p.suffix.lower() in (".mp3", ".wav")]

        assert len(audio_files) > 0, "No test audio files found"
        test_file = audio_files[0]

        result = service.predict(test_file)

        assert result["status"] == "ok"
        assert result["model_id"] == "aasist"
        assert result["score_orientation"] == "higher_is_bonafide"
        assert result["raw_recording_score"] is not None
        assert isinstance(result["windows"], list)
        assert len(result["windows"]) > 0
        assert "is_speech" in result["windows"][0]
        assert "bonafide_score" in result["windows"][0]
