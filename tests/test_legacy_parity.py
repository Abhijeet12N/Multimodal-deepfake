"""Parity test ensuring wrapped LegacyWav2VecDetector reproduces reference predictions."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.audio_detection import (
    LegacyWav2VecDetector,
    StandaloneAudioService,
)


class TestLegacyParity(unittest.TestCase):
    """Test suite ensuring wrapped legacy model maintains strict numerical parity."""

    def setUp(self):
        self.fixture_path = ROOT / "tests" / "fixtures" / "legacy_prediction_reference.json"
        if not self.fixture_path.exists():
            self.skipTest(f"Reference fixture missing: {self.fixture_path}")

        with open(self.fixture_path, "r", encoding="utf-8") as f:
            self.reference = json.load(f)

        self.audio_path = Path(self.reference["input"]["filepath"])
        self.ckpt_path = Path(self.reference["checkpoint"]["filepath"])

        if not self.audio_path.exists() or not self.ckpt_path.exists():
            self.skipTest("Audio file or checkpoint file not available on local filesystem.")

        self.detector = LegacyWav2VecDetector(str(self.ckpt_path), device="cpu")
        self.service = StandaloneAudioService(self.detector, batch_size=8, vad_enabled=True)

    def test_reproduction_verdict_parity(self):
        res = self.service.predict(self.audio_path)
        self.assertEqual(res["verdict"].upper(), self.reference["decision"]["verdict"])

    def test_reproduction_score_parity(self):
        res = self.service.predict(self.audio_path)
        ref_score = float(self.reference["decision"]["recording_spoof_probability"])
        service_score = float(res["raw_recording_score"])
        # Tolerance: 1e-4
        self.assertAlmostEqual(service_score, ref_score, places=4)

    def test_reproduction_window_count_parity(self):
        res = self.service.predict(self.audio_path)
        self.assertEqual(res["total_windows"], self.reference["decision"]["window_count"])

    def test_score_orientation_metadata(self):
        res = self.service.predict(self.audio_path)
        self.assertEqual(res["score_orientation"], "higher_is_fake")


if __name__ == "__main__":
    unittest.main()
