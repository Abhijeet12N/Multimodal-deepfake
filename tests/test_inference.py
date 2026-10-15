"""Unit tests for inference and verdict interpretation logic."""
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.inference import (
    DeepfakePredictor,
    category_from_path,
    interpret,
)
from deepfake_detector.audio_inference import StandaloneAudioPredictor


class TestInference(unittest.TestCase):
    """Test suite for verdict decisions, label parsing, and predictor guards."""

    def test_interpret_real(self):
        res = interpret({"video": 0.1, "audio": 0.2, "any": 0.3})
        self.assertEqual(res["label"], "REAL")
        self.assertEqual(res["severity"], "safe")

    def test_interpret_fake_both(self):
        res = interpret({"video": 0.9, "audio": 0.85, "any": 0.95})
        self.assertEqual(res["label"], "FAKE (both)")
        self.assertEqual(res["severity"], "danger")

    def test_interpret_fake_video_only(self):
        res = interpret({"video": 0.9, "audio": 0.2, "any": 0.85})
        self.assertEqual(res["label"], "FAKE (video)")
        self.assertEqual(res["severity"], "danger")

    def test_interpret_fake_audio_only(self):
        res = interpret({"video": 0.1, "audio": 0.92, "any": 0.88})
        self.assertEqual(res["label"], "FAKE (audio)")
        self.assertEqual(res["severity"], "danger")

    def test_interpret_fake_fusion_only(self):
        res = interpret({"video": 0.45, "audio": 0.48, "any": 0.75})
        self.assertEqual(res["label"], "FAKE (combined score)")
        self.assertEqual(res["severity"], "danger")

    def test_category_from_path_known(self):
        self.assertEqual(
            category_from_path("data/FakeAVCeleb/RealVideo-RealAudio/id0001/001.mp4"),
            ("R-R", "real", "real", False),
        )
        self.assertEqual(
            category_from_path("data\\FakeAVCeleb\\FakeVideo-RealAudio\\id0001\\001.mp4"),
            ("F-R", "fake", "real", True),
        )
        self.assertEqual(
            category_from_path("data/FakeAVCeleb/RealVideo-FakeAudio/id0001/001.mp4"),
            ("R-F", "real", "fake", True),
        )
        self.assertEqual(
            category_from_path("data/FakeAVCeleb/FakeVideo-FakeAudio/id0001/001.mp4"),
            ("F-F", "fake", "fake", True),
        )

    def test_category_from_path_unknown(self):
        self.assertEqual(
            category_from_path("user_downloads/random_video.mp4"),
            (None, None, None, None),
        )

    def test_predictor_missing_checkpoint_raises(self):
        with self.assertRaises(FileNotFoundError):
            DeepfakePredictor("nonexistent_model_checkpoint.pt")

        with self.assertRaises(FileNotFoundError):
            StandaloneAudioPredictor("nonexistent_audio_checkpoint.pt")


if __name__ == "__main__":
    unittest.main()

