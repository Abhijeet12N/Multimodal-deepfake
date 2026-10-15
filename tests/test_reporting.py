"""Unit tests for reporting and risk scoring."""
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.reporting import build_report, risk_level


class TestReporting(unittest.TestCase):
    """Test suite for human-readable report formatting and risk evaluation."""

    def test_risk_level_low_when_any_below_threshold(self):
        self.assertEqual(risk_level({"video": 0.9, "audio": 0.1, "any": 0.2}), "low")

    def test_risk_level_medium_when_any_above_threshold_but_not_high(self):
        self.assertEqual(risk_level({"video": 0.6, "audio": 0.55, "any": 0.65}), "medium")

    def test_risk_level_high_when_signal_is_strong(self):
        self.assertEqual(risk_level({"video": 0.2, "audio": 0.91, "any": 0.92}), "high")
        self.assertEqual(risk_level({"video": 0.88, "audio": 0.1, "any": 0.55}), "high")

    def test_build_report_structure(self):
        report = build_report(
            scores={"video": 0.2, "audio": 0.8, "any": 0.85},
            decision={"label": "FAKE (audio)", "sub": "Synthetic voice detected.", "severity": "danger"},
            info={"face_prob": 0.95, "audio_rms": 0.03, "preprocess_s": 1.2, "inference_s": 0.3, "total_s": 1.5},
            ground_truth={"category": "R-F", "video_fake": False, "audio_fake": True, "any_fake": True},
            threshold=0.5,
        )

        self.assertEqual(report["risk"], "high")
        self.assertEqual(report["scores"]["audio"], 0.8)
        self.assertEqual(report["scores"]["video"], 0.2)
        self.assertEqual(report["scores"]["any"], 0.85)
        self.assertEqual(report["timing"]["total_s"], 1.5)
        self.assertEqual(report["ground_truth"]["category"], "R-F")
        self.assertTrue(any("audio stream" in f for f in report["findings"]))
        self.assertTrue(any("Face detection confidence" in f for f in report["findings"]))
        self.assertTrue(any("Audio RMS" in f for f in report["findings"]))

    def test_build_report_all_real(self):
        report = build_report(
            scores={"video": 0.05, "audio": 0.12, "any": 0.08},
            decision={"label": "REAL", "sub": "Media is authentic.", "severity": "safe"},
            info={},
            threshold=0.5,
        )
        self.assertEqual(report["risk"], "low")
        self.assertTrue(any("stayed below" in f for f in report["findings"]))


if __name__ == "__main__":
    unittest.main()
