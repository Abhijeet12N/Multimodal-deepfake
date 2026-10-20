"""Unit tests verifying score orientation and class index mappings."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from deepfake_detector.audio_detection.detectors.aasist import AASISTDetector
from deepfake_detector.audio_detection.detectors.legacy_wav2vec import LegacyWav2VecDetector


class TestScoreDirection(unittest.TestCase):
    """Ensure score semantics and orientations are explicitly tracked and verified."""

    def test_aasist_score_semantics_and_mapping(self):
        detector = AASISTDetector()
        reqs = detector.requirements()

        # Class 0: spoof, Class 1: bonafide
        self.assertEqual(reqs["class_index_mapping"], {0: "spoof", 1: "bonafide"})
        # Upstream AASIST evaluates index 1 (bonafide score)
        self.assertEqual(reqs["score_orientation"], "higher_is_bonafide")
        self.assertEqual(reqs["raw_score_semantics"], "two_class_logits")
        # Native length: 64600 samples (~4.04s)
        self.assertEqual(reqs["input_length_samples"], 64600)
        self.assertEqual(reqs["padding_mode"], "repeat")

    def test_legacy_score_semantics_and_mapping(self):
        detector = LegacyWav2VecDetector()
        reqs = detector.requirements()

        # Class 0: bonafide, Class 1: spoof
        self.assertEqual(reqs["class_index_mapping"], {0: "bonafide", 1: "spoof"})
        self.assertEqual(reqs["score_orientation"], "higher_is_fake")
        self.assertEqual(reqs["raw_score_semantics"], "two_class_logits")
        # Native length: 32000 samples (2.0s)
        self.assertEqual(reqs["input_length_samples"], 32000)
        self.assertEqual(reqs["padding_mode"], "zero")

    def test_aasist_prediction_score_direction_on_noise(self):
        ckpt_path = Path(__file__).resolve().parents[2] / "aasist" / "models" / "weights" / "AASIST.pth"
        if not ckpt_path.exists():
            self.skipTest("AASIST.pth checkpoint not available.")

        detector = AASISTDetector(str(ckpt_path), device="cpu")
        noise = torch.randn(2, 64600)
        res = detector.predict_batch(noise)

        # For noise, bonafide logit should be significantly lower than spoof logit
        for i in range(2):
            spoof_logit = res["raw_scores"][i][0]
            bonafide_logit = res["raw_scores"][i][1]
            self.assertGreater(spoof_logit, bonafide_logit)
            # fake_score is spoof - bonafide, so it must be positive for noise
            self.assertGreater(res["fake_scores"][i], 0.0)

    def test_raw_scores_are_not_restricted_to_probabilities(self):
        ckpt_path = Path(__file__).resolve().parents[2] / "aasist" / "models" / "weights" / "AASIST.pth"
        if not ckpt_path.exists():
            self.skipTest("AASIST.pth checkpoint not available.")

        detector = AASISTDetector(str(ckpt_path), device="cpu")
        noise = torch.randn(2, 64600)
        res = detector.predict_batch(noise)

        # Raw logits can be well outside [0, 1] range
        raw = res["raw_scores"]
        self.assertTrue(np.any(raw > 1.0) or np.any(raw < 0.0))


if __name__ == "__main__":
    unittest.main()
