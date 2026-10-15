"""Unit tests for PyTorch neural network model architectures."""
import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.model import (
    AudioClassifier,
    MultiTaskDetector,
    SyncStream,
)


class TestModels(unittest.TestCase):
    """Test suite ensuring network layers, activations, and forward pass shapes are correct."""

    def test_audio_classifier_forward(self):
        batch_size = 4
        input_dim = 768
        classifier = AudioClassifier(input_dim=input_dim)
        classifier.eval()

        dummy_features = torch.randn(batch_size, input_dim)
        with torch.no_grad():
            out = classifier(dummy_features)

        self.assertEqual(out.shape, (batch_size,))
        self.assertFalse(torch.isnan(out).any())

    def test_sync_stream_forward(self):
        batch_size = 2
        embed_dim = 256
        sync = SyncStream(embed_dim=embed_dim)
        sync.eval()

        # lips: (B, N_FRAMES, 3, LIP_SIZE, LIP_SIZE)
        dummy_lips = torch.randn(batch_size, 8, 3, 112, 112)
        # mel: (B, 1, 64, MEL_T)
        dummy_mel = torch.randn(batch_size, 1, 64, 301)

        with torch.no_grad():
            out = sync(dummy_lips, dummy_mel)

        self.assertEqual(out.shape, (batch_size, embed_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_multitask_detector_forward(self):
        batch_size = 2
        embed_dim = 128
        # Test offline without downloading wav2vec weights
        detector = MultiTaskDetector(embed_dim=embed_dim, load_wav2vec=False)
        detector.eval()

        dummy_faces = torch.randn(batch_size, 8, 3, 299, 299)
        dummy_lips = torch.randn(batch_size, 8, 3, 112, 112)
        dummy_audio = torch.randn(batch_size, 48000)
        dummy_mel = torch.randn(batch_size, 1, 64, 301)

        with torch.no_grad():
            preds = detector(dummy_faces, dummy_audio, dummy_lips, dummy_mel)

        self.assertIn("video", preds)
        self.assertIn("audio", preds)
        self.assertIn("any", preds)

        self.assertEqual(preds["video"].shape, (batch_size,))
        self.assertEqual(preds["audio"].shape, (batch_size,))
        self.assertEqual(preds["any"].shape, (batch_size,))


if __name__ == "__main__":
    unittest.main()

