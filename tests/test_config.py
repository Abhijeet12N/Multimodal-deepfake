"""Unit tests for configuration constants."""
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    AUDIO_LEN,
    AUDIO_SR,
    AUDIO_WINDOW_HOP,
    EMBED_DIM,
    IMG_MEAN,
    IMG_SIZE,
    IMG_STD,
    LIP_SIZE,
    MEL_T,
    N_FRAMES,
)


class TestConfig(unittest.TestCase):
    """Test suite ensuring consistent global configuration."""

    def test_dimensions(self):
        self.assertEqual(N_FRAMES, 8)
        self.assertEqual(AUDIO_SR, 16000)
        self.assertEqual(AUDIO_LEN, 3.0)
        self.assertEqual(AUDIO_WINDOW_HOP, 1.5)
        self.assertEqual(IMG_SIZE, 299)
        self.assertEqual(LIP_SIZE, 112)
        self.assertEqual(EMBED_DIM, 512)
        self.assertEqual(MEL_T, 301)

    def test_image_normalization(self):
        self.assertEqual(len(IMG_MEAN), 3)
        self.assertEqual(len(IMG_STD), 3)

    def test_allowed_extensions(self):
        self.assertIn(".mp4", ALLOWED_VIDEO_EXTENSIONS)
        self.assertIn(".avi", ALLOWED_VIDEO_EXTENSIONS)
        self.assertIn(".wav", ALLOWED_AUDIO_EXTENSIONS)
        self.assertIn(".mp3", ALLOWED_AUDIO_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()

