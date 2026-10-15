"""Unit tests for video and audio preprocessing pipelines."""
import unittest
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.config import (
    AUDIO_LEN,
    AUDIO_SR,
    AUDIO_WINDOW_HOP,
    IMG_SIZE,
    LIP_SIZE,
    MEL_T,
    N_FRAMES,
)
from deepfake_detector.preprocessing import (
    VideoPreprocessor,
    prepare_tensors_for_model,
)


class TestPreprocessing(unittest.TestCase):
    """Test suite for frame cropping, normalization, and tensor preparation."""

    def setUp(self):
        # Initialize without GPU requirement
        self.preprocessor = VideoPreprocessor(device="cpu")
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Put some non-zero pixels inside the bounding box
        self.dummy_frame[100:300, 100:300] = 128

    def test_crop_face_shape(self):
        box = np.array([100.0, 100.0, 300.0, 300.0])
        face = self.preprocessor.crop_face(self.dummy_frame, box, margin=40)
        self.assertIsNotNone(face)
        self.assertEqual(face.shape, (IMG_SIZE, IMG_SIZE, 3))

    def test_crop_lip_shape(self):
        box = np.array([100.0, 100.0, 300.0, 300.0])
        lip = self.preprocessor.crop_lip(self.dummy_frame, box, margin=10)
        self.assertIsNotNone(lip)
        self.assertEqual(lip.shape, (LIP_SIZE, LIP_SIZE, 3))

    def test_crop_invalid_box_returns_none(self):
        box = np.array([500.0, 500.0, 400.0, 400.0])  # inverted / empty
        face = self.preprocessor.crop_face(self.dummy_frame, box, margin=0)
        self.assertIsNone(face)

    def test_prepare_tensors_for_model(self):
        mock_data = {
            "faces": np.zeros((N_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
            "lips": np.zeros((N_FRAMES, LIP_SIZE, LIP_SIZE, 3), dtype=np.uint8),
            "audio": np.zeros(int(AUDIO_SR * AUDIO_LEN), dtype=np.float32),
            "mel": np.zeros((64, MEL_T), dtype=np.float32),
        }
        faces_t, audio_t, lips_t, mel_t = prepare_tensors_for_model(mock_data, torch.device("cpu"))

        self.assertEqual(faces_t.shape, (1, N_FRAMES, 3, IMG_SIZE, IMG_SIZE))
        self.assertEqual(audio_t.shape, (1, int(AUDIO_SR * AUDIO_LEN)))
        self.assertEqual(lips_t.shape, (1, N_FRAMES, 3, LIP_SIZE, LIP_SIZE))
        self.assertEqual(mel_t.shape, (1, 1, 64, MEL_T))
        self.assertEqual(faces_t.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()

