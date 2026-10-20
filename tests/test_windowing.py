"""Unit tests for window generation, padding, and timestamp tracking."""
import unittest
import numpy as np

from deepfake_detector.audio_detection.windowing import (
    AudioWindow,
    AudioWindowGenerator,
)


class TestWindowing(unittest.TestCase):
    """Test suite ensuring window slicing, padding modes, and duration distinctions."""

    def setUp(self):
        self.sr = 16000
        self.win_samples = 32000  # 2.0s
        self.hop_samples = 16000  # 1.0s

    def test_short_audio_padding_distinction(self):
        # 1.0 second of audio (< 2.0s window)
        audio = np.ones(16000, dtype=np.float32) * 0.1
        generator = AudioWindowGenerator(
            sample_rate=self.sr,
            window_samples=self.win_samples,
            hop_samples=self.hop_samples,
            padding_mode="zero",
        )
        windows = generator.generate_windows(audio)

        self.assertEqual(len(windows), 1)
        win = windows[0]
        # Valid length is actual duration (16,000 samples = 1.0s)
        self.assertEqual(win.valid_length_samples, 16000)
        self.assertEqual(win.start_seconds, 0.0)
        self.assertEqual(win.end_seconds, 1.0)
        # Padded array length is 32,000 samples (2.0s)
        self.assertEqual(win.total_padded_samples, 32000)
        self.assertEqual(len(win.data), 32000)
        # Trailing 16000 samples should be zero
        self.assertTrue(np.all(win.data[16000:] == 0.0))

    def test_repeat_padding_mode(self):
        # Short audio with repeat padding (AASIST policy)
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        generator = AudioWindowGenerator(
            sample_rate=self.sr,
            window_samples=6,
            hop_samples=3,
            padding_mode="repeat",
        )
        windows = generator.generate_windows(audio)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].valid_length_samples, 3)
        self.assertEqual(windows[0].total_padded_samples, 6)
        # Check repeated pattern: [1, 2, 3, 1, 2, 3]
        np.testing.assert_array_equal(windows[0].data, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])

    def test_exact_window_audio(self):
        # Exactly 2.0 seconds of audio
        audio = np.ones(32000, dtype=np.float32) * 0.05
        generator = AudioWindowGenerator(
            sample_rate=self.sr,
            window_samples=self.win_samples,
            hop_samples=self.hop_samples,
        )
        windows = generator.generate_windows(audio)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].valid_length_samples, 32000)
        self.assertEqual(windows[0].start_seconds, 0.0)
        self.assertEqual(windows[0].end_seconds, 2.0)

    def test_trailing_partial_window_coverage(self):
        # 3.5 seconds of audio (56,000 samples)
        # Window 0: [0.0s, 2.0s] (0 - 32,000)
        # Window 1: [1.0s, 3.0s] (16,000 - 48,000)
        # Trailing window: [2.0s, 3.5s] (32,000 - 56,000, 24,000 samples padded to 32,000)
        audio = np.ones(56000, dtype=np.float32) * 0.05
        generator = AudioWindowGenerator(
            sample_rate=self.sr,
            window_samples=self.win_samples,
            hop_samples=self.hop_samples,
            min_duration_seconds=0.5,
        )
        windows = generator.generate_windows(audio)

        self.assertEqual(len(windows), 3)
        # Tail window
        tail = windows[-1]
        self.assertEqual(tail.start_seconds, 2.0)
        self.assertEqual(tail.end_seconds, 3.5)
        self.assertEqual(tail.valid_length_samples, 24000)
        self.assertEqual(tail.total_padded_samples, 32000)
        self.assertEqual(len(tail.data), 32000)

    def test_vad_silence_detection(self):
        # Audio with loud section and near-silent section
        audio = np.zeros(64000, dtype=np.float32)
        audio[:32000] = 0.1  # Loud speech
        audio[32000:] = 0.0001  # Silence

        generator = AudioWindowGenerator(
            sample_rate=self.sr,
            window_samples=self.win_samples,
            hop_samples=self.hop_samples,
            min_rms_vad=0.015,
        )
        windows = generator.generate_windows(audio)

        self.assertTrue(windows[0].is_speech)
        self.assertFalse(windows[-1].is_speech)


if __name__ == "__main__":
    unittest.main()
