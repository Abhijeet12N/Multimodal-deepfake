"""Backward-compatibility shim. Core audio inference has moved to src/deepfake_detector."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.audio_inference import StandaloneAudioPredictor
from deepfake_detector.model import AudioClassifier

__all__ = ["StandaloneAudioPredictor", "AudioClassifier"]
