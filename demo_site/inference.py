"""Backward-compatibility shim. Core model architectures and inference have moved to src/deepfake_detector."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.config import (
    AUDIO_LEN,
    AUDIO_SR,
    DEVICE,
    EMBED_DIM,
    IMG_MEAN,
    IMG_SIZE,
    IMG_STD,
    LIP_SIZE,
    MEL_T,
    N_FRAMES,
)
from deepfake_detector.inference import (
    DeepfakePredictor,
    category_from_path,
    interpret,
)
from deepfake_detector.model import (
    AudioStream,
    ImageStream,
    MultiTaskDetector,
    SyncStream,
)
from deepfake_detector.preprocessing import VideoPreprocessor

__all__ = [
    "DeepfakePredictor",
    "interpret",
    "category_from_path",
    "VideoPreprocessor",
    "MultiTaskDetector",
    "ImageStream",
    "AudioStream",
    "SyncStream",
    "DEVICE",
    "EMBED_DIM",
    "N_FRAMES",
    "AUDIO_SR",
    "AUDIO_LEN",
    "IMG_SIZE",
    "LIP_SIZE",
    "MEL_T",
]

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("model_path")
    p.add_argument("video_path")
    a = p.parse_args()
    pred = DeepfakePredictor(a.model_path)
    scores, info = pred.predict(a.video_path)
    print("SCORES:", scores)
    print("INFO:", info)
    if scores:
        print("DECISION:", interpret(scores))
        print("GROUND TRUTH (path-derived):", category_from_path(a.video_path))
