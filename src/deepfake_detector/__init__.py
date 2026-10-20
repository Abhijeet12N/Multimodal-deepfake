"""Deepfake Detector - Multimodal Deepfake Detection System.

Combines visual features (Xception), audio representations (Wav2Vec 2.0),
and lip-sync consistency (3D/2D SyncStream) for robust deepfake analysis.
"""

from .config import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    AUDIO_LEN,
    AUDIO_SR,
    AUDIO_WINDOW_HOP,
    DEVICE,
    EMBED_DIM,
    IMG_SIZE,
    LIP_SIZE,
    N_FRAMES,
)
from .model import (
    AudioClassifier,
    AudioStream,
    ImageStream,
    MultiTaskDetector,
    SyncStream,
)
from .preprocessing import (
    VideoPreprocessor,
    load_audio_windows,
    prepare_tensors_for_model,
)
from .inference import (
    DeepfakePredictor,
    category_from_path,
    interpret,
)
from .audio_inference import StandaloneAudioPredictor
from .reporting import build_report, risk_level
from .audio_detection import (
    AASISTDetector,
    AudioDetector,
    LegacyWav2VecDetector,
    SSLAASISTDetector,
    StandaloneAudioService,
    XLSRMamBoDetector,
    get_detector,
    list_detectors,
)

__version__ = "1.0.0"

__all__ = [
    # Inference
    "DeepfakePredictor",
    "StandaloneAudioPredictor",
    "StandaloneAudioService",
    "AudioDetector",
    "LegacyWav2VecDetector",
    "AASISTDetector",
    "SSLAASISTDetector",
    "XLSRMamBoDetector",
    "get_detector",
    "list_detectors",
    "interpret",
    "category_from_path",
    "build_report",
    "risk_level",
    # Preprocessing
    "VideoPreprocessor",
    "prepare_tensors_for_model",
    "load_audio_windows",
    # Models
    "MultiTaskDetector",
    "ImageStream",
    "AudioStream",
    "SyncStream",
    "AudioClassifier",
    # Constants
    "DEVICE",
    "N_FRAMES",
    "AUDIO_SR",
    "AUDIO_LEN",
    "AUDIO_WINDOW_HOP",
    "IMG_SIZE",
    "LIP_SIZE",
    "EMBED_DIM",
    "ALLOWED_VIDEO_EXTENSIONS",
    "ALLOWED_AUDIO_EXTENSIONS",
]
