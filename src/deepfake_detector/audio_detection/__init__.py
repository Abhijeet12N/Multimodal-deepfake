"""Standalone audio anti-spoofing and deepfake detection package."""
from .interfaces import AudioDetector
from .registry import (
    DETECTOR_REGISTRY,
    get_detector,
    list_detectors,
    register_detector,
)
from .detectors import (
    AASISTDetector,
    LegacyWav2VecDetector,
    SSLAASISTDetector,
    XLSRMamBoDetector,
)
from .preprocessing import load_audio
from .windowing import AudioWindow, AudioWindowGenerator
from .aggregation import RecordingScoreAggregator, merge_suspicious_intervals
from .service import StandaloneAudioService

__all__ = [
    "AudioDetector",
    "DETECTOR_REGISTRY",
    "get_detector",
    "list_detectors",
    "register_detector",
    "LegacyWav2VecDetector",
    "AASISTDetector",
    "SSLAASISTDetector",
    "XLSRMamBoDetector",
    "load_audio",
    "AudioWindow",
    "AudioWindowGenerator",
    "RecordingScoreAggregator",
    "merge_suspicious_intervals",
    "StandaloneAudioService",
]
