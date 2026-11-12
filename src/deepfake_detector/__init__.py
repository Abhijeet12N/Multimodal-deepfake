"""Multimodal detection APIs, loaded on demand.

Audio-only evaluation and Colab training avoid importing face/video dependencies.
Existing public imports retain their names.
"""
from importlib import import_module

__version__ = "1.0.0"
_EXPORT_GROUPS = {
    ".config": ["ALLOWED_AUDIO_EXTENSIONS", "ALLOWED_VIDEO_EXTENSIONS", "AUDIO_LEN",
                "AUDIO_SR", "AUDIO_WINDOW_HOP", "DEVICE", "EMBED_DIM", "IMG_SIZE",
                "LIP_SIZE", "N_FRAMES"],
    ".model": ["AudioClassifier", "AudioStream", "ImageStream", "MultiTaskDetector", "SyncStream"],
    ".preprocessing": ["VideoPreprocessor", "load_audio_windows", "prepare_tensors_for_model"],
    ".inference": ["DeepfakePredictor", "category_from_path", "interpret"],
    ".audio_inference": ["StandaloneAudioPredictor"],
    ".reporting": ["build_report", "risk_level"],
    ".audio_detection": ["AASISTDetector", "AudioDetector", "LegacyWav2VecDetector",
                         "SSLAASISTDetector", "StandaloneAudioService", "XLSRMamBoDetector",
                         "get_detector", "list_detectors"],
}
_EXPORTS = {name: module for module, names in _EXPORT_GROUPS.items() for name in names}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
