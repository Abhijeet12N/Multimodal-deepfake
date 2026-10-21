"""Detector registry for dynamically resolving and creating AudioDetector instances."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from .interfaces import AudioDetector
from .detectors.legacy_wav2vec import LegacyWav2VecDetector
from .detectors.aasist import AASISTDetector
from .detectors.ssl_aasist import SSLAASISTDetector
from .detectors.xlsr_mambo import XLSRMamBoDetector

DETECTOR_REGISTRY: dict[str, Type[AudioDetector]] = {
    "legacy_wav2vec_mlp": LegacyWav2VecDetector,
    "legacy_wav2vec": LegacyWav2VecDetector,
    "aasist": AASISTDetector,
    "aasist_l": AASISTDetector,
    "ssl_aasist": SSLAASISTDetector,
    "xlsr_mambo": XLSRMamBoDetector,
}


def register_detector(name: str, detector_cls: Type[AudioDetector]) -> None:
    """Register a new AudioDetector class."""
    DETECTOR_REGISTRY[name.lower()] = detector_cls


def get_detector(
    name: str,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> AudioDetector:
    """Instantiate and optionally load an AudioDetector by identifier.

    Parameters
    ----------
    name : str
        Model identifier (e.g. 'legacy_wav2vec_mlp', 'aasist', 'ssl_aasist').
    checkpoint_path : str | None
        Optional path to model weights to load.
    device : str
        Target device ('cpu', 'cuda').
    """
    key = name.lower()
    if key not in DETECTOR_REGISTRY:
        available = list(DETECTOR_REGISTRY.keys())
        raise KeyError(f"Unknown detector '{name}'. Available detectors: {available}")

    detector_cls = DETECTOR_REGISTRY[key]
    if key == "aasist_l":
        kwargs["is_lightweight"] = True

    detector = detector_cls(checkpoint_path=checkpoint_path, device=device, **kwargs)
    return detector


def list_detectors() -> list[str]:
    """Return all registered detector identifiers."""
    return sorted(DETECTOR_REGISTRY.keys())
