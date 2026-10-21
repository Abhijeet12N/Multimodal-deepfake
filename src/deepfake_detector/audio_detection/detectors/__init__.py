"""Audio detector implementations conforming to the AudioDetector interface."""
from .legacy_wav2vec import LegacyWav2VecDetector
from .aasist import AASISTDetector
from .ssl_aasist import SSLAASISTDetector
from .xlsr_mambo import XLSRMamBoDetector

__all__ = [
    "LegacyWav2VecDetector",
    "AASISTDetector",
    "SSLAASISTDetector",
    "XLSRMamBoDetector",
]
