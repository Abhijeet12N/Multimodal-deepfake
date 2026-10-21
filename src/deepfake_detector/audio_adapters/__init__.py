"""Audio model adapter registry.

Provides a uniform interface for multiple audio anti-spoofing / deepfake
detection backends so that the evaluation harness and inference pipeline
can switch models without code changes.

Adapter IDs
-----------
- ``legacy_wav2vec_mlp``  – existing Wav2Vec 2.0 + MLP (baseline)
- ``aasist``              – standalone AASIST spectro-temporal graph network
- ``ssl_aasist``          – XLSR/Wav2Vec 2.0 frontend + AASIST backend
- ``xlsr_mambo``          – (optional) Hybrid Mamba-Attention backbone

Note: SSL-AASIST and XLSR-MamBo both use the Wav2Vec/XLSR family of
self-supervised speech encoders as their frontend.  Original AASIST
operates directly on raw waveform via learned sinc filterbanks and
does **not** use any Wav2Vec/XLSR model.
"""

from .base import AudioModelAdapter, AdapterInfo, ADAPTER_REGISTRY, get_adapter
from ..audio_detection import (
    AudioDetector,
    LegacyWav2VecDetector,
    AASISTDetector,
    SSLAASISTDetector,
    XLSRMamBoDetector,
    StandaloneAudioService,
)

__all__ = [
    "AudioModelAdapter",
    "AdapterInfo",
    "ADAPTER_REGISTRY",
    "get_adapter",
    "AudioDetector",
    "LegacyWav2VecDetector",
    "AASISTDetector",
    "SSLAASISTDetector",
    "XLSRMamBoDetector",
    "StandaloneAudioService",
]
