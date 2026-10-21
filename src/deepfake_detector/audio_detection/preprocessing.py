"""Audio decoding and preprocessing module.

Separates raw media decoding from window generation and model inference.
Supports WAV, MP3, FLAC, M4A, AAC, OGG formats with automatic fallback.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    file_path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, float]:
    """Load an audio recording and return normalized float32 waveform and duration.

    Parameters
    ----------
    file_path : str | Path
        Path to the audio file.
    target_sr : int
        Target sample rate in Hz (default: 16000).
    mono : bool
        Whether to convert multi-channel audio to mono.

    Returns
    -------
    waveform : np.ndarray
        Floating-point audio array in [-1.0, 1.0].
    duration_seconds : float
        Actual duration of the audio in seconds.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Check container format
    is_mp4_container = False
    try:
        with open(path, "rb") as f:
            header = f.read(16)
            if b"ftyp" in header:
                is_mp4_container = True
    except Exception:
        pass

    # Strategy 1: soundfile (fastest native reader for WAV, FLAC, standard MP3)
    if not is_mp4_container:
        try:
            audio, sr = sf.read(str(path), dtype="float32")
            if mono and audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            duration = float(len(audio) / target_sr)
            return audio.astype(np.float32, copy=False), duration
        except Exception:
            pass

    # Strategy 2: imageio_ffmpeg binary pipe fallback
    try:
        import subprocess
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-v", "quiet", "-i", str(path),
            "-f", "f32le", "-ac", "1" if mono else "2",
            "-ar", str(target_sr), "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        if len(audio) > 0:
            duration = float(len(audio) / target_sr)
            return audio, duration
    except Exception:
        pass

    # Strategy 3: librosa fallback
    audio, sr = librosa.load(str(path), sr=target_sr, mono=mono)
    duration = float(len(audio) / target_sr)
    return audio.astype(np.float32, copy=False), duration
