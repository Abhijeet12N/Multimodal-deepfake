"""Preprocessing pipelines for video frames, faces, lips, and audio signals."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import cv2
import librosa
import numpy as np
import torch
from facenet_pytorch import MTCNN

from .config import (
    AUDIO_LEN,
    AUDIO_SR,
    AUDIO_WINDOW_HOP,
    DEVICE,
    IMG_MEAN,
    IMG_SIZE,
    IMG_STD,
    LIP_SIZE,
    MEL_T,
    N_FRAMES,
)

# Standard normalization constants as numpy arrays
IMG_MEAN_ARR = np.array(IMG_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
IMG_STD_ARR = np.array(IMG_STD, dtype=np.float32).reshape(1, 3, 1, 1)


class VideoPreprocessor:
    """Extracts aligned face crops, lip crops, raw waveform, and mel spectrogram from a video."""

    def __init__(self, device: torch.device | str | None = None):
        self.device = device or DEVICE
        self.mtcnn = MTCNN(
            image_size=IMG_SIZE,
            margin=40,
            min_face_size=60,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
            device=self.device,
        )

    def crop_face(self, frame: np.ndarray, box: np.ndarray, margin: int = 40) -> np.ndarray | None:
        """Crop and resize the facial bounding box."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

    def crop_lip(self, frame: np.ndarray, box: np.ndarray, margin: int = 10) -> np.ndarray | None:
        """Crop and resize the lower face/mouth region."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        lh = y2 - y1
        ly1 = max(0, y1 + int(lh * 0.55) - margin)
        y2 = min(h, y2 + margin)
        lx1 = max(0, x1 - margin)
        lx2 = min(w, x2 + margin)
        crop = frame[ly1:y2, lx1:lx2]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (LIP_SIZE, LIP_SIZE))

    def process(
        self,
        video_path: str | Path,
        n_frames: int = N_FRAMES,
    ) -> Tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
        """Extract multi-stream features from a video file."""
        video_path = str(video_path)
        info: dict[str, Any] = {"video_path": video_path, "fail_reason": None}

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info["total_frames"] = total
        info["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        if total < n_frames:
            cap.release()
            info["fail_reason"] = "too_few_frames"
            return None, info

        idxs = np.linspace(0, total - 1, n_frames, dtype=int)

        # Detect face on the middle frame
        mid = idxs[len(idxs) // 2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, mf = cap.read()
        if not ret:
            cap.release()
            info["fail_reason"] = "cannot_read_mid"
            return None, info

        rgb = cv2.cvtColor(mf, cv2.COLOR_BGR2RGB)
        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None or len(boxes) == 0:
            cap.release()
            info["fail_reason"] = "no_face"
            return None, info

        best_idx = int(np.argmax(probs))
        box = boxes[best_idx]
        info["face_prob"] = float(np.max(probs))
        info["face_box"] = box.tolist()

        faces, lips = [], []
        for ix in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, ix)
            ret, fr = cap.read()
            if not ret:
                if faces:
                    faces.append(faces[-1])
                    lips.append(lips[-1])
                continue
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            face = self.crop_face(rgb, box)
            lip = self.crop_lip(rgb, box)
            if face is None or lip is None:
                if faces:
                    faces.append(faces[-1])
                    lips.append(lips[-1])
                continue
            faces.append(face)
            lips.append(lip)
        cap.release()

        if len(faces) < n_frames // 2:
            info["fail_reason"] = "too_few_face_crops"
            return None, info

        while len(faces) < n_frames:
            faces.append(faces[-1])
        while len(lips) < n_frames:
            lips.append(lips[-1])

        faces = faces[:n_frames]
        lips = lips[:n_frames]

        # Extract audio waveform
        try:
            audio, _ = librosa.load(video_path, sr=AUDIO_SR, mono=True, duration=AUDIO_LEN)
            tlen = int(AUDIO_SR * AUDIO_LEN)
            if len(audio) < tlen:
                audio = np.pad(audio, (0, tlen - len(audio)))
            else:
                audio = audio[:tlen]
            info["audio_rms"] = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        except Exception as exc:
            audio = np.zeros(int(AUDIO_SR * AUDIO_LEN), dtype=np.float32)
            info["audio_rms"] = 0.0
            info["fail_reason"] = f"audio_err:{str(exc)[:30]}"

        # Extract normalized log-mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=AUDIO_SR,
            n_mels=64,
            n_fft=512,
            hop_length=160,
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        if mel.shape[1] < MEL_T:
            mel = np.pad(mel, ((0, 0), (0, MEL_T - mel.shape[1])))
        else:
            mel = mel[:, :MEL_T]

        return {
            "faces": np.stack(faces).astype(np.uint8),
            "lips": np.stack(lips).astype(np.uint8),
            "audio": audio.astype(np.float32),
            "mel": mel.astype(np.float32),
        }, info


def prepare_tensors_for_model(
    data: dict[str, np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw preprocessing dictionary into normalized PyTorch model input tensors."""
    faces = data["faces"].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    faces = (faces - IMG_MEAN_ARR) / IMG_STD_ARR
    lips = data["lips"].astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    mel = data["mel"][None]  # (1, 64, MEL_T)

    faces_t = torch.from_numpy(faces).float().unsqueeze(0).to(device)  # (1, N, 3, H, W)
    lips_t = torch.from_numpy(lips).float().unsqueeze(0).to(device)
    audio_t = torch.from_numpy(data["audio"]).float().unsqueeze(0).to(device)
    mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)
    return faces_t, audio_t, lips_t, mel_t


def load_audio_windows(
    audio_path: str | Path,
    sample_rate: int = AUDIO_SR,
    audio_seconds: float = AUDIO_LEN,
    window_hop_seconds: float = AUDIO_WINDOW_HOP,
) -> Tuple[np.ndarray, list[int], float]:
    """Load an audio recording and compute window start offsets for sliding-window analysis."""
    waveform, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if waveform.size == 0:
        raise ValueError("Decoded audio waveform is empty.")

    decoded_seconds = len(waveform) / sample_rate
    sample_count = int(sample_rate * audio_seconds)

    if len(waveform) <= sample_count:
        starts = [0]
    else:
        hop = int(sample_rate * window_hop_seconds)
        starts = list(range(0, len(waveform) - sample_count + 1, hop))
        tail_start = len(waveform) - sample_count
        if starts[-1] != tail_start:
            starts.append(tail_start)

    return waveform, starts, decoded_seconds

