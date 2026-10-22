"""Inference engine for multimodal deepfake detection."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from torch.cuda.amp import autocast

from .config import DEVICE, EMBED_DIM
from .model import MultiTaskDetector
from .preprocessing import VideoPreprocessor, prepare_tensors_for_model


class DeepfakePredictor:
    """Predictor for multimodal deepfake detection using best_model.pt checkpoints."""

    def __init__(self, model_path: str | Path, device: torch.device | str | None = None):
        self.device = device or DEVICE
        model_path = Path(model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        print(f"[predictor] Initializing detector on device={self.device}")
        self.model = MultiTaskDetector(EMBED_DIM).to(self.device)

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[predictor] Successfully loaded checkpoint: {model_path}")
        self.preprocessor = VideoPreprocessor(device=self.device)

    @torch.no_grad()
    def predict(self, video_path: str | Path) -> Tuple[dict[str, float] | None, dict[str, Any]]:
        """Run full multimodal inference pipeline on a video file.

        Returns:
            scores: Dictionary with 'video', 'audio', and 'any' fake probabilities (0.0 - 1.0).
            info: Metadata dictionary containing preprocessing, detection confidence, and timings.
        """
        t0 = time.time()
        data, info = self.preprocessor.process(video_path)
        info["preprocess_s"] = round(time.time() - t0, 2)
        if data is None:
            return None, info

        faces_t, audio_t, lips_t, mel_t = prepare_tensors_for_model(data, self.device)

        t1 = time.time()
        if self.device.type == "cuda":
            with autocast():
                preds = self.model(faces_t, audio_t, lips_t, mel_t)
        else:
            preds = self.model(faces_t, audio_t, lips_t, mel_t)

        scores = {
            "video": float(torch.sigmoid(preds["video"]).item()),
            "audio": float(torch.sigmoid(preds["audio"]).item()),
            "any": float(torch.sigmoid(preds["any"]).item()),
        }

        # Encode small thumbnails for inspection gallery
        try:
            import base64
            import cv2
            face_crops = []
            for face in data.get("faces", []):
                thumb = cv2.resize(face, (96, 96), interpolation=cv2.INTER_AREA)
                thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    face_crops.append("data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8"))
            info["face_crops"] = face_crops

            lip_crops = []
            for lip in data.get("lips", []):
                thumb = cv2.resize(lip, (64, 64), interpolation=cv2.INTER_AREA)
                thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(".jpg", thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    lip_crops.append("data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8"))
            info["lip_crops"] = lip_crops
        except Exception:
            pass

        info["inference_s"] = round(time.time() - t1, 2)
        info["total_s"] = round(time.time() - t0, 2)
        return scores, info



def interpret(scores: dict[str, float], threshold: float = 0.5) -> dict[str, str]:
    """Turn the three head probabilities into a structured human-readable decision."""
    v = scores.get("video", 0.0)
    a = scores.get("audio", 0.0)
    y = scores.get("any", 0.0)

    is_video_fake = v > threshold
    is_audio_fake = a > threshold
    is_any_fake = y > threshold

    if not is_any_fake:
        label = "REAL"
        sub = "The model found both the video and audio streams consistent with authentic media."
        sev = "safe"
    else:
        if is_video_fake and is_audio_fake:
            label = "FAKE (both)"
            sub = "Manipulation detected in both visual and audio streams."
        elif is_video_fake:
            label = "FAKE (video)"
            sub = "Visual manipulation detected (e.g., face swap or altered lip movements)."
        elif is_audio_fake:
            label = "FAKE (audio)"
            sub = "Synthetic or manipulated audio detected (e.g., voice cloning or TTS)."
        else:
            label = "FAKE (combined score)"
            sub = "The combined multimodal fusion score indicates manipulation despite marginal individual stream signals."
        sev = "danger"

    return {"label": label, "sub": sub, "severity": sev}


def category_from_path(video_path: str | Path) -> Tuple[str | None, str | None, str | None, bool | None]:
    """Infer ground truth category from FakeAVCeleb directory structure or filename.

    Returns:
        (category_code, video_truth, audio_truth, any_fake_bool)
        Example: ('F-R', 'fake', 'real', True)
    """
    p = str(video_path).replace("\\", "/").lower()
    if "realvideo-realaudio" in p:
        return ("R-R", "real", "real", False)
    if "fakevideo-realaudio" in p:
        return ("F-R", "fake", "real", True)
    if "realvideo-fakeaudio" in p:
        return ("R-F", "real", "fake", True)
    if "fakevideo-fakeaudio" in p:
        return ("F-F", "fake", "fake", True)
    return (None, None, None, None)

