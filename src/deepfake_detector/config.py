"""Configuration parameters and constants for multimodal deepfake detection."""
import os
from pathlib import Path
import torch

# Audio & Video Preprocessing Constants
N_FRAMES = 8
AUDIO_SR = 16000
AUDIO_LEN = 3.0
AUDIO_WINDOW_HOP = 1.5
IMG_SIZE = 299
LIP_SIZE = 112
EMBED_DIM = 512
MEL_T = 301  # ~3s @ sr=16000, hop_length=160

# Image Normalization (ImageNet standard)
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

# Device Selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Allowed media formats
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

# Default Paths
DEFAULT_MODEL_PATH = Path("models/best_model.pt")
DEFAULT_AUDIO_MODEL_PATH = Path("models/audio_model.pt")

