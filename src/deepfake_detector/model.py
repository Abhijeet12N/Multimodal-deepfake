"""PyTorch neural network architectures for multimodal deepfake detection."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from .config import EMBED_DIM, IMG_SIZE


class ImageStream(nn.Module):
    """Visual stream using an Xception backbone with mean-std temporal pooling."""

    def __init__(self, embed_dim: int = EMBED_DIM, pretrained: bool = False):
        super().__init__()
        self.bb = timm.create_model("xception", pretrained=pretrained, num_classes=0)
        bdim = self.bb.num_features
        self.proj = nn.Sequential(
            nn.Linear(2 * bdim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, N, 3, H, W)
        B, N = x.shape[:2]
        f = self.bb(x.view(B * N, 3, IMG_SIZE, IMG_SIZE))
        f = f.view(B, N, -1)
        pooled = torch.cat([f.mean(1), f.std(1)], dim=1)
        return self.proj(pooled)


class AudioStream(nn.Module):
    """Audio stream extracting representations from a frozen Wav2Vec 2.0 encoder."""

    def __init__(self, embed_dim: int = EMBED_DIM, wav2vec_model: str = "facebook/wav2vec2-base-960h"):
        super().__init__()
        self.w2v = Wav2Vec2Model.from_pretrained(wav2vec_model)
        for p in self.w2v.parameters():
            p.requires_grad = False
        self.w2v.eval()
        self.proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T_audio)
        with torch.no_grad():
            self.w2v.eval()
            h = self.w2v(x).last_hidden_state.mean(dim=1)
        return self.proj(h.float())


class SyncStream(nn.Module):
    """Audiovisual synchronization stream combining 3D lip crops and 2D mel spectrogram."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, (3, 3, 3), (1, 2, 2), (1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.audio = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (1, 2), (1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3), (1, 2), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, lips: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        # lips: (B, N, 3, H, W) -> permute to (B, 3, N, H, W)
        # mel: (B, 1, n_mels, T)
        v = self.visual(lips.permute(0, 2, 1, 3, 4)).flatten(1)
        a = self.audio(mel).flatten(1)
        return self.proj(torch.cat([v, a], dim=1))


class MultiTaskDetector(nn.Module):
    """Multimodal fusion network with three binary classification heads.

    Heads:
        - video: Visual manipulation detection
        - audio: Audio manipulation detection
        - any: Overall (fused) manipulation detection
    """

    def __init__(self, embed_dim: int = EMBED_DIM, load_wav2vec: bool = True):
        super().__init__()
        self.image = ImageStream(embed_dim, pretrained=False)
        self.audio = AudioStream(embed_dim) if load_wav2vec else None
        self.sync = SyncStream(embed_dim)
        total_dim = 3 * embed_dim
        self.trunk = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head_video = nn.Linear(256, 1)
        self.head_audio = nn.Linear(256, 1)
        self.head_any = nn.Linear(256, 1)

    def forward(
        self,
        faces: torch.Tensor,
        audio: torch.Tensor,
        lips: torch.Tensor,
        mel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        i = self.image(faces)
        if self.audio is not None:
            a = self.audio(audio)
        else:
            a = torch.zeros(faces.shape[0], self.trunk[0].in_features // 3, device=faces.device)
        s = self.sync(lips, mel)
        h = self.trunk(torch.cat([i, a, s], dim=1))
        return {
            "video": self.head_video(h).squeeze(1),
            "audio": self.head_audio(h).squeeze(1),
            "any": self.head_any(h).squeeze(1),
        }


class AudioClassifier(nn.Module):
    """Standalone classifier head operating on Wav2Vec 2.0 embeddings."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(1)
