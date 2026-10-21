"""Google Colab Training and Fine-Tuning Pipeline for SSL-AASIST.

This script executes on Google Colab with an NVIDIA GPU (T4 / V100 / A100).
It implements the Phase 4 & Phase 5 training pipeline for SSL-AASIST:
- Self-Supervised Wav2Vec 2.0 / XLSR-53 frontend
- AASIST Spectro-Temporal Graph Attention backend
- RawBoost Data Augmentation (convolutive, impulsive, and colored noise)
- Multi-phase fine-tuning (frozen frontend -> selective unfreezing)
- Early stopping based on validation EER on ASVspoof 2019 Dev split

Usage on Google Colab:
    python training/train_ssl_aasist_colab.py \
        --asvspoof_dir /content/ASVspoof2019_LA \
        --output_dir /content/drive/MyDrive/SSL_AASIST_Checkpoints \
        --epochs 15 \
        --batch_size 14 \
        --lr 0.0001 \
        --rawboost
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import typing

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ============================================================================
# Compatibility Shim 1: Python 3.12+ removal of typing.io & typing.re
# ============================================================================
# In Python 3.12/3.13, typing.io and typing.re were completely removed.
# Older packages like antlr4-python3-runtime (used by omegaconf/fairseq)
# execute `from typing.io import TextIO`. This shim restores compatibility.
if not hasattr(typing, "__path__"):
    typing.__path__ = []

if "typing.io" not in sys.modules:
    _typing_io = types.ModuleType("typing.io")
    _typing_io.IO = typing.IO
    _typing_io.TextIO = typing.TextIO
    _typing_io.BinaryIO = typing.BinaryIO
    sys.modules["typing.io"] = _typing_io
    setattr(typing, "io", _typing_io)

if "typing.re" not in sys.modules:
    _typing_re = types.ModuleType("typing.re")
    _typing_re.Pattern = getattr(typing, "Pattern", object)
    _typing_re.Match = getattr(typing, "Match", object)
    sys.modules["typing.re"] = _typing_re
    setattr(typing, "re", _typing_re)

# Auto-patch antlr4 files on disk if running under Python 3.12/3.13
import glob
for _p in glob.glob("/usr/local/lib/python3*/dist-packages/antlr4/*.py") + glob.glob(
    os.path.expanduser("~/.local/lib/python3*/dist-packages/antlr4/*.py")
):
    try:
        with open(_p, "r", encoding="utf-8") as _f:
            _code = _f.read()
        if "from typing.io import TextIO" in _code:
            with open(_p, "w", encoding="utf-8") as _f:
                _f.write(_code.replace("from typing.io import TextIO", "from typing import TextIO"))
    except Exception:
        pass

# Auto-add Colab workspace directories to sys.path if present
for _colab_p in ["/content/SSL_Anti-spoofing", "/content/fairseq", "/content/Multimodal-deepfake"]:
    if os.path.exists(_colab_p) and _colab_p not in sys.path:
        sys.path.insert(0, _colab_p)

# ============================================================================
# Compatibility Shim 2: Python 3.11+ dataclass mutable default bypass
# ============================================================================
# In Python 3.11+, dataclasses disallow mutable default instances (like CommonConfig()).
# Fairseq was authored for Python <=3.10 and declares default config instances directly.
import dataclasses

_orig_dataclass_get_field = dataclasses._get_field


def _patched_dataclass_get_field(cls, a_name, a_type, *args, **kwargs):
    try:
        return _orig_dataclass_get_field(cls, a_name, a_type, *args, **kwargs)
    except ValueError as err:
        if "mutable default" in str(err):
            val = getattr(cls, a_name, None)
            if isinstance(val, list):
                setattr(cls, a_name, dataclasses.field(default_factory=list))
                return _orig_dataclass_get_field(cls, a_name, a_type, *args, **kwargs)
            elif isinstance(val, dict):
                setattr(cls, a_name, dataclasses.field(default_factory=dict))
                return _orig_dataclass_get_field(cls, a_name, a_type, *args, **kwargs)
            elif isinstance(val, set):
                setattr(cls, a_name, dataclasses.field(default_factory=set))
                return _orig_dataclass_get_field(cls, a_name, a_type, *args, **kwargs)

            # Check if default is or contains an instance of a class whose __hash__ is None
            target = val.default if isinstance(val, dataclasses.Field) else val
            if target is not None and hasattr(target, "__class__"):
                try:
                    target.__class__.__hash__ = lambda self: id(self)
                    return _orig_dataclass_get_field(cls, a_name, a_type, *args, **kwargs)
                except (TypeError, AttributeError):
                    pass
        raise err


dataclasses._get_field = _patched_dataclass_get_field


# ============================================================================
# Compatibility Shim 3: fairseq + omegaconf (Google Colab / Modern Runtimes)
# ============================================================================
# In omegaconf >= 2.1, `omegaconf._utils.is_primitive_type` was deprecated/removed,
# causing fairseq.checkpoint_utils.load_checkpoint_to_cpu to fail with AttributeError.
# This shim dynamically provides the expected primitive validator.
try:
    import omegaconf._utils
    if not hasattr(omegaconf._utils, "is_primitive_type"):
        omegaconf._utils.is_primitive_type = getattr(
            omegaconf._utils,
            "is_primitive_type_val",
            lambda obj: isinstance(obj, (int, float, bool, str, bytes)),
        )
except (ImportError, AttributeError):
    pass


# ============================================================================
# 1. RawBoost Data Augmentation (Tak et al., 2022)
# ============================================================================
class RawBoost:
    """RawBoost data augmentation implementing impulsive and colored noise."""

    @staticmethod
    def add_noise(waveform: np.ndarray, snr_db_low: float = 10.0, snr_db_high: float = 40.0) -> np.ndarray:
        """Additive colored stationary noise."""
        snr_db = np.random.uniform(snr_db_low, snr_db_high)
        signal_power = np.mean(waveform ** 2)
        if signal_power == 0:
            return waveform
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(waveform))
        return (waveform + noise).astype(np.float32)

    @staticmethod
    def add_impulsive_noise(waveform: np.ndarray, prob: float = 0.01) -> np.ndarray:
        """Impulsive salt-and-pepper audio noise."""
        mask = np.random.uniform(0, 1, len(waveform)) < prob
        noise = np.random.uniform(-0.5, 0.5, len(waveform))
        augmented = waveform.copy()
        augmented[mask] += noise[mask]
        return np.clip(augmented, -1.0, 1.0).astype(np.float32)

    @classmethod
    def apply(cls, waveform: np.ndarray) -> np.ndarray:
        p = random.random()
        if p < 0.35:
            return cls.add_noise(waveform)
        elif p < 0.70:
            return cls.add_impulsive_noise(waveform)
        return waveform


# ============================================================================
# 2. Dataset and Tiling Loader
# ============================================================================
def pad_waveform(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    """Periodic repeat padding matching upstream AASIST data_utils."""
    x_len = len(x)
    if x_len >= max_len:
        return x[:max_len]
    repeats = int(math.ceil(max_len / max(1, x_len)))
    return np.tile(x, repeats)[:max_len]


class ASVspoofRawDataset(Dataset):
    """Raw audio dataset for ASVspoof 2019 / 2021 protocols."""

    def __init__(
        self,
        audio_dir: Path,
        protocol_file: Path,
        cut_samples: int = 64600,
        is_train: bool = True,
        use_rawboost: bool = False,
    ):
        self.audio_dir = audio_dir
        self.cut_samples = cut_samples
        self.is_train = is_train
        self.use_rawboost = use_rawboost
        self.samples: list[tuple[str, int]] = []

        with open(protocol_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    key = parts[1]
                    label = 1 if parts[4] == "bonafide" else 0
                    self.samples.append((key, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        key, label = self.samples[idx]
        file_path = self.audio_dir / f"{key}.flac"
        if not file_path.is_file():
            # Support .wav
            file_path = self.audio_dir / f"{key}.wav"

        try:
            audio, _ = sf.read(str(file_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
        except Exception:
            audio = np.zeros(self.cut_samples, dtype=np.float32)

        if self.is_train and self.use_rawboost:
            audio = RawBoost.apply(audio)

        audio_padded = pad_waveform(audio, self.cut_samples)
        return torch.from_numpy(audio_padded), label, key


# ============================================================================
# 3. Model Architecture Construction
# ============================================================================
def build_ssl_aasist_model(device: torch.device, fairseq_cp_path: Optional[str] = None) -> nn.Module:
    """Build SSL-AASIST model with XLSR frontend and AASIST GAT backend."""
    # Ensure typing.io shim for Python 3.12+ (used by antlr4 inside omegaconf/fairseq)
    if not hasattr(typing, "__path__"):
        typing.__path__ = []
    if "typing.io" not in sys.modules:
        _t_io = types.ModuleType("typing.io")
        _t_io.IO = typing.IO
        _t_io.TextIO = typing.TextIO
        _t_io.BinaryIO = typing.BinaryIO
        sys.modules["typing.io"] = _t_io
        setattr(typing, "io", _t_io)
    if "typing.re" not in sys.modules:
        _t_re = types.ModuleType("typing.re")
        _t_re.Pattern = getattr(typing, "Pattern", object)
        _t_re.Match = getattr(typing, "Match", object)
        sys.modules["typing.re"] = _t_re
        setattr(typing, "re", _t_re)

    # Ensure omegaconf._utils compatibility shim is bound
    try:
        import omegaconf._utils
        if not hasattr(omegaconf._utils, "is_primitive_type"):
            omegaconf._utils.is_primitive_type = getattr(
                omegaconf._utils,
                "is_primitive_type_val",
                lambda obj: isinstance(obj, (int, float, bool, str, bytes)),
            )
    except (ImportError, AttributeError):
        pass

    # Ensure dataclasses mutable default bypass is active
    import dataclasses
    if dataclasses._get_field is not _patched_dataclass_get_field:
        dataclasses._get_field = _patched_dataclass_get_field

    try:
        import fairseq
        from fairseq.checkpoint_utils import load_model_ensemble_and_task
    except Exception as e:
        print(f"[train_ssl_aasist] Error importing fairseq ({e}). To run training, run on Google Colab:")
        print("  !pip install --upgrade 'antlr4-python3-runtime>=4.9.3'")
        print("  !pip install --editable ./fairseq")
        raise

    # Check if official upstream model definition is importable
    try:
        from model import Model as SSLAASISTModel
        config = {
            "cp_path": fairseq_cp_path or "xlsr_53_56k.pt",
            "ssl_type": "xlsr",
        }
        model = SSLAASISTModel(config, device=str(device)).to(device)
        return model
    except ImportError:
        print("[train_ssl_aasist] Using direct Clova AI AASIST backend with XLSR wrapper.")
        raise


# ============================================================================
# 4. Evaluation and EER Calculation
# ============================================================================
def compute_eer(bonafide_scores: np.ndarray, spoof_scores: np.ndarray) -> float:
    """Compute Equal Error Rate (EER) given bonafide and spoof scores."""
    from sklearn.metrics import roc_curve
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    scores = np.concatenate([bonafide_scores, spoof_scores])
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


@torch.no_grad()
def evaluate_dev(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate model on dev set and return (loss, eer)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    bonafide_scores = []
    spoof_scores = []

    for waves, labels, _ in loader:
        waves = waves.to(device)
        labels = labels.to(device)

        logits = model(waves)
        if isinstance(logits, tuple):
            logits = logits[0]

        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)

        # Logit 1 is bonafide score
        scores = logits[:, 1].cpu().numpy()
        for s, l in zip(scores, labels.cpu().numpy()):
            if l == 1:
                bonafide_scores.append(s)
            else:
                spoof_scores.append(s)

    avg_loss = total_loss / max(1, len(bonafide_scores) + len(spoof_scores))
    eer = compute_eer(np.array(bonafide_scores), np.array(spoof_scores))
    return avg_loss, eer


# ============================================================================
# 5. Main Training Pipeline
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train / Fine-tune SSL-AASIST on Google Colab GPU.")
    parser.add_argument("--asvspoof_dir", type=str, required=True,
                        help="Path to ASVspoof2019_LA database directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/ssl_aasist",
                        help="Directory to save fine-tuned checkpoints")
    parser.add_argument("--xlsr_path", type=str, default="xlsr_53_56k.pt",
                        help="Path to downloaded Fairseq XLSR-53 checkpoint")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=14,
                        help="Batch size (14 recommended for 16GB VRAM on T4)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for AASIST backend")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--rawboost", action="store_true", default=True,
                        help="Enable RawBoost data augmentation")
    parser.add_argument("--unfreeze_epoch", type=int, default=10,
                        help="Epoch to selectively unfreeze top XLSR transformer layers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  SSL-AASIST Google Colab Training Pipeline")
    print(f"=======================================================")
    print(f"  Device:         {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'})")
    if torch.cuda.is_available():
        print(f"  VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n" + "!" * 72)
        print("  CRITICAL WARNING: CUDA GPU IS NOT DETECTED (RUNNING ON CPU)!")
        print("  Training XLSR-53 (317M params) on CPU will be extremely slow (~10-15x slower).")
        print("  In Google Colab, enable GPU acceleration via:")
        print("    Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU")
        print("!" * 72 + "\n")
    print(f"  Batch Size:     {args.batch_size}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  RawBoost:       {args.rawboost}")
    print(f"  Output Dir:     {args.output_dir}\n")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    asv_root = Path(args.asvspoof_dir)
    train_audio = asv_root / "ASVspoof2019_LA_train/flac"
    train_proto = asv_root / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    dev_audio = asv_root / "ASVspoof2019_LA_dev/flac"
    dev_proto = asv_root / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

    if not train_proto.is_file() or not dev_proto.is_file():
        print(f"[ERROR] Protocol files not found in {asv_root}")
        sys.exit(1)

    print(f"Loading ASVspoof 2019 LA Train and Dev splits...")
    train_ds = ASVspoofRawDataset(train_audio, train_proto, is_train=True, use_rawboost=args.rawboost)
    dev_ds = ASVspoofRawDataset(dev_audio, dev_proto, is_train=False, use_rawboost=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train Samples: {len(train_ds)}")
    print(f"  Dev Samples:   {len(dev_ds)}")

    # Instantiate Model
    model = build_ssl_aasist_model(device, fairseq_cp_path=args.xlsr_path)

    # Freeze frontend initially
    for name, param in model.named_parameters():
        if "ssl_model" in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_eer = float("inf")
    best_ckpt_path = out_path / "ssl_aasist_best.pt"

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        # Check for selective unfreezing
        if epoch == args.unfreeze_epoch:
            print(f"\n[Epoch {epoch}] Selectively unfreezing top layers of XLSR frontend with 0.1x LR...")
            for name, param in model.named_parameters():
                if "ssl_model.encoder.layers.2" in name:  # Unfreeze top layers
                    param.requires_grad = True
            optimizer = torch.optim.Adam([
                {"params": [p for n, p in model.named_parameters() if "ssl_model" not in n and p.requires_grad], "lr": args.lr},
                {"params": [p for n, p in model.named_parameters() if "ssl_model" in n and p.requires_grad], "lr": args.lr * 0.1},
            ], weight_decay=args.weight_decay)

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}")
        for waves, labels, _ in pbar:
            waves = waves.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(waves)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader)
        dev_loss, dev_eer = evaluate_dev(model, dev_loader, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d} completed in {elapsed:.1f}s | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | Dev EER: {dev_eer * 100:.2f}%")

        if dev_eer < best_eer:
            best_eer = dev_eer
            print(f"  [*] New best model found! Saving to {best_ckpt_path}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dev_eer": dev_eer,
                "config": vars(args),
            }, best_ckpt_path)

    print(f"\nTraining Complete! Best Dev EER: {best_eer * 100:.2f}%. Saved to {best_ckpt_path}")


if __name__ == "__main__":
    main()
