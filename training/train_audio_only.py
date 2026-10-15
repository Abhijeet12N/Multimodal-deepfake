"""Train a standalone audio fake/real classifier from FakeAVCeleb videos.

This is separate from deepfake_v3.py. It uses pretrained Wav2Vec 2.0 as a
frozen feature extractor, then trains a small binary classifier on audio_fake.

Expected inputs are the subject-disjoint metadata CSVs made by:
    python notebook_v2/deepfake_v3.py --stage scan

Stages:
    extract: decode each complete recording into overlapping windows and cache Wav2Vec features
    train:   train/save audio_model.pt and report held-out test metrics
    all:     run both stages
"""
import argparse
import os
import random
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import Wav2Vec2Model

SEED = 42
SAMPLE_RATE = 16000
AUDIO_SECONDS = 3.0
WINDOW_HOP_SECONDS = 1.5
WAV2VEC_NAME = "facebook/wav2vec2-base-960h"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORK_DIR = Path(os.environ.get("WORK_DIR", "/content/work"))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_windows(path):
    """Decode a complete recording and split it into overlapping 3-second windows."""
    try:
        wave, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        if wave.size == 0:
            raise ValueError("decoded audio is empty")
        window = int(SAMPLE_RATE * AUDIO_SECONDS)
        hop = int(SAMPLE_RATE * WINDOW_HOP_SECONDS)
        if len(wave) <= window:
            starts = [0]
        else:
            starts = list(range(0, len(wave) - window + 1, hop))
            tail_start = len(wave) - window
            if starts[-1] != tail_start:
                starts.append(tail_start)
        windows = []
        for start in starts:
            segment = wave[start:start + window]
            if len(segment) < window:
                segment = np.pad(segment, (0, window - len(segment)))
            windows.append(segment.astype(np.float32))
        return windows, True
    except Exception as exc:
        print(f"[audio decode skipped] {path}: {exc}")
        return [], False


def extract_split(split, batch_size=8):
    csv_path = WORK_DIR / f"meta_{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run deepfake_v3.py --stage scan first.")
    df = pd.read_csv(csv_path)
    if not {"path", "audio_fake"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must include path and audio_fake columns")

    cache_path = WORK_DIR / f"audio_features_{split}.npz"
    model = Wav2Vec2Model.from_pretrained(WAV2VEC_NAME).to(DEVICE).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    feature_batches = []
    group_batches = []
    label_batches = []
    decode_ok = np.zeros(len(df), dtype=np.uint8)
    batch_waves, batch_groups, batch_labels = [], [], []
    amp_enabled = DEVICE.type == "cuda"

    def encode_batch():
        if not batch_waves:
            return
        wave_tensor = torch.from_numpy(np.stack(batch_waves)).to(DEVICE)
        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden = model(wave_tensor).last_hidden_state.mean(dim=1)
            else:
                hidden = model(wave_tensor).last_hidden_state.mean(dim=1)
        feature_batches.append(hidden.float().cpu().numpy())
        group_batches.append(np.asarray(batch_groups, dtype=np.int32))
        label_batches.append(np.asarray(batch_labels, dtype=np.float32))
        batch_waves.clear()
        batch_groups.clear()
        batch_labels.clear()

    for row_id, (path, label) in enumerate(tqdm(
        zip(df["path"].astype(str), df["audio_fake"].astype(np.float32)),
        total=len(df), desc=f"Extract all audio windows ({split})"
    )):
        windows, valid = load_windows(path)
        decode_ok[row_id] = int(valid)
        for wave in windows:
            batch_waves.append(wave)
            batch_groups.append(row_id)
            batch_labels.append(label)
            if len(batch_waves) >= batch_size:
                encode_batch()
    encode_batch()
    model.to("cpu")
    del model

    if not feature_batches:
        raise RuntimeError(f"No audio windows were decoded for split '{split}'. Check FFmpeg/audio decoding.")
    features = np.concatenate(feature_batches, axis=0)
    group_ids = np.concatenate(group_batches)
    window_labels = np.concatenate(label_batches)
    clip_labels = df["audio_fake"].astype(np.float32).to_numpy()
    np.savez_compressed(
        cache_path,
        features=features,
        group_ids=group_ids,
        window_labels=window_labels,
        clip_labels=clip_labels,
        decode_ok=decode_ok,
    )
    print(
        f"Saved {cache_path}: {len(features)} windows from "
        f"{int(decode_ok.sum())}/{len(df)} decoded recordings"
    )


class AudioClassifier(nn.Module):
    def __init__(self, input_dim=768):
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

    def forward(self, features):
        return self.net(features).squeeze(1)


def read_cache(split):
    path = WORK_DIR / f"audio_features_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run --stage extract first.")
    data = np.load(path)
    decode_ok = data["decode_ok"].astype(bool)
    group_ids = data["group_ids"]
    keep_windows = decode_ok[group_ids]
    return (
        data["features"][keep_windows],
        data["window_labels"][keep_windows],
        group_ids[keep_windows],
        data["clip_labels"],
        decode_ok,
    )


def aggregate_recordings(window_probabilities, group_ids, clip_labels, decode_ok):
    """Average window probabilities so the final prediction describes the full recording."""
    scores, labels = [], []
    for group_id, label in enumerate(clip_labels):
        if not decode_ok[group_id]:
            continue
        selected = window_probabilities[group_ids == group_id]
        if selected.size:
            scores.append(float(np.mean(selected)))
            labels.append(float(label))
    return np.asarray(scores, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def safe_auc(labels, probabilities):
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")


def metrics(labels, probabilities, threshold):
    pred = probabilities >= threshold
    return {
        "auc": safe_auc(labels, probabilities),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "threshold": float(threshold),
    }


def predict_array(model, x, batch_size=512):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start:start + batch_size]).float().to(DEVICE)
            outputs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def train(epochs=30, batch_size=256, patience=5, learning_rate=1e-3):
    set_seed()
    x_train, y_train, train_groups, train_clip_labels, train_ok = read_cache("train")
    x_val, y_val, val_groups, val_clip_labels, val_ok = read_cache("val")
    x_test, y_test, test_groups, test_clip_labels, test_ok = read_cache("test")
    model = AudioClassifier(x_train.shape[1]).to(DEVICE)
    valid_clip_labels = train_clip_labels[train_ok]
    positives = max(float(valid_clip_labels.sum()), 1.0)
    negatives = max(float(len(valid_clip_labels) - valid_clip_labels.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    group_counts = np.bincount(train_groups, minlength=len(train_clip_labels)).clip(min=1)
    sample_weights = (1.0 / group_counts[train_groups]).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).float(),
        torch.from_numpy(sample_weights).float(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    best_auc, stale = float("-inf"), 0
    checkpoint_path = WORK_DIR / "audio_model.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for features, labels, weights in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            weights = weights.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            per_window_loss = nn.functional.binary_cross_entropy_with_logits(
                model(features), labels, pos_weight=loss_fn.pos_weight, reduction="none"
            )
            loss = (per_window_loss * weights).sum() / weights.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        val_window_prob = predict_array(model, x_val)
        val_prob, val_record_labels = aggregate_recordings(
            val_window_prob, val_groups, val_clip_labels, val_ok
        )
        val_auc = safe_auc(val_record_labels, val_prob)
        print(f"Epoch {epoch}: train_loss={total_loss / max(len(loader), 1):.4f} val_recording_auc={val_auc:.4f}")
        score = val_auc if np.isfinite(val_auc) else -1.0
        if score > best_auc:
            best_auc, stale = score, 0
            thresholds = np.arange(0.05, 0.951, 0.01)
            threshold = max(
                thresholds,
                key=lambda value: f1_score(val_record_labels, val_prob >= value, zero_division=0),
            )
            torch.save({
                "classifier_state_dict": model.cpu().state_dict(),
                "input_dim": int(x_train.shape[1]),
                "wav2vec_model": WAV2VEC_NAME,
                "sample_rate": SAMPLE_RATE,
                "audio_seconds": AUDIO_SECONDS,
                "window_hop_seconds": WINDOW_HOP_SECONDS,
                "aggregation": "mean_window_probability",
                "pipeline_version": 2,
                "threshold": float(threshold),
                "validation_recordings": int(len(val_record_labels)),
                "validation_metrics": metrics(val_record_labels, val_prob, threshold),
            }, checkpoint_path)
            model.to(DEVICE)
            print(f"Saved best checkpoint: {checkpoint_path} (threshold={threshold:.2f})")
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping: validation AUC did not improve.")
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["classifier_state_dict"])
    test_window_prob = predict_array(model, x_test)
    test_prob, test_record_labels = aggregate_recordings(
        test_window_prob, test_groups, test_clip_labels, test_ok
    )
    test_metrics = metrics(test_record_labels, test_prob, checkpoint["threshold"])
    test_metrics["recordings"] = int(len(test_record_labels))
    checkpoint["test_metrics"] = test_metrics
    torch.save(checkpoint, checkpoint_path)
    print("Held-out test metrics:", test_metrics)
    print(f"Audio checkpoint: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["extract", "train", "all"], default="all")
    parser.add_argument("--extract-batch", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("extract", "all"):
        for split in ("train", "val", "test"):
            extract_split(split, args.extract_batch)
    if args.stage in ("train", "all"):
        train(args.epochs, args.batch, args.patience, args.lr)


if __name__ == "__main__":
    main()
