"""Fine-tune raw-waveform AASIST using independent train/val/dev/eval splits.

GPU Colab entry point. Selects weights on val, fits recording calibration on
dev, then evaluates once on eval. No Fairseq or SSL-AASIST is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detector.audio_detection import get_detector, StandaloneAudioService
from deepfake_detector.audio_detection.calibration import ScoreCalibrator
from deepfake_detector.audio_detection.preprocessing import load_audio


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_splits(manifest: Path, reserved_hashes: set[str]):
    splits = {split: [] for split in ("train", "val", "dev", "eval")}
    seen = {}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            split = row.get("split")
            if split not in splits or row.get("label") not in {"0", "1"}:
                raise ValueError("Require split=train/val/dev/eval and label=0(real)/1(fake).")
            path = Path(row["path"])
            path = (path if path.is_absolute() else manifest.parent / path).resolve()
            if path.suffix.lower() not in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"} or not path.is_file():
                raise ValueError(f"Raw audio required: {path}")
            digest = sha256(path)
            if digest in reserved_hashes and split != "eval":
                raise ValueError(f"Reserved evaluation clip cannot enter {split}: {path}")
            identities = [("file_hash", digest)]
            for key in ("speaker_id", "source_id"):
                value = row.get(key, "").strip()
                if not value or value.startswith(("speaker_unknown", "asv2019_")):
                    raise ValueError(f"A real {key} is required for {path}")
                identities.append((key, value))
            for identity in identities:
                previous = seen.setdefault(identity, split)
                if previous != split:
                    raise ValueError(f"Split leakage: {identity} occurs in {previous} and {split}")
            row.update(path=str(path), label=int(row["label"]))
            splits[split].append(row)
    for split, rows in splits.items():
        if {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"{split} needs both genuine and spoof recordings")
    if len(splits["dev"]) < 1000:
        raise ValueError("At least 1000 separate dev recordings are required for studio calibration.")
    return splits


class TrainWindows(Dataset):
    def __init__(self, rows, length=64600, augment=True):
        self.rows, self.length, self.augment = rows, length, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wave, _ = load_audio(row["path"], target_sr=16000)
        if not len(wave) or not np.isfinite(wave).all():
            raise ValueError(f"Invalid waveform: {row['path']}")
        if len(wave) > self.length:
            start = random.randint(0, len(wave) - self.length)
            wave = wave[start:start + self.length].copy()
        else:
            wave = np.tile(wave, int(np.ceil(self.length / len(wave))))[:self.length].copy()
        if self.augment:
            # Both classes receive the same augmentation distribution.
            wave *= random.uniform(0.7, 1.3)
            if random.random() < 0.5:
                rms = np.sqrt(np.mean(wave.astype(np.float64) ** 2))
                sigma = rms / (10 ** (random.uniform(20, 40) / 20))
                wave += np.random.normal(0, sigma, len(wave)).astype(np.float32)
        # Upstream AASIST CE convention is 0=spoof, 1=bonafide.
        return torch.from_numpy(wave.astype(np.float32)), 1 - row["label"]


def score_recordings(detector, rows, output, batch_size):
    service = StandaloneAudioService(detector, batch_size=batch_size)
    scores, labels = [], []
    with output.open("w", newline="", encoding="utf-8") as stream:
        fields = ["sample_id", "dataset", "split", "speaker_id", "source_id", "ground_truth", "predicted_score"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            result = service.predict(row["path"])
            if result["status"] != "ok":
                raise ValueError(f"Scoring failed: {row['path']}")
            score = result["raw_recording_score"]
            scores.append(score)
            labels.append(row["label"])
            writer.writerow({"sample_id": row.get("sample_id", row["path"]), "dataset": row.get("dataset", "custom"),
                             "split": row["split"], "speaker_id": row["speaker_id"], "source_id": row["source_id"],
                             "ground_truth": row["label"], "predicted_score": score})
            if (index + 1) % 100 == 0:
                print(f"Scored {index + 1}/{len(rows)} {row['split']} recordings", flush=True)
                stream.flush()
    return np.asarray(scores), np.asarray(labels), service


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["aasist", "aasist_l"], default="aasist")
    parser.add_argument("--reserved-hashes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 2 or args.lr <= 0:
        parser.error("Require epochs/patience >=1, batch-size >=2 and a positive learning rate.")
    if not torch.cuda.is_available():
        parser.error("Select a GPU Colab runtime before fine-tuning.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    reserved_doc = json.loads(args.reserved_hashes.read_text(encoding="utf-8"))
    reserved_hashes = {r["sha256"] for r in reserved_doc["reserved_eval_clips"]}
    if not reserved_hashes:
        raise ValueError("Reserved evaluation exclusion list cannot be empty")
    splits = read_splits(args.manifest, reserved_hashes)
    args.output.mkdir(parents=True, exist_ok=True)
    best_path = args.output / f"{args.model}_finetuned.pth"
    if best_path.resolve() == args.checkpoint.resolve() or best_path.exists():
        raise ValueError("Use a fresh output directory; preserve existing checkpoints.")
    detector = get_detector(args.model, checkpoint_path=str(args.checkpoint), device="cuda")
    counts = np.bincount([1 - r["label"] for r in splits["train"]], minlength=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(1 / counts, dtype=torch.float32, device="cuda"))
    optimizer = torch.optim.AdamW(detector.model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TrainWindows(splits["train"]), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, drop_last=True)
    if not len(loader):
        raise ValueError("Training split must contain at least one full batch")
    best_auc, stale, history = -1.0, 0, []
    for epoch in range(args.epochs):
        detector.model.train()
        loss_sum = 0.0
        for waves, targets in loader:
            waves, targets = waves.cuda(), targets.cuda()
            optimizer.zero_grad(set_to_none=True)
            _, logits = detector.model(waves)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()
        detector.model.eval()
        val_scores, val_labels, _ = score_recordings(detector, splits["val"], args.output / "val_scores.csv", args.batch_size)
        auc = float(roc_auc_score(val_labels, -val_scores))
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(loader), "val_auc": auc})
        (args.output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(history[-1], flush=True)
        if auc > best_auc:
            best_auc, stale = auc, 0
            torch.save({"model_state_dict": detector.model.state_dict(), "model_id": args.model,
                        "epoch": epoch + 1, "val_auc": auc, "parent_checkpoint_sha256": sha256(args.checkpoint),
                        "manifest_sha256": sha256(args.manifest)}, best_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    # Reload through the production adapter so checkpoint fingerprints agree locally.
    detector.load(str(best_path), device="cuda")
    dev_scores, dev_labels, service = score_recordings(detector, splits["dev"], args.output / "dev_scores.csv", args.batch_size)
    cal = ScoreCalibrator(model_id=args.model, score_orientation="higher_is_bonafide", uncertain_band=(0.4, 0.6))
    cal.fit(dev_scores, dev_labels)
    cal.fit_data = {"source_file": "dev_scores.csv", "split": "dev", "sample_count": len(dev_labels),
                    "protocol": "studio_sliding_window", "score_level": "recording",
                    "pipeline_signature": service.pipeline_signature(), "manifest_sha256": sha256(args.manifest)}
    pipeline = {"model_id": args.model, "score_orientation": "higher_is_bonafide",
                "pipeline_signature": service.pipeline_signature()}
    (args.output / "pipeline.json").write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    # Preserve the completed dev fit even if the longer eval stage is interrupted.
    cal.save(args.output / f"{args.model}_platt.json")
    eval_scores, eval_labels, _ = score_recordings(detector, splits["eval"], args.output / "eval_scores.csv", args.batch_size)
    probabilities = cal.predict_proba(eval_scores)
    predicted = probabilities >= cal.operating_threshold
    real, fake = eval_labels == 0, eval_labels == 1
    fpr, tpr, _ = roc_curve(eval_labels, -eval_scores)
    eer_index = int(np.argmin(np.abs(fpr - (1 - tpr))))
    metrics = {**cal.evaluate(eval_scores, eval_labels), "roc_auc": float(roc_auc_score(eval_labels, probabilities)),
               "balanced_accuracy": float(balanced_accuracy_score(eval_labels, predicted)),
               "real_false_positive_rate": float(predicted[real].mean()), "fake_recall": float(predicted[fake].mean()),
               "uncertain_rate": float(((probabilities >= 0.4) & (probabilities <= 0.6)).mean())}
    metrics["eer"] = float((fpr[eer_index] + (1 - tpr[eer_index])) / 2)
    metrics["decision_policy"] = "Binary metrics at dev-calibrated 0.5; uncertain rate reported separately."
    cal.held_out_evaluation = {"source_file": "eval_scores.csv", "split": "eval", "metrics": metrics}
    cal.save(args.output / f"{args.model}_platt.json")
    (args.output / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Export checkpoint, calibration and reports from {args.output}")


if __name__ == "__main__":
    main()
