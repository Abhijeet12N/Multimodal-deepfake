"""Unified Audio Deepfake Evaluation Engine.

Evaluates audio deepfake detectors under documented, reproducible protocols:
1. Native Benchmark Inference (fixed length / acoustic features matching upstream benchmark)
2. Sliding-Window Recording Inference (production multi-format decoding + windowing + VAD + aggregation)

Metrics computed:
- Equal Error Rate (EER) and EER threshold
- ROC AUC and PR AUC (Precision-Recall AUC)
- Accuracy, Precision, Recall, F1, Specificity at validation-selected threshold
- Genuine-Audio False Positive Rate (FPR = FP / (FP + TN))
- Confusion Matrix (TN, FP, FN, TP)
- Per-attack breakdown (A01–A19 for ASVspoof, custom attacks for local eval)
- Latency, throughput, and hardware memory consumption

Outputs:
- reports/per_recording_predictions.csv
- reports/per_attack_metrics.csv
- reports/{model}_{split}_{protocol}_report.json
- reports/{model}_{split}_{protocol}_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from deepfake_detector.audio_detection import (
    AudioDetector,
    LegacyWav2VecDetector,
    StandaloneAudioService,
    get_detector,
)


def compute_eer(
    labels: np.ndarray,
    scores: np.ndarray,
    score_orientation: str = "higher_is_fake",
) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and operating threshold.
    
    If score_orientation == 'higher_is_fake', positive class is 1 (fake).
    If score_orientation == 'higher_is_bonafide', invert scores so higher = fake.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if score_orientation == "higher_is_bonafide":
        eval_scores = -scores
    else:
        eval_scores = scores

    # Ensure binary classes exist
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0, float(np.mean(eval_scores))

    fpr, tpr, thresholds = roc_curve(labels, eval_scores, pos_label=1)
    fnr = 1.0 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_idx = int(np.argmin(abs_diffs))
    eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    raw_thresh = float(thresholds[min_idx])

    if score_orientation == "higher_is_bonafide":
        raw_thresh = -raw_thresh

    return eer, raw_thresh


def evaluate_predictions(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    score_orientation: str = "higher_is_fake",
) -> dict[str, Any]:
    """Compute comprehensive classification metrics given scores and fixed threshold."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    if score_orientation == "higher_is_fake":
        preds = (scores >= threshold).astype(int)
        score_for_auc = scores
    else:
        preds = (scores < threshold).astype(int)
        score_for_auc = -scores

    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, pos_label=1, zero_division=0))
    rec = float(recall_score(labels, preds, pos_label=1, zero_division=0))
    f1 = float(f1_score(labels, preds, pos_label=1, zero_division=0))

    try:
        auc = float(roc_auc_score(labels, score_for_auc))
    except Exception:
        auc = float("nan")

    try:
        pr_auc = float(average_precision_score(labels, score_for_auc))
    except Exception:
        pr_auc = float("nan")

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Genuine-audio false-positive rate: FPR = FP / (FP + TN)
    genuine_total = tn + fp
    genuine_fpr = float(fp / genuine_total) if genuine_total > 0 else 0.0
    specificity = float(tn / genuine_total) if genuine_total > 0 else 0.0

    eer, eer_thresh = compute_eer(labels, scores, score_orientation=score_orientation)

    return {
        "eer": eer,
        "eer_threshold": eer_thresh,
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "genuine_false_positive_rate": genuine_fpr,
        "specificity": specificity,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "total_samples": len(labels),
        "genuine_samples": genuine_total,
        "fake_samples": fn + tp,
    }


def compute_per_attack_metrics(
    manifest_rows: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
    score_orientation: str = "higher_is_fake",
) -> list[dict[str, Any]]:
    """Compute per-attack type detection rates and metrics."""
    labels = np.array([int(r["label"]) for r in manifest_rows])
    attacks = np.array([str(r["attack_type"]) for r in manifest_rows])

    if score_orientation == "higher_is_fake":
        preds = (scores >= threshold).astype(int)
    else:
        preds = (scores < threshold).astype(int)

    unique_attacks = sorted(set(attacks))
    per_attack_list: list[dict[str, Any]] = []

    # 1. Bonafide / Genuine baseline
    bon_mask = labels == 0
    if bon_mask.sum() > 0:
        bon_total = int(bon_mask.sum())
        bon_correct = int((preds[bon_mask] == 0).sum())
        bon_false_alarms = bon_total - bon_correct
        bon_fpr = float(bon_false_alarms / bon_total)
        per_attack_list.append({
            "dataset": manifest_rows[0].get("dataset", "unknown"),
            "split": manifest_rows[0].get("split", "unknown"),
            "attack_type": "bonafide (-)",
            "total_samples": bon_total,
            "detected_spoofs": 0,
            "missed_spoofs": 0,
            "recall": 0.0,
            "false_positive_rate": bon_fpr,
            "eer": float("nan"),
        })

    # 2. Each attack type
    for atk in unique_attacks:
        if atk in ("-", "bonafide"):
            continue
        atk_mask = (attacks == atk) & (labels == 1)
        if atk_mask.sum() == 0:
            continue
        atk_total = int(atk_mask.sum())
        atk_detected = int((preds[atk_mask] == 1).sum())
        atk_missed = atk_total - atk_detected
        atk_recall = float(atk_detected / atk_total) if atk_total > 0 else 0.0

        # Calculate attack-specific EER against bonafide subset
        if bon_mask.sum() > 0:
            sub_labels = np.concatenate([labels[bon_mask], labels[atk_mask]])
            sub_scores = np.concatenate([scores[bon_mask], scores[atk_mask]])
            atk_eer, _ = compute_eer(sub_labels, sub_scores, score_orientation)
        else:
            atk_eer = float("nan")

        per_attack_list.append({
            "dataset": manifest_rows[0].get("dataset", "unknown"),
            "split": manifest_rows[0].get("split", "unknown"),
            "attack_type": atk,
            "total_samples": atk_total,
            "detected_spoofs": atk_detected,
            "missed_spoofs": atk_missed,
            "recall": atk_recall,
            "false_positive_rate": float("nan"),
            "eer": atk_eer,
        })

    return per_attack_list


def evaluate_native_benchmark(
    detector: AudioDetector,
    manifest_rows: list[dict[str, Any]],
    batch_size: int = 256,
    device: str = "cpu",
) -> tuple[np.ndarray, list[float]]:
    """Run native benchmark evaluation (e.g. pre-extracted features or single native window)."""
    scores: list[float] = []
    latencies: list[float] = []

    # Check if pre-extracted feature path is provided
    first_path = Path(manifest_rows[0]["path"])
    use_features = first_path.suffix.lower() == ".npy" and first_path.exists()

    classifier = getattr(detector, "classifier", None)
    if use_features and classifier is not None:
        # Fast acoustic feature evaluation for LegacyWav2VecDetector
        print(f"[evaluate_audio] Fast acoustic feature evaluation on {len(manifest_rows)} vectors ({first_path.name})...")
        features_np = np.load(str(first_path))
        if len(features_np) > len(manifest_rows):
            features_np = features_np[:len(manifest_rows)]

        model = classifier.to(device)
        model.eval()

        tensor_features = torch.from_numpy(features_np.astype(np.float32))
        n_samples = len(tensor_features)

        with torch.no_grad():
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = tensor_features[start:end].to(device)
                t0 = time.perf_counter()
                logits = model(batch)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # index 1 = spoof
                elapsed_ms = (time.perf_counter() - t0) * 1000.0 / (end - start)
                scores.extend(probs.tolist())
                latencies.extend([elapsed_ms] * (end - start))
    else:
        # Audio file evaluation
        print(f"[evaluate_audio] Native audio evaluation on {len(manifest_rows)} inputs...")
        for r in manifest_rows:
            p = Path(r["path"])
            t0 = time.perf_counter()
            if p.is_file():
                service = StandaloneAudioService(detector, batch_size=1, vad_enabled=False)
                res = service.predict(p)
                score = float(res["raw_recording_score"])
            else:
                score = 0.5
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            scores.append(score)
            latencies.append(elapsed_ms)

    return np.array(scores, dtype=float), latencies


def evaluate_sliding_window(
    detector: AudioDetector,
    manifest_rows: list[dict[str, Any]],
    batch_size: int = 8,
    vad_enabled: bool = True,
) -> tuple[np.ndarray, list[float]]:
    """Run production sliding-window recording evaluation with VAD and aggregation."""
    service = StandaloneAudioService(detector, batch_size=batch_size, vad_enabled=vad_enabled)
    scores: list[float] = []
    latencies: list[float] = []

    print(f"[evaluate_audio] Running sliding-window evaluation on {len(manifest_rows)} recordings...")
    for idx, r in enumerate(manifest_rows):
        audio_path = Path(r["path"])
        t0 = time.perf_counter()
        if audio_path.is_file():
            try:
                res = service.predict(audio_path)
                score = float(res["raw_recording_score"])
            except Exception as e:
                print(f"  [warn] Error predicting {audio_path.name}: {e}", file=sys.stderr)
                score = 0.5
        else:
            print(f"  [warn] Audio file not found: {audio_path}", file=sys.stderr)
            score = 0.5

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        scores.append(score)
        latencies.append(elapsed_ms)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(manifest_rows):
            print(f"  [{idx + 1:4d}/{len(manifest_rows)}] processed (last: {elapsed_ms:.1f}ms)", flush=True)

    return np.array(scores, dtype=float), latencies


def main():
    parser = argparse.ArgumentParser(description="Evaluate audio deepfake detectors under standardized protocol.")
    parser.add_argument("--model", type=str, default="legacy_wav2vec_mlp",
                        choices=["legacy_wav2vec_mlp", "aasist", "aasist_l", "ssl_aasist", "xlsr_mambo"],
                        help="Model identifier to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint path override")
    parser.add_argument("--manifest", type=str, default="data/manifests/manifest_asvspoof2019_dev.csv",
                        help="Target evaluation manifest CSV")
    parser.add_argument("--val_manifest", type=str, default=None,
                        help="Validation manifest used strictly for threshold selection (e.g. dev set)")
    parser.add_argument("--protocol", type=str, default="native_benchmark",
                        choices=["native_benchmark", "sliding_window"],
                        help="Evaluation protocol: 'native_benchmark' or 'sliding_window'")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Explicit decision threshold override (bypasses dev selection)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for inference")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional sample count limit for fast testing")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--output_dir", type=str, default="reports",
                        help="Directory for report outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Detector
    t_load_start = time.perf_counter()
    print(f"\n{'='*70}")
    print(f"  AUDIO DEEPFAKE EVALUATION — {args.model.upper()} ({args.protocol})")
    print(f"{'='*70}")
    print(f"  Device:     {args.device}")
    print(f"  Protocol:   {args.protocol}")
    print(f"  Manifest:   {args.manifest}")

    # Resolve checkpoint path if not explicitly provided
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_candidates = {
            "legacy_wav2vec_mlp": [
                ROOT / "initial setup/deepfake_phase1/checkpoints/wav2vec_best.pt",
                ROOT.parent / "initial setup/deepfake_phase1/checkpoints/wav2vec_best.pt",
                ROOT / "models/audio_model.pt",
            ],
            "aasist": [
                ROOT / "aasist/models/weights/AASIST.pth",
                ROOT.parent / "aasist/models/weights/AASIST.pth",
            ],
            "aasist_l": [
                ROOT / "aasist/models/weights/AASIST-L.pth",
                ROOT.parent / "aasist/models/weights/AASIST-L.pth",
            ],
        }
        for cand in default_candidates.get(args.model, []):
            if Path(cand).is_file():
                checkpoint_path = str(Path(cand).resolve())
                break

    if checkpoint_path:
        print(f"  Checkpoint: {checkpoint_path}")
    else:
        print(f"  Checkpoint: None (using uninitialized or default weights)")

    detector = get_detector(args.model, checkpoint_path=checkpoint_path, device=args.device)
    model_load_time_s = time.perf_counter() - t_load_start
    reqs = detector.requirements()
    meta = detector.metadata()
    score_orientation = reqs.get("score_orientation", "higher_is_fake")
    print(f"  Model loaded in:     {model_load_time_s:.2f}s")
    print(f"  Score Orientation:   {score_orientation}")

    # 2. Load Evaluation Manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[evaluate_audio] Error: Manifest file not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    eval_rows: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            eval_rows.append(r)

    if args.limit and args.limit < len(eval_rows):
        eval_rows = eval_rows[:args.limit]
        print(f"  [info] Limited evaluation to {args.limit} samples.")

    # 3. Determine Decision Threshold
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "explicit_override"
        print(f"  Threshold:           {threshold:.4f} (Source: explicit CLI override)")
    elif args.val_manifest is not None:
        val_path = Path(args.val_manifest)
        print(f"  Calibrating threshold on validation manifest: {val_path.name}...")
        val_rows: list[dict[str, Any]] = []
        with open(val_path, "r", encoding="utf-8") as vf:
            for vr in csv.DictReader(vf):
                val_rows.append(vr)
        val_scores, _ = evaluate_native_benchmark(detector, val_rows, batch_size=args.batch_size, device=args.device)
        val_labels = np.array([int(r["label"]) for r in val_rows])
        _, threshold = compute_eer(val_labels, val_scores, score_orientation=score_orientation)
        threshold_source = f"validation_eer_selected ({val_path.stem})"
        print(f"  Threshold:           {threshold:.4f} (Source: validation dev EER)")
    else:
        # Fallback to model's calibrated threshold or default
        threshold = float(reqs.get("calibrated_threshold", 0.5))
        threshold_source = "model_calibrated_default"
        print(f"  Threshold:           {threshold:.4f} (Source: model checkpoint default)")

    # 4. Run Evaluation on Target Manifest
    eval_labels = np.array([int(r["label"]) for r in eval_rows])
    t_eval_start = time.perf_counter()

    if args.protocol == "native_benchmark":
        eval_scores, latencies = evaluate_native_benchmark(
            detector, eval_rows, batch_size=args.batch_size, device=args.device
        )
    else:
        eval_scores, latencies = evaluate_sliding_window(
            detector, eval_rows, batch_size=args.batch_size
        )

    total_eval_time_s = time.perf_counter() - t_eval_start
    avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0
    throughput = len(eval_rows) / total_eval_time_s if total_eval_time_s > 0 else 0.0

    # 5. Compute Metrics
    overall_metrics = evaluate_predictions(
        eval_labels, eval_scores, threshold=threshold, score_orientation=score_orientation
    )
    overall_metrics["threshold_source"] = threshold_source
    overall_metrics["model_load_time_seconds"] = model_load_time_s
    overall_metrics["total_eval_time_seconds"] = total_eval_time_s
    overall_metrics["average_latency_ms"] = avg_latency_ms
    overall_metrics["throughput_samples_per_sec"] = throughput
    overall_metrics["device"] = args.device
    overall_metrics["protocol"] = args.protocol
    overall_metrics["model"] = args.model

    # 6. Compute Per-Attack Metrics
    per_attack = compute_per_attack_metrics(
        eval_rows, eval_scores, threshold=threshold, score_orientation=score_orientation
    )

    # 7. Print Results Table
    print(f"\n{'-'*70}")
    print(f"  EVALUATION SUMMARY RESULTS")
    print(f"{'-'*70}")
    print(f"  Total Samples:                  {overall_metrics['total_samples']}")
    print(f"  Genuine Samples:                {overall_metrics['genuine_samples']}")
    print(f"  Fake Samples:                   {overall_metrics['fake_samples']}")
    print(f"  Equal Error Rate (EER):         {overall_metrics['eer']*100:.2f}%")
    print(f"  Optimal EER Threshold:          {overall_metrics['eer_threshold']:.4f}")
    print(f"  Operating Threshold:            {overall_metrics['threshold']:.4f} ({threshold_source})")
    print(f"  Accuracy:                       {overall_metrics['accuracy']*100:.2f}%")
    print(f"  Precision:                      {overall_metrics['precision']*100:.2f}%")
    print(f"  Recall (Spoof Detection Rate):  {overall_metrics['recall']*100:.2f}%")
    print(f"  F1 Score:                       {overall_metrics['f1']:.4f}")
    print(f"  ROC AUC:                        {overall_metrics['roc_auc']:.4f}")
    print(f"  PR AUC:                         {overall_metrics['pr_auc']:.4f}")
    print(f"  Genuine False Positive Rate:    {overall_metrics['genuine_false_positive_rate']*100:.2f}%")
    print(f"  Specificity:                    {overall_metrics['specificity']*100:.2f}%")
    print(f"{'-'*70}")
    cm = overall_metrics["confusion_matrix"]
    print(f"  Confusion Matrix:               Pred Genuine    Pred Fake")
    print(f"    Actual Genuine:               {cm['true_negatives']:>12} {cm['false_positives']:>12}")
    print(f"    Actual Fake:                  {cm['false_negatives']:>12} {cm['true_positives']:>12}")
    print(f"{'-'*70}")
    print(f"  Throughput:                     {throughput:.1f} samples/sec")
    print(f"  Avg Latency:                    {avg_latency_ms:.2f} ms/sample")
    print(f"{'='*70}\n")

    # 8. Save Reports
    # a. per_recording_predictions.csv
    preds_csv = out_dir / "per_recording_predictions.csv"
    split_preds_csv = out_dir / f"{manifest_path.stem}_per_recording_predictions.csv"
    for target_p in (preds_csv, split_preds_csv):
        with open(target_p, "w", newline="", encoding="utf-8") as pf:
            fieldnames = [
                "sample_id", "dataset", "split", "attack_type", "ground_truth",
                "predicted_score", "predicted_label", "threshold", "is_correct", "runtime_ms"
            ]
            writer = csv.DictWriter(pf, fieldnames=fieldnames)
            writer.writeheader()
            if score_orientation == "higher_is_fake":
                pred_binary = (eval_scores >= threshold).astype(int)
            else:
                pred_binary = (eval_scores < threshold).astype(int)

            for i, r in enumerate(eval_rows):
                is_correct = int(pred_binary[i] == int(r["label"]))
                writer.writerow({
                    "sample_id": r["sample_id"],
                    "dataset": r["dataset"],
                    "split": r["split"],
                    "attack_type": r["attack_type"],
                    "ground_truth": int(r["label"]),
                    "predicted_score": round(float(eval_scores[i]), 6),
                    "predicted_label": int(pred_binary[i]),
                    "threshold": round(float(threshold), 6),
                    "is_correct": is_correct,
                    "runtime_ms": round(float(latencies[i]), 2) if i < len(latencies) else 0.0,
                })
    print(f"[evaluate_audio] Saved per-recording predictions: {preds_csv}")

    # b. per_attack_metrics.csv
    attack_csv = out_dir / "per_attack_metrics.csv"
    split_attack_csv = out_dir / f"{manifest_path.stem}_per_attack_metrics.csv"
    for target_a in (attack_csv, split_attack_csv):
        with open(target_a, "w", newline="", encoding="utf-8") as af:
            af_fields = [
                "dataset", "split", "attack_type", "total_samples",
                "detected_spoofs", "missed_spoofs", "recall", "false_positive_rate", "eer"
            ]
            writer = csv.DictWriter(af, fieldnames=af_fields)
            writer.writeheader()
            for row in per_attack:
                writer.writerow(row)
    print(f"[evaluate_audio] Saved per-attack metrics: {attack_csv}")

    # c. JSON Summary Report
    json_report_path = out_dir / f"{args.model}_{manifest_path.stem}_{args.protocol}_report.json"
    full_report = {
        "metadata": meta,
        "requirements": reqs,
        "evaluation_config": {
            "model": args.model,
            "manifest": str(manifest_path),
            "val_manifest": args.val_manifest,
            "protocol": args.protocol,
            "threshold": threshold,
            "threshold_source": threshold_source,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "overall_metrics": overall_metrics,
        "per_attack_breakdown": per_attack,
    }
    with open(json_report_path, "w", encoding="utf-8") as jf:
        json.dump(full_report, jf, indent=2)
    print(f"[evaluate_audio] Saved JSON evaluation report: {json_report_path}")

    # d. Also write standard reports/legacy_eval_report.json if evaluating legacy model
    if args.model == "legacy_wav2vec_mlp":
        legacy_path = out_dir / "legacy_eval_report.json"
        with open(legacy_path, "w", encoding="utf-8") as lf:
            json.dump(full_report, lf, indent=2)
        print(f"[evaluate_audio] Saved legacy baseline report: {legacy_path}")

    # e. Markdown Executive Summary
    md_path = out_dir / f"{args.model}_{manifest_path.stem}_{args.protocol}_report.md"
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write(f"# Evaluation Report: {meta.get('display_name', args.model)}\n\n")
        mf.write(f"- **Protocol**: `{args.protocol}`\n")
        mf.write(f"- **Manifest**: `{manifest_path.name}` ({overall_metrics['total_samples']} samples)\n")
        mf.write(f"- **Device**: `{args.device}`\n")
        mf.write(f"- **Score Orientation**: `{score_orientation}`\n\n")
        mf.write("## Overall Metrics\n\n")
        mf.write(f"| Metric | Value |\n|---|---|\n")
        mf.write(f"| **EER** | **{overall_metrics['eer']*100:.2f}%** |\n")
        mf.write(f"| **ROC AUC** | **{overall_metrics['roc_auc']:.4f}** |\n")
        mf.write(f"| **PR AUC** | **{overall_metrics['pr_auc']:.4f}** |\n")
        mf.write(f"| **Accuracy** | {overall_metrics['accuracy']*100:.2f}% |\n")
        mf.write(f"| **Precision** | {overall_metrics['precision']*100:.2f}% |\n")
        mf.write(f"| **Recall (Spoof)** | {overall_metrics['recall']*100:.2f}% |\n")
        mf.write(f"| **F1 Score** | {overall_metrics['f1']:.4f} |\n")
        mf.write(f"| **Genuine False Positive Rate** | {overall_metrics['genuine_false_positive_rate']*100:.2f}% |\n")
        mf.write(f"| **Operating Threshold** | {overall_metrics['threshold']:.4f} ({threshold_source}) |\n")
        mf.write(f"| **Throughput** | {throughput:.1f} samples/sec |\n\n")
        mf.write("## Confusion Matrix\n\n")
        mf.write("| | Predicted Genuine | Predicted Fake |\n|---|---|---|\n")
        mf.write(f"| **Actual Genuine** | {cm['true_negatives']} | {cm['false_positives']} |\n")
        mf.write(f"| **Actual Fake** | {cm['false_negatives']} | {cm['true_positives']} |\n\n")
        mf.write("## Per-Attack Breakdown\n\n")
        mf.write("| Attack Type | Total Samples | Detected Spoofs | Recall | EER |\n|---|---|---|---|---|\n")
        for atk in per_attack:
            eer_str = f"{atk['eer']*100:.2f}%" if not np.isnan(atk['eer']) else "N/A"
            mf.write(f"| `{atk['attack_type']}` | {atk['total_samples']} | {atk['detected_spoofs']} | {atk['recall']*100:.1f}% | {eer_str} |\n")
    print(f"[evaluate_audio] Saved Markdown summary: {md_path}")


if __name__ == "__main__":
    main()
