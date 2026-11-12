"""Evaluate reserved local clips using the same loaders as the studio.

Never fits weights, calibration or thresholds. Saves each completed prediction
immediately so interrupted runs can be inspected, including stage timings.
"""
from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows: list[dict]) -> dict:
    completed = [r for r in rows if not r.get("error")]
    real = [r for r in completed if r["label"] == "real"]
    fake = [r for r in completed if r["label"] == "fake"]
    tp = sum(r["verdict"] == "fake" for r in fake)
    tn = sum(r["verdict"] == "real" for r in real)
    fp = sum(r["verdict"] == "fake" for r in real)
    fn = sum(r["verdict"] == "real" for r in fake)
    uncertain = sum(r["verdict"] == "uncertain" for r in completed)
    # Abstentions count as unresolved rather than silently dropping them.
    tpr = tp / len(fake) if fake else None
    tnr = tn / len(real) if real else None
    return {
        "completed": len(completed), "errors": len(rows) - len(completed),
        "correct": tp + tn, "uncertain": uncertain,
        "accuracy_including_abstentions": (tp + tn) / len(completed) if completed else None,
        "real_false_positive_rate": fp / len(real) if real else None,
        "fake_recall": tpr,
        "balanced_accuracy_including_abstentions": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "true_positives": tp, "true_negatives": tn, "false_positives": fp, "false_negatives": fn,
        "total_analysis_seconds": sum(r.get("analysis_seconds", 0) for r in completed),
    }


def write_report(rows, summary, output):
    lines = ["# Audio pipeline evaluation", "",
             "Fresh evaluation of eight reserved clips through the studio loaders. "
             "No weights, thresholds or calibration were fitted using these clips.", "",
             "Folder labels are provisional; this small set does not establish general accuracy. "
             "The old six-clip AASIST calibrator and native-input Wav2Vec calibrator were not applied.", "",
             "| Detector | Correct / 8 | Inconclusive | Real clips flagged / 4 | Fake clips detected / 4 | Analysis seconds |",
             "|---|---:|---:|---:|---:|---:|"]
    for model_id, result in summary["models"].items():
        if "load_error" in result:
            lines.append(f"| {model_id} | Load failed | | | | |")
        else:
            lines.append(f"| {model_id} | {result['correct']} | {result['uncertain']} | "
                         f"{result['false_positives']} | {result['true_positives']} | {result['total_analysis_seconds']:.2f} |")
    lines += ["", "Inconclusive results remain unresolved and count against correct/8. "
              "Times exclude model loading and are one CPU run with two PyTorch threads, not a repeated latency benchmark.", "",
              "## Per-clip decisions", "", "| File | Label | " + " | ".join(summary["models"]) + " |",
              "|---|---|" + "---|" * len(summary["models"])]
    files = list(dict.fromkeys(r["filename"] for r in rows))
    for filename in files:
        matches = {r["model_id"]: r for r in rows if r["filename"] == filename}
        first = next(iter(matches.values()))
        lines.append("| " + filename + " | " + first["label"] + " | " + " | ".join(
            matches.get(model, {}).get("verdict", "error") for model in summary["models"]) + " |")
    lines += ["", "## Timing breakdown", "", "| Detector | Decode seconds | Model seconds |",
              "|---|---:|---:|"]
    for model in summary["models"]:
        times = [r.get("timings", {}) for r in rows if r["model_id"] == model]
        lines.append(f"| {model} | {sum(t.get('decode_seconds', 0) for t in times):.3f} | "
                     f"{sum(t.get('model_seconds', 0) for t in times):.3f} |")
    lines += ["", "Raw window scores, timestamps, thresholds, calibration status and file SHA-256 values "
              "are saved in `predictions.jsonl`; flat results are in `predictions.csv`.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    for console in (sys.stdout, sys.stderr):
        if hasattr(console, "reconfigure"):
            console.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", type=Path, default=ROOT.parent / "initial setup" / "deepfake_phase1" / "Audios")
    parser.add_argument("--models", nargs="+", default=["legacy_standalone", "legacy_wav2vec_mlp", "aasist", "aasist_l"],
                        choices=["legacy_standalone", "legacy_wav2vec_mlp", "aasist", "aasist_l"])
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "audio_pipeline_evaluation")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--offline", action="store_true", help="Require already cached Hugging Face weights")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    faulthandler.dump_traceback_later(60, repeat=True)
    print("Importing runtime...", flush=True)
    import torch
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(ROOT / "ui"))
    import app as studio
    clips = [(label.lower(), path) for label in ("Real", "Fake")
             for path in sorted((args.audio_root / label).glob("*"))
             if path.is_file() and path.suffix.lower() in studio.ALLOWED_AUDIO_EXTENSIONS]
    if not clips:
        parser.error("No evaluation clips found under Real/ and Fake/")
    rows, summaries = [], {}
    detail_path = args.output / "predictions.jsonl"
    with detail_path.open("w", encoding="utf-8") as stream:
        for model_id in args.models:
            faulthandler.dump_traceback_later(60)
            print(f"Loading {model_id}...", flush=True)
            started = time.perf_counter()
            predictor = studio.load_audio_predictor(model_id, force_retry=True)
            load_seconds = time.perf_counter() - started
            if predictor is None:
                summaries[model_id] = {"load_error": studio.audio_predictor_error, "load_seconds": load_seconds}
                print(f"LOAD ERROR: {studio.audio_predictor_error}", flush=True)
                continue
            model_rows = []
            for label, clip in clips:
                faulthandler.dump_traceback_later(60)
                started = time.perf_counter()
                row = {"model_id": model_id, "label": label, "filename": clip.name,
                       "sha256": hashlib.sha256(clip.read_bytes()).hexdigest()}
                print(f"  {model_id}: {clip.name}", flush=True)
                try:
                    result = predictor.predict(clip)
                    row.update(
                        verdict=result.get("verdict", result.get("label")),
                        score=result.get("raw_recording_score", result.get("fake_probability")),
                        calibrated_fake_probability=result.get("calibrated_fake_probability"),
                        threshold=result.get("threshold"),
                        flagged_windows=result.get("fake_windows_count", sum(w["is_fake"] for w in result.get("windows", []))),
                        total_windows=result.get("evaluated_windows", result.get("windows_analyzed")),
                        analysis_seconds=time.perf_counter() - started,
                        timings=result.get("timings", {}), result=result,
                    )
                    print(f"    {row['verdict']} score={row['score']} seconds={row['analysis_seconds']:.3f}", flush=True)
                except Exception as exc:
                    row.update(error=str(exc), analysis_seconds=time.perf_counter() - started)
                    print(f"    ERROR: {exc}", flush=True)
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
                model_rows.append(row)
                rows.append(row)
            summaries[model_id] = {**summarize(model_rows), "load_seconds": load_seconds}
            del predictor
    faulthandler.cancel_dump_traceback_later()
    fields = ["model_id", "label", "filename", "verdict", "score", "calibrated_fake_probability",
              "threshold", "flagged_windows", "total_windows", "analysis_seconds", "sha256", "error"]
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": "studio_sliding_window", "split": "eval", "fit_on_clips": False,
        "threads": args.threads, "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "models": summaries,
        "limitations": ["Labels are supplied by folder names; source authenticity has not been independently verified.",
                        "Eight clips are a smoke evaluation, not an estimate of general accuracy.",
                        "EER and AUC need a larger independent test set for reliable comparison."],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(rows, summary, args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
