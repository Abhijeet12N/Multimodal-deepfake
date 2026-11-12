"""Fit score calibration and uncertainty quantification parameters on development data.

Produces versioned calibration artifacts for AudioDetector models using Platt scaling,
isotonic regression, or temperature scaling.

Usage:
    python scripts/fit_calibration.py \
        --predictions-csv reports/calibration_fit_wav2vec_dev/manifest_asvspoof2019_dev_per_recording_predictions.csv \
        --split dev \
        --method platt \
        --model-id legacy_wav2vec_mlp \
        --score-orientation higher_is_fake \
        --output configs/calibration/legacy_wav2vec_mlp_platt.json

This example fits the historical native-feature benchmark. Studio recording calibration
requires independent recording scores and a matching --pipeline-json descriptor.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from deepfake_detector.audio_detection.calibration import (
    ScoreCalibrator,
    compute_ece,
    compute_reliability_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fit_calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit score calibration on development set predictions."
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="Path to CSV containing columns 'ground_truth' (or 'label') and 'predicted_score' (or 'raw_score').",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["dev", "val"],
        required=True,
        help="Required calibration split. Evaluation/test splits are never valid fit data.",
    )
    parser.add_argument(
        "--method",
        choices=["platt", "isotonic", "temperature"],
        default="platt",
        help="Calibration algorithm: 'platt' (logistic sigmoid), 'isotonic' (non-parametric), or 'temperature'.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unknown_model",
        help="Identifier of the model being calibrated.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="1.0.0",
        help="Semantic version string for this calibration artifact.",
    )
    parser.add_argument(
        "--score-orientation",
        choices=["higher_is_fake", "higher_is_bonafide"],
        default="higher_is_bonafide",
        help="Whether larger raw scores denote fake or genuine audio.",
    )
    parser.add_argument(
        "--operating-threshold",
        type=float,
        default=0.5,
        help="Calibrated probability threshold for classifying as fake (default: 0.5).",
    )
    parser.add_argument(
        "--uncertain-lower",
        type=float,
        default=0.40,
        help="Lower bound for uncertain probability region.",
    )
    parser.add_argument(
        "--uncertain-upper",
        type=float,
        default=0.60,
        help="Upper bound for uncertain probability region.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for output JSON calibration artifact.",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=None,
        help="Optional independent held-out CSV to evaluate calibration generalization.",
    )
    parser.add_argument(
        "--prediction-protocol", choices=["native_benchmark", "studio_sliding_window"],
        default="native_benchmark", help="Protocol that generated the calibration scores.",
    )
    parser.add_argument(
        "--pipeline-json", type=Path,
        help="For studio recording calibration: descriptor containing pipeline_signature and model_id.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["eval", "test"],
        default="eval",
        help="Split name in the independent held-out CSV.",
    )
    return parser.parse_args()


def load_data(
    csv_path: Path, split_filter: str | None = None, *, require_split: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)

    if require_split and (not split_filter or "split" not in df.columns):
        raise ValueError(f"{csv_path} must contain a split column matching '{split_filter}'.")
    if split_filter and "split" in df.columns:
        split_values = df["split"].astype(str).str.lower()
        df = df[split_values == split_filter.lower()]
        if df.empty:
            raise ValueError(f"No rows with split='{split_filter}' found in {csv_path}.")
        logger.info("Filtered to split '%s': %d rows", split_filter, len(df))

    # Detect score column
    score_col = None
    for candidate in ["predicted_score", "raw_score", "score", "fake_score"]:
        if candidate in df.columns:
            score_col = candidate
            break
    if not score_col:
        raise ValueError(f"Could not find score column in {csv_path}. Columns: {list(df.columns)}")

    # Detect label column
    label_col = None
    for candidate in ["ground_truth", "label", "is_fake", "target"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if not label_col:
        raise ValueError(f"Could not find label column in {csv_path}. Columns: {list(df.columns)}")

    scores = df[score_col].to_numpy(dtype=np.float64)
    labels = df[label_col].to_numpy(dtype=np.int64)

    # Validate classes
    valid_mask = ~np.isnan(scores) & ~np.isnan(labels)
    scores = scores[valid_mask]
    labels = labels[valid_mask]
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores must be finite and labels must be 0 (real) or 1 (fake).")

    return scores, labels


def main():
    args = parse_args()

    pipeline = {}
    if args.prediction_protocol == "studio_sliding_window":
        if not args.pipeline_json or not args.pipeline_json.is_file():
            raise ValueError("Studio recording calibration requires --pipeline-json from dev scoring.")
        pipeline = json.loads(args.pipeline_json.read_text(encoding="utf-8"))
        if pipeline.get("model_id") != args.model_id or not pipeline.get("pipeline_signature"):
            raise ValueError("Pipeline descriptor must identify this model and its pipeline_signature.")
        if pipeline.get("score_orientation") != args.score_orientation:
            raise ValueError("Pipeline score orientation does not match --score-orientation.")

    if not args.predictions_csv.is_file():
        logger.error("Predictions file not found: %s", args.predictions_csv)
        sys.exit(1)

    if args.eval_csv:
        if not args.eval_csv.is_file():
            raise ValueError(f"Held-out evaluation file not found: {args.eval_csv}")
        fit_rows = pd.read_csv(args.predictions_csv)
        held_rows = pd.read_csv(args.eval_csv)
        fit_rows = fit_rows[fit_rows["split"].astype(str).str.lower() == args.split]
        held_rows = held_rows[held_rows["split"].astype(str).str.lower() == args.eval_split]
        for identity in ("sample_id", "source_id", "sha256", "speaker_id"):
            if identity in fit_rows and identity in held_rows:
                fit_ids = set(fit_rows[identity].dropna().astype(str)) - {""}
                held_ids = set(held_rows[identity].dropna().astype(str)) - {""}
                if fit_ids & held_ids:
                    raise ValueError(f"Calibration/evaluation overlap detected in {identity}.")

    scores, labels = load_data(args.predictions_csv, args.split, require_split=True)
    logger.info(
        "Loaded %d samples from %s (%d fake, %d real)",
        len(scores),
        args.predictions_csv,
        np.sum(labels == 1),
        np.sum(labels == 0),
    )

    if len(np.unique(labels)) < 2:
        logger.error("Data must contain both classes (0 and 1) to fit calibration.")
        sys.exit(1)

    # Uncertainty band
    uncertain_band = (args.uncertain_lower, args.uncertain_upper)
    if args.uncertain_lower >= args.uncertain_upper:
        uncertain_band = None

    calibrator = ScoreCalibrator(
        method=args.method,
        model_id=args.model_id,
        version=args.version,
        score_orientation=args.score_orientation,
        operating_threshold=args.operating_threshold,
        uncertain_band=uncertain_band,
    )

    # Pre-calibration ECE (using naive min-max scaling for comparison)
    min_s, max_s = np.min(scores), np.max(scores)
    if args.score_orientation == "higher_is_bonafide":
        naive_probs = 1.0 - (scores - min_s) / max(max_s - min_s, 1e-6)
    else:
        naive_probs = (scores - min_s) / max(max_s - min_s, 1e-6)
    uncal_ece = compute_ece(naive_probs, labels)

    logger.info("Fitting '%s' calibration...", args.method)
    calibrator.fit(scores, labels)
    fit_df = pd.read_csv(args.predictions_csv)
    fit_df = fit_df[fit_df["split"].astype(str).str.lower() == args.split]
    calibrator.fit_data = {
        "source_file": str(args.predictions_csv),
        "dataset": sorted(fit_df["dataset"].dropna().astype(str).unique().tolist()) if "dataset" in fit_df else [],
        "split": args.split,
        "sample_count": int(len(scores)),
        "protocol": args.prediction_protocol,
        "score_level": "recording" if args.prediction_protocol == "studio_sliding_window" else "native_input",
        "pipeline_signature": pipeline.get("pipeline_signature"),
    }

    metrics = calibrator.evaluate(scores, labels)
    logger.info("=== Calibration Results (Fitting Set) ===")
    logger.info("  Uncalibrated ECE (naive min-max): %.4f", uncal_ece)
    logger.info("  Calibrated ECE:                   %.4f", metrics["ece"])
    logger.info("  Brier Score:                      %.4f", metrics["brier_score"])
    logger.info("  Log Loss:                         %.4f", metrics["log_loss"])
    logger.info("  Accuracy @ %.2f:                   %.2f%%", args.operating_threshold, metrics["accuracy"] * 100)
    logger.info("  Fitted Parameters:                %s", calibrator.params)

    # Evaluate on held-out if provided
    if args.eval_csv and args.eval_csv.is_file():
        eval_scores, eval_labels = load_data(args.eval_csv, args.eval_split, require_split=True)
        eval_metrics = calibrator.evaluate(eval_scores, eval_labels)
        eval_df = pd.read_csv(args.eval_csv)
        eval_df = eval_df[eval_df["split"].astype(str).str.lower() == args.eval_split]
        calibrator.held_out_evaluation = {
            "source_file": str(args.eval_csv),
            "dataset": sorted(eval_df["dataset"].dropna().astype(str).unique().tolist()) if "dataset" in eval_df else [],
            "split": args.eval_split,
            "metrics": eval_metrics,
        }
        logger.info("=== Calibration Generalization (Held-out Set) ===")
        logger.info("  Held-out ECE:         %.4f", eval_metrics["ece"])
        logger.info("  Held-out Brier Score: %.4f", eval_metrics["brier_score"])
        logger.info("  Held-out Log Loss:    %.4f", eval_metrics["log_loss"])
        logger.info("  Held-out Accuracy:    %.2f%%", eval_metrics["accuracy"] * 100)

    # Save output artifact
    out_path = args.output
    if not out_path:
        out_dir = REPO_ROOT / "configs" / "calibration"
        out_path = out_dir / f"{args.model_id}_{args.method}.json"

    if not hasattr(calibrator, "held_out_evaluation"):
        calibrator.held_out_evaluation = None
    calibrator.save(out_path)
    logger.info("Saved calibration artifact to: %s", out_path)


if __name__ == "__main__":
    main()
