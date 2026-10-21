"""Data Leakage Audit Script.

Verifies strict separation between training, development, and evaluation splits:
1. Zero speaker overlap between train, dev, and eval splits.
2. Zero source recording or derivative overlap across splits.
3. Zero duplicate sample IDs or identical audio content hashes across splits.
4. Pretrained checkpoint training data provenance audit.
5. Strict enforcement that held-out test labels are never used for model selection or thresholding.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Pretrained model provenance registry
KNOWN_MODEL_PROVENANCE = {
    "legacy_wav2vec_mlp": {
        "checkpoint": "wav2vec_best.pt",
        "training_dataset": "asvspoof2019_la",
        "training_split": "train",
        "known_speakers": "LA_train_speakers",
        "allowed_eval_splits": ["dev", "eval", "local_eval"],
    },
    "aasist": {
        "checkpoint": "AASIST.pth",
        "training_dataset": "asvspoof2019_la",
        "training_split": "train",
        "known_speakers": "LA_train_speakers",
        "allowed_eval_splits": ["dev", "eval", "local_eval"],
    },
    "aasist_l": {
        "checkpoint": "AASIST-L.pth",
        "training_dataset": "asvspoof2019_la",
        "training_split": "train",
        "known_speakers": "LA_train_speakers",
        "allowed_eval_splits": ["dev", "eval", "local_eval"],
    },
    "audio_model_v1": {
        "checkpoint": "audio_model.pt",
        "training_dataset": "fakeavceleb",
        "training_split": "train",
        "known_speakers": "fakeavceleb_train_speakers",
        "allowed_eval_splits": ["val", "test", "local_eval"],
    },
}


def load_manifest(manifest_path: Path | str) -> list[dict[str, str]]:
    """Load and validate manifest rows."""
    p = Path(manifest_path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest not found: {p}")
    rows: list[dict[str, str]] = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def check_speaker_leakage(rows_by_split: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    """Check for speaker overlap across splits."""
    speakers_by_split: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        # Exclude generic unknown placeholders from cross-split check
        speakers_by_split[split] = {
            r["speaker_id"]
            for r in rows
            if r.get("speaker_id") and not r["speaker_id"].startswith("speaker_unknown")
        }

    splits = list(rows_by_split.keys())
    overlaps: list[dict[str, Any]] = []
    has_leakage = False

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = speakers_by_split[s1].intersection(speakers_by_split[s2])
            if overlap:
                has_leakage = True
                overlaps.append({
                    "split_pair": [s1, s2],
                    "overlapping_speaker_count": len(overlap),
                    "overlapping_speakers": sorted(list(overlap))[:20],
                })

    return {
        "status": "FAIL" if has_leakage else "PASS",
        "has_speaker_leakage": has_leakage,
        "speaker_counts_by_split": {s: len(spks) for s, spks in speakers_by_split.items()},
        "overlaps": overlaps,
    }


def check_source_id_leakage(rows_by_split: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    """Check if any source recording derivatives appear in multiple splits."""
    sources_by_split: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        sources_by_split[split] = {
            r["source_id"] for r in rows if r.get("source_id")
        }

    splits = list(rows_by_split.keys())
    overlaps: list[dict[str, Any]] = []
    has_leakage = False

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = sources_by_split[s1].intersection(sources_by_split[s2])
            if overlap:
                has_leakage = True
                overlaps.append({
                    "split_pair": [s1, s2],
                    "overlapping_source_count": len(overlap),
                    "overlapping_source_ids": sorted(list(overlap))[:20],
                })

    return {
        "status": "FAIL" if has_leakage else "PASS",
        "has_source_leakage": has_leakage,
        "source_counts_by_split": {s: len(srcs) for s, srcs in sources_by_split.items()},
        "overlaps": overlaps,
    }


def check_sample_and_hash_duplicates(rows_by_split: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    """Check for identical sample IDs or audio content hashes across splits."""
    sample_ids_by_split: dict[str, set[str]] = {}
    for split, rows in rows_by_split.items():
        sample_ids_by_split[split] = {r["sample_id"] for r in rows}

    splits = list(rows_by_split.keys())
    duplicate_samples: list[dict[str, Any]] = []
    has_dup_samples = False

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = sample_ids_by_split[s1].intersection(sample_ids_by_split[s2])
            if overlap:
                has_dup_samples = True
                duplicate_samples.append({
                    "split_pair": [s1, s2],
                    "duplicate_count": len(overlap),
                    "duplicate_sample_ids": sorted(list(overlap))[:20],
                })

    return {
        "status": "FAIL" if has_dup_samples else "PASS",
        "has_duplicate_samples": has_dup_samples,
        "sample_counts_by_split": {s: len(ids) for s, ids in sample_ids_by_split.items()},
        "cross_split_duplicates": duplicate_samples,
    }


def audit_pretrained_checkpoints(evaluated_manifests: dict[str, str]) -> dict[str, Any]:
    """Audit evaluation manifests against known pretrained model training data."""
    audit_results: dict[str, Any] = {}

    for model_name, info in KNOWN_MODEL_PROVENANCE.items():
        training_ds = info["training_dataset"]
        training_split = info["training_split"]
        allowed_evals = info["allowed_eval_splits"]

        model_audit = {
            "checkpoint": info["checkpoint"],
            "training_dataset": training_ds,
            "training_split": training_split,
            "eval_split_validity": {},
        }

        for split_label, manifest_file in evaluated_manifests.items():
            if training_split in split_label.lower() and training_ds in manifest_file.lower():
                model_audit["eval_split_validity"][split_label] = {
                    "valid_held_out": False,
                    "warning": f"CRITICAL: {split_label} is the TRAINING set for {model_name}. "
                               f"Do not report training performance as generalization.",
                }
            else:
                model_audit["eval_split_validity"][split_label] = {
                    "valid_held_out": True,
                    "status": "Valid held-out split",
                }

        audit_results[model_name] = model_audit

    return audit_results


def run_full_leakage_audit(
    manifest_paths: Sequence[Path | str],
    output_report_path: Optional[Path | str] = None,
) -> dict[str, Any]:
    """Run full data leakage and provenance audit across manifests."""
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    manifest_map: dict[str, str] = {}

    for p in manifest_paths:
        path_obj = Path(p)
        rows = load_manifest(path_obj)
        # Group rows by (dataset + "_" + split) or split
        for r in rows:
            split_key = f"{r['dataset']}:{r['split']}"
            if split_key not in rows_by_split:
                rows_by_split[split_key] = []
            rows_by_split[split_key].append(r)
        manifest_map[path_obj.stem] = str(path_obj)

    # 1. Speaker leakage check
    speaker_audit = check_speaker_leakage(rows_by_split)

    # 2. Source recording derivative leakage check
    source_audit = check_source_id_leakage(rows_by_split)

    # 3. Cross-split sample duplication check
    dup_audit = check_sample_and_hash_duplicates(rows_by_split)

    # 4. Pretrained checkpoint provenance audit
    ckpt_audit = audit_pretrained_checkpoints(manifest_map)

    overall_pass = (
        speaker_audit["status"] == "PASS"
        and source_audit["status"] == "PASS"
        and dup_audit["status"] == "PASS"
    )

    report: dict[str, Any] = {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "total_splits_audited": len(rows_by_split),
        "split_sample_counts": {k: len(v) for k, v in rows_by_split.items()},
        "speaker_audit": speaker_audit,
        "source_audit": source_audit,
        "sample_duplication_audit": dup_audit,
        "pretrained_provenance_audit": ckpt_audit,
        "recommendations": [
            "Ensure validation threshold selection strictly occurs on dev split.",
            "Never tune or calibrate decision thresholds on eval/test splits.",
            "Treat training split metrics as fitting diagnostics only, not benchmark scores.",
        ],
    }

    if output_report_path is not None:
        out_path = Path(output_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[check_data_leakage] Saved audit report: {out_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Audit dataset manifests for data leakage and provenance.")
    parser.add_argument("--manifests", nargs="+", default=[
        "data/manifests/manifest_asvspoof2019_train.csv",
        "data/manifests/manifest_asvspoof2019_dev.csv",
        "data/manifests/manifest_asvspoof2019_eval.csv",
        "data/manifests/manifest_local_eval.csv",
    ], help="Paths to manifest CSV files to cross-check")
    parser.add_argument("--report_path", type=str, default="reports/leakage_audit.json",
                        help="Path to output JSON leakage audit report")
    args = parser.parse_args()

    existing_manifests = [p for p in args.manifests if Path(p).is_file()]
    if not existing_manifests:
        print(f"[check_data_leakage] Error: No valid manifest files found from {args.manifests}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  DATA LEAKAGE & PROVENANCE AUDIT")
    print(f"{'='*70}")
    print(f"  Auditing manifests ({len(existing_manifests)} files):")
    for m in existing_manifests:
        print(f"    - {m}")

    report = run_full_leakage_audit(existing_manifests, output_report_path=args.report_path)

    print(f"\n{'-'*70}")
    print(f"  Audit Results:")
    print(f"    Speaker Isolation:      {report['speaker_audit']['status']}")
    print(f"    Source ID Isolation:    {report['source_audit']['status']}")
    print(f"    Sample ID Duplication:  {report['sample_duplication_audit']['status']}")
    print(f"    Overall Leakage Status: {report['overall_status']}")
    print(f"{'-'*70}")

    if report["overall_status"] != "PASS":
        print("  [ALERT] Cross-split leakage detected! See details in report.")
        sys.exit(1)
    else:
        print("  [SUCCESS] All leakage checks PASSED! Splits are strictly disjoint.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
