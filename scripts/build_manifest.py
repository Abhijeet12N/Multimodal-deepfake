"""Standardized Dataset Manifest Builder.

Constructs unified CSV manifests adhering to the project schema:
sample_id,path,label,dataset,split,speaker_id,source_id,attack_type,language

Application-wide label convention:
0 = genuine (bonafide)
1 = fake (spoof)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Required schema columns in canonical order
MANIFEST_COLUMNS: list[str] = [
    "sample_id",
    "path",
    "label",
    "dataset",
    "split",
    "speaker_id",
    "source_id",
    "attack_type",
    "language",
]


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute SHA-256 hash of a file."""
    p = Path(filepath)
    if not p.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_manifest_row(row: dict[str, Any]) -> None:
    """Validate a single manifest record against the strict project schema."""
    for col in MANIFEST_COLUMNS:
        if col not in row:
            raise ValueError(f"Missing required manifest column: {col}")
    
    label = row["label"]
    if label not in (0, 1, "0", "1"):
        raise ValueError(f"Invalid label {label!r} for sample {row.get('sample_id')}. Must be 0 or 1.")
    
    if not str(row["sample_id"]).strip():
        raise ValueError("sample_id cannot be empty")


def write_manifest_csv(rows: Sequence[dict[str, Any]], output_path: Path | str) -> Path:
    """Write validated manifest rows to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for r in rows:
        validate_manifest_row(r)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows:
            # Enforce clean string/int types
            row_dict = {
                "sample_id": str(r["sample_id"]).strip(),
                "path": str(r["path"]).strip(),
                "label": int(r["label"]),
                "dataset": str(r["dataset"]).strip(),
                "split": str(r["split"]).strip(),
                "speaker_id": str(r["speaker_id"]).strip(),
                "source_id": str(r["source_id"]).strip(),
                "attack_type": str(r["attack_type"]).strip(),
                "language": str(r["language"]).strip(),
            }
            writer.writerow(row_dict)

    return out


def build_asvspoof_features_manifest(
    features_dir: Path | str,
    splits: Sequence[str] = ("train", "dev", "eval"),
    protocol_dir: Optional[Path | str] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build manifests from pre-extracted ASVspoof 2019 LA features."""
    import numpy as np

    feat_dir = Path(features_dir)
    if not feat_dir.is_dir():
        raise FileNotFoundError(f"Features directory not found: {feat_dir}")

    split_rows: dict[str, list[dict[str, Any]]] = {}

    for split in splits:
        fn_path = feat_dir / f"{split}_wav2vec_filenames.txt"
        lbl_path = feat_dir / f"{split}_wav2vec_labels.npy"
        atk_path = feat_dir / f"{split}_wav2vec_attacks.npy"
        feat_path = feat_dir / f"{split}_wav2vec_features.npy"

        if not fn_path.exists() or not lbl_path.exists():
            print(f"[build_manifest] Skipping split '{split}': files not found in {feat_dir}")
            continue

        with open(fn_path, "r", encoding="utf-8") as f:
            filenames = [line.strip() for line in f if line.strip()]

        labels = np.load(lbl_path)
        attacks = np.load(atk_path) if atk_path.exists() else None

        if len(filenames) != len(labels):
            raise ValueError(f"Length mismatch in {split}: {len(filenames)} filenames vs {len(labels)} labels")

        # Optional protocol parsing for speaker IDs
        speaker_map: dict[str, str] = {}
        if protocol_dir is not None:
            prot_dir = Path(protocol_dir)
            prot_files = list(prot_dir.glob(f"*{split}*.txt"))
            if prot_files:
                with open(prot_files[0], "r", encoding="utf-8") as pf:
                    for line in pf:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            speaker_map[parts[1]] = parts[0]

        rows: list[dict[str, Any]] = []
        for i, (fn, lbl) in enumerate(zip(filenames, labels)):
            atk = str(attacks[i]) if attacks is not None else ("-" if int(lbl) == 0 else "unknown")
            spk = speaker_map.get(fn, f"asv2019_{split}_spk")
            rows.append({
                "sample_id": fn,
                "path": str(feat_path) if feat_path.exists() else fn,
                "label": int(lbl),
                "dataset": "asvspoof2019_la",
                "split": split,
                "speaker_id": spk,
                "source_id": fn,
                "attack_type": atk,
                "language": "en",
            })

        split_rows[split] = rows
        print(f"[build_manifest] ASVspoof 2019 LA split '{split}': {len(rows)} samples "
              f"({sum(1 for r in rows if r['label'] == 0)} bonafide, {sum(1 for r in rows if r['label'] == 1)} spoof)")

    return split_rows


def build_local_audio_manifest(
    audio_dir: Path | str,
    dataset_name: str = "local_eval",
    split: str = "eval",
) -> list[dict[str, Any]]:
    """Scan local directory of audio files and construct manifest."""
    root = Path(audio_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {root}")

    valid_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    rows: list[dict[str, Any]] = []

    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in valid_exts:
            rel = path.relative_to(root)
            parts = [p.lower() for p in rel.parts]

            # Determine label from subdirectories or filename
            if any("fake" in p or "spoof" in p for p in parts):
                label = 1
                attack_type = "voice_clone_or_deepfake"
            elif any("real" in p or "bonafide" in p for p in parts):
                label = 0
                attack_type = "-"
            else:
                # Default heuristic based on known file names in repo
                if "jay_shetty_vocals" in path.name.lower():
                    label = 1
                    attack_type = "voice_clone"
                else:
                    label = 0
                    attack_type = "-"

            # Speaker heuristic
            name_lower = path.name.lower()
            if "jay_shetty" in name_lower:
                speaker_id = "jay_shetty"
            elif "zelensky" in name_lower:
                speaker_id = "volodymyr_zelensky"
            elif "aliabhatt" in name_lower or "alia bhatt" in name_lower:
                speaker_id = "alia_bhatt"
            else:
                speaker_id = "speaker_unknown"

            # Language heuristic
            if "zelensky" in name_lower:
                language = "uk"
            else:
                language = "en"

            sample_id = path.stem
            rows.append({
                "sample_id": sample_id,
                "path": str(path.resolve()),
                "label": int(label),
                "dataset": dataset_name,
                "split": split,
                "speaker_id": speaker_id,
                "source_id": sample_id,
                "attack_type": attack_type,
                "language": language,
            })

    print(f"[build_manifest] Local audio dir '{root}': {len(rows)} audio files "
          f"({sum(1 for r in rows if r['label'] == 0)} genuine, {sum(1 for r in rows if r['label'] == 1)} fake)")
    return rows


def generate_dataset_summary(all_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Generate comprehensive machine-readable dataset summary."""
    summary: dict[str, Any] = {
        "total_samples": len(all_rows),
        "datasets": {},
        "splits": {},
        "labels": {"genuine_count": 0, "fake_count": 0, "fake_ratio": 0.0},
        "attack_types": {},
        "speakers": {"total_unique_speakers": 0, "speakers_by_dataset": {}},
        "languages": {},
    }

    speakers_all: set[str] = set()

    for r in all_rows:
        ds = r["dataset"]
        sp = r["split"]
        lbl = int(r["label"])
        atk = r["attack_type"]
        spk = r["speaker_id"]
        lang = r["language"]

        # Dataset breakdown
        if ds not in summary["datasets"]:
            summary["datasets"][ds] = {"total": 0, "genuine": 0, "fake": 0, "splits": {}}
        summary["datasets"][ds]["total"] += 1
        if lbl == 0:
            summary["datasets"][ds]["genuine"] += 1
        else:
            summary["datasets"][ds]["fake"] += 1

        if sp not in summary["datasets"][ds]["splits"]:
            summary["datasets"][ds]["splits"][sp] = {"total": 0, "genuine": 0, "fake": 0}
        summary["datasets"][ds]["splits"][sp]["total"] += 1
        if lbl == 0:
            summary["datasets"][ds]["splits"][sp]["genuine"] += 1
        else:
            summary["datasets"][ds]["splits"][sp]["fake"] += 1

        # Global split breakdown
        if sp not in summary["splits"]:
            summary["splits"][sp] = {"total": 0, "genuine": 0, "fake": 0}
        summary["splits"][sp]["total"] += 1
        if lbl == 0:
            summary["splits"][sp]["genuine"] += 1
        else:
            summary["splits"][sp]["fake"] += 1

        # Global labels
        if lbl == 0:
            summary["labels"]["genuine_count"] += 1
        else:
            summary["labels"]["fake_count"] += 1

        # Attack types
        summary["attack_types"][atk] = summary["attack_types"].get(atk, 0) + 1

        # Speakers
        speakers_all.add(spk)
        if ds not in summary["speakers"]["speakers_by_dataset"]:
            summary["speakers"]["speakers_by_dataset"][ds] = set()
        summary["speakers"]["speakers_by_dataset"][ds].add(spk)

        # Languages
        summary["languages"][lang] = summary["languages"].get(lang, 0) + 1

    if summary["total_samples"] > 0:
        summary["labels"]["fake_ratio"] = round(summary["labels"]["fake_count"] / summary["total_samples"], 4)

    summary["speakers"]["total_unique_speakers"] = len(speakers_all)
    # Convert sets to counts for clean JSON serialization
    summary["speakers"]["speakers_by_dataset"] = {
        ds: len(spks) for ds, spks in summary["speakers"]["speakers_by_dataset"].items()
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Build standardized dataset manifests for audio deepfake detection.")
    parser.add_argument("--features_dir", type=str, default="../initial setup/deepfake_phase1/features_wav2vec",
                        help="Path to pre-extracted ASVspoof 2019 features")
    parser.add_argument("--audio_dir", type=str, default="../initial setup/deepfake_phase1/Audios",
                        help="Path to local audio evaluation files")
    parser.add_argument("--output_dir", type=str, default="data/manifests",
                        help="Output directory for CSV manifests")
    parser.add_argument("--summary_path", type=str, default="reports/dataset_summary.json",
                        help="Path to output dataset summary JSON")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []

    # 1. Process ASVspoof 2019 pre-extracted features
    feat_path = Path(args.features_dir)
    if not feat_path.is_absolute():
        feat_path = (Path.cwd() / feat_path).resolve()

    if feat_path.exists():
        asv_splits = build_asvspoof_features_manifest(feat_path)
        for split_name, rows in asv_splits.items():
            split_out = out_dir / f"manifest_asvspoof2019_{split_name}.csv"
            write_manifest_csv(rows, split_out)
            all_rows.extend(rows)
            print(f"[build_manifest] Saved: {split_out}")
    else:
        print(f"[build_manifest] Warning: features_dir not found at {feat_path}")

    # 2. Process local audio evaluation directory
    aud_path = Path(args.audio_dir)
    if not aud_path.is_absolute():
        aud_path = (Path.cwd() / aud_path).resolve()

    if aud_path.exists():
        local_rows = build_local_audio_manifest(aud_path, dataset_name="local_eval", split="eval")
        local_out = out_dir / "manifest_local_eval.csv"
        write_manifest_csv(local_rows, local_out)
        all_rows.extend(local_rows)
        print(f"[build_manifest] Saved: {local_out}")
    else:
        print(f"[build_manifest] Warning: audio_dir not found at {aud_path}")

    # 3. Write unified manifest
    all_out = out_dir / "manifest_all.csv"
    write_manifest_csv(all_rows, all_out)
    print(f"\n[build_manifest] Saved unified manifest ({len(all_rows)} total rows): {all_out}")

    # 4. Generate and save dataset summary
    summary = generate_dataset_summary(all_rows)
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[build_manifest] Saved dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
