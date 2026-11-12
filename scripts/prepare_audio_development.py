"""Audit waveform training/dev/test data and reserve the eight local eval clips.

This command never assigns the reserved clips to training or development. It
checks existing splits rather than randomly splitting derivatives of speakers.
Manifest paths are absolute or relative to the source manifest's directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def audit(manifests, reserved_root):
    reserved = []
    for folder, label in (("Real", 0), ("Fake", 1)):
        for path in sorted((reserved_root / folder).glob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                reserved.append({"path": str(path.resolve()), "label": label, "split": "eval",
                                 "sha256": file_hash(path), "reserved": True})
    if not reserved:
        raise ValueError("Reserved evaluation audio was not found; refusing to create an empty exclusion list.")
    reserved_hashes = {row["sha256"] for row in reserved}
    problems, counts = Counter(), Counter()
    examples, identities, hashes = [], {}, {}
    split_classes = defaultdict(Counter)
    for manifest in manifests:
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                split = row.get("split", "").lower()
                if split == "test":
                    split = "eval"
                label = row.get("label", "")
                counts[split] += 1
                if split not in {"train", "val", "dev", "eval"} or label not in {"0", "1"}:
                    problems["invalid_split_or_label"] += 1
                    continue
                path = Path(row.get("path", ""))
                path = path if path.is_absolute() else manifest.parent / path
                if path.suffix.lower() not in AUDIO_EXTENSIONS:
                    problems["feature_cache_instead_of_waveform"] += 1
                    continue
                if not path.is_file():
                    problems["missing_audio_file"] += 1
                    if len(examples) < 5:
                        examples.append(str(path))
                    continue
                digest = file_hash(path)
                if digest in reserved_hashes and split != "eval":
                    problems["reserved_eval_clip_in_training_or_dev"] += 1
                previous_split = hashes.setdefault(digest, split)
                if previous_split != split:
                    problems["audio_hash_overlap_between_splits"] += 1
                for key in ("speaker_id", "source_id"):
                    value = row.get(key, "").strip()
                    if not value or value.startswith(("speaker_unknown", "asv2019_")):
                        problems[f"missing_real_{key}"] += 1
                        continue
                    identity = (key, value)
                    previous_split = identities.setdefault(identity, split)
                    if previous_split != split:
                        problems[f"{key}_overlap_between_splits"] += 1
                split_classes[split][label] += 1
    for split in ("train", "val", "dev", "eval"):
        if set(split_classes[split]) != {"0", "1"}:
            problems[f"{split}_needs_both_classes_of_raw_audio"] += 1
    return {
        "ready_for_waveform_training": not problems,
        "source_manifests": [str(p) for p in manifests],
        "manifest_rows_by_split": dict(counts),
        "usable_waveforms_by_split_and_label": {k: dict(v) for k, v in split_classes.items()},
        "problems": dict(problems), "missing_examples": examples,
        "reserved_eval_clips": reserved,
        "notes": ["Authenticity labels require verified source provenance; filename/folder labels are provisional.",
                  "Hashes catch exact duplicates. Speaker and source IDs must also cover re-encodes, crops and augmentations."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, default=[
        ROOT / "data" / "manifests" / f"manifest_asvspoof2019_{split}.csv" for split in ("train", "dev", "eval")])
    parser.add_argument("--reserved-root", type=Path, default=ROOT.parent / "initial setup" / "deepfake_phase1" / "Audios")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "audio_development_readiness.json")
    args = parser.parse_args()
    report = audit(args.manifests, args.reserved_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "reserved_eval_clips"}, indent=2))
    print(f"Reserved {len(report['reserved_eval_clips'])} evaluation clips; report: {args.output}")


if __name__ == "__main__":
    main()
