"""Download the official ASVspoof 2019 LA archive into a Colab/local workspace.

Uses public HTTPS downloads, checks the publisher's size/MD5, extracts safely,
and checks every protocol-referenced waveform. No Kaggle/Hugging Face login.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile

RECORD_URL = "https://zenodo.org/records/6906306"
SOURCES = [
    "https://zenodo.org/api/records/6906306/files/LA.zip/content",
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y",
]
# Verified against the publisher's Zenodo record/API, not a third-party mirror.
ARCHIVE_BYTES = 7640952520
ARCHIVE_MD5 = "30c98f11d8b2bc21f2c257bfd78bb5c5"
COUNTS = {"train": 25380, "dev": 24844, "eval": 71237}
PROTOCOL_SUFFIXES = {"train": "train.trn", "dev": "dev.trl", "eval": "eval.trl"}


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_resumable(url, destination, expected_bytes, retries=5, opener=urllib.request.urlopen, delay=5):
    """Resume partial files; restart safely when the server ignores Range."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        offset = destination.stat().st_size if destination.exists() else 0
        if offset == expected_bytes:
            return
        if offset > expected_bytes:
            raise ValueError(f"Partial download is larger than the official archive: {destination}")
        headers = {"User-Agent": "ASVspoof-Research-Downloader/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener(request, timeout=60) as response:
                status = response.status
                if status not in (200, 206):
                    raise ValueError(f"Unexpected download HTTP status {status}")
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Invalid resume range: {content_range}")
                else:
                    offset = 0  # The server returned the whole archive, so overwrite the partial.
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type or "application/json" in content_type:
                    raise ValueError(f"Server returned {content_type}, not a ZIP archive")
                mode = "ab" if offset else "wb"
                last_progress = 0.0
                with destination.open(mode) as stream:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        if offset + len(block) > expected_bytes:
                            raise ValueError("Download exceeds the official archive size")
                        stream.write(block)
                        offset += len(block)
                        if time.monotonic() - last_progress > 5:
                            print(f"Download: {offset / 1e9:.2f}/{expected_bytes / 1e9:.2f} GB "
                                  f"({100 * offset / expected_bytes:.1f}%)", flush=True)
                            last_progress = time.monotonic()
            if offset != expected_bytes:
                raise OSError(f"Download interrupted at {offset}/{expected_bytes} bytes")
            return
        except (OSError, ValueError, urllib.error.URLError, http.client.IncompleteRead) as exc:
            print(f"Download attempt {attempt + 1}/{retries}: {exc}", flush=True)
            if isinstance(exc, urllib.error.HTTPError) and exc.code in {401, 403, 404}:
                raise
            if attempt + 1 == retries:
                raise
            time.sleep(delay)


def extract_archive(archive_path, destination, reserve_bytes=2 * 1024 ** 3):
    """Validate members before extraction; ZIP CRC errors remain fatal."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"ZIP symlinks are not supported: {member.filename}")
        required = sum(member.file_size for member in members) + reserve_bytes
        if shutil.disk_usage(destination).free < required:
            raise OSError(f"Extraction needs {required / 1e9:.1f} GB free including working space.")
        print(f"Extracting {len(members):,} entries ({sum(m.file_size for m in members) / 1e9:.1f} GB)...", flush=True)
        for index, member in enumerate(members, 1):
            archive.extract(member, destination)
            if index % 10000 == 0:
                print(f"Extracted {index:,}/{len(members):,}", flush=True)


def find_dataset_root(destination):
    candidates = [destination / "LA", destination]
    candidates.extend(path.parent for path in destination.glob("*/ASVspoof2019_LA_cm_protocols"))
    for candidate in candidates:
        if (candidate / "ASVspoof2019_LA_cm_protocols").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"ASVspoof2019_LA_cm_protocols not found under {destination}")


def validate_dataset(root, expected_counts=None):
    expected_counts = COUNTS if expected_counts is None else expected_counts
    summary = {}
    for split, expected in expected_counts.items():
        protocol = root / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{PROTOCOL_SUFFIXES[split]}.txt"
        if not protocol.is_file():
            raise FileNotFoundError(f"Missing protocol: {protocol}")
        labels, seen = {"bonafide": 0, "spoof": 0}, set()
        for line in protocol.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) != 5 or fields[4] not in labels:
                raise ValueError(f"Invalid protocol line in {protocol}: {line}")
            sample = fields[1]
            if sample in seen or Path(sample).name != sample:
                raise ValueError(f"Invalid or repeated sample ID in {protocol}: {sample}")
            seen.add(sample)
            audio = root / f"ASVspoof2019_LA_{split}" / "flac" / f"{sample}.flac"
            if not audio.is_file() or audio.stat().st_size == 0:
                raise FileNotFoundError(f"Missing/empty waveform: {audio}")
            labels[fields[4]] += 1
        if len(seen) != expected or not all(labels.values()):
            raise ValueError(f"{split}: expected {expected} files with both classes; found {len(seen)}, {labels}")
        summary[split] = {"recordings": len(seen), **labels}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/datasets/asvspoof2019"))
    parser.add_argument("--keep-archive", action="store_true", help="Keep the 7.6 GB ZIP after successful extraction")
    parser.add_argument("--validate-only", action="store_true", help="Check an existing download without network access")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "dataset_ready.json"
    if args.validate_only or marker.is_file():
        root = find_dataset_root(args.output_dir)
        summary = validate_dataset(root)
        print(f"Dataset already available: {root}\n{json.dumps(summary, indent=2)}")
        return
    archive = args.output_dir / "LA.zip"
    partial = args.output_dir / "LA.zip.part"
    if not archive.is_file():
        remaining = max(0, ARCHIVE_BYTES - (partial.stat().st_size if partial.exists() else 0))
        if shutil.disk_usage(args.output_dir).free < remaining + 2 * 1024 ** 3:
            raise OSError("Not enough space to download the archive; free Colab disk space first.")
        for index, source in enumerate(SOURCES):
            try:
                download_resumable(source, partial, ARCHIVE_BYTES)
                break
            except (OSError, ValueError, http.client.IncompleteRead) as exc:
                if index + 1 == len(SOURCES):
                    raise
                print(f"Trying official alternate download source after: {exc}", flush=True)
        print("Verifying publisher MD5...", flush=True)
        if partial.stat().st_size != ARCHIVE_BYTES or md5_file(partial) != ARCHIVE_MD5:
            raise ValueError(f"Archive checksum failed. Remove only {partial} and rerun the download cell.")
        partial.replace(archive)
    elif archive.stat().st_size != ARCHIVE_BYTES or md5_file(archive) != ARCHIVE_MD5:
        raise ValueError(f"Existing archive checksum failed: {archive}")
    extract_archive(archive, args.output_dir)
    root = find_dataset_root(args.output_dir)
    summary = validate_dataset(root)
    marker.write_text(json.dumps({"data_root": str(root), "source_record": RECORD_URL,
                                  "archive_md5": ARCHIVE_MD5, "archive_bytes": ARCHIVE_BYTES,
                                  "splits": summary}, indent=2), encoding="utf-8")
    if not args.keep_archive:
        archive.unlink()  # Fixed downloaded file, after checksum/extraction/protocol validation.
    print(f"Dataset ready: {root}\n{json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
