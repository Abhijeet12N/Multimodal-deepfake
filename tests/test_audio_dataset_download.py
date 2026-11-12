"""Exercise resumable dataset download and archive validation without a large download."""
import io
from pathlib import Path
import stat
from types import SimpleNamespace
import zipfile

import pytest

from scripts.download_asvspoof2019 import (
    PROTOCOL_SUFFIXES, download_resumable, extract_archive,
    find_dataset_root, md5_file, validate_dataset,
)


class Response(io.BytesIO):
    def __init__(self, data, status=200, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Type": "application/octet-stream"}
        if content_range:
            self.headers["Content-Range"] = content_range


def test_partial_download_resumes_exact_bytes(tmp_path):
    path = tmp_path / "LA.zip.part"
    path.write_bytes(b"abc")
    def opener(request, timeout):
        assert request.get_header("Range") == "bytes=3-"
        return Response(b"defghij", 206, "bytes 3-9/10")
    download_resumable("https://example.test/LA.zip", path, 10, opener=opener)
    assert path.read_bytes() == b"abcdefghij"


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path):
    path = tmp_path / "LA.zip.part"
    path.write_bytes(b"abc")
    download_resumable("https://example.test/LA.zip", path, 10,
                       opener=lambda *a, **k: Response(b"abcdefghij", 200))
    assert path.read_bytes() == b"abcdefghij"


def test_connection_ending_early_retries_from_partial_offset(tmp_path):
    path = tmp_path / "LA.zip.part"
    requests = []
    def opener(request, timeout):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return Response(b"abcd")
        return Response(b"efghij", 206, "bytes 4-9/10")
    download_resumable("https://example.test/LA.zip", path, 10, retries=2, opener=opener, delay=0)
    assert requests == [None, "bytes=4-"]
    assert path.read_bytes() == b"abcdefghij"


def test_wrong_resume_range_is_rejected_without_changing_partial(tmp_path):
    path = tmp_path / "LA.zip.part"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="Invalid resume range"):
        download_resumable("https://example.test/LA.zip", path, 10, retries=1,
                           opener=lambda *a, **k: Response(b"defghij", 206, "bytes 1-7/10"))
    assert path.read_bytes() == b"abc"


def test_html_error_page_does_not_become_archive(tmp_path):
    path = tmp_path / "LA.zip.part"
    response = Response(b"error")
    response.headers["Content-Type"] = "text/html"
    with pytest.raises(ValueError, match="not a ZIP archive"):
        download_resumable("https://example.test/LA.zip", path, 5, retries=1,
                           opener=lambda *a, **k: response)
    assert not path.exists()


def test_md5_matches_published_algorithm(tmp_path):
    path = tmp_path / "sample"
    path.write_bytes(b"abc")
    assert md5_file(path) == "900150983cd24fb0d6963f7d28e17f72"


def test_extract_rejects_traversal_and_symlink(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escaped.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        extract_archive(archive, tmp_path / "data", reserve_bytes=0)
    assert not (tmp_path / "escaped.txt").exists()
    with zipfile.ZipFile(archive, "w") as stream:
        member = zipfile.ZipInfo("link")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        stream.writestr(member, "target")
    with pytest.raises(ValueError, match="symlinks"):
        extract_archive(archive, tmp_path / "data", reserve_bytes=0)


def test_extract_checks_disk_space_and_crc(tmp_path, monkeypatch):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as stream:
        stream.writestr("LA/sample.flac", b"payload")
    from scripts import download_asvspoof2019 as downloader
    with monkeypatch.context() as context:
        context.setattr(downloader.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
        with pytest.raises(OSError, match="Extraction needs"):
            extract_archive(archive, tmp_path / "data", reserve_bytes=0)
    archive.write_bytes(archive.read_bytes().replace(b"payload", b"PAYLOAD", 1))
    with pytest.raises(zipfile.BadZipFile):
        extract_archive(archive, tmp_path / "data", reserve_bytes=0)


def test_protocol_validation_checks_every_audio_path(tmp_path):
    root = tmp_path / "LA"
    protocols = root / "ASVspoof2019_LA_cm_protocols"
    protocols.mkdir(parents=True)
    for split in ("train", "dev", "eval"):
        folder = root / f"ASVspoof2019_LA_{split}" / "flac"
        folder.mkdir(parents=True)
        (folder / "real.flac").write_bytes(b"fLaC")
        (folder / "fake.flac").write_bytes(b"fLaC")
        (protocols / f"ASVspoof2019.LA.cm.{PROTOCOL_SUFFIXES[split]}.txt").write_text(
            "speaker real - - bonafide\nspeaker fake - A01 spoof\n")
    assert find_dataset_root(tmp_path) == root
    counts = {"train": 2, "dev": 2, "eval": 2}
    report = validate_dataset(root, counts)
    assert report["train"] == {"recordings": 2, "bonafide": 1, "spoof": 1}
    (root / "ASVspoof2019_LA_eval" / "flac" / "fake.flac").unlink()
    with pytest.raises(FileNotFoundError, match="Missing/empty waveform"):
        validate_dataset(root, counts)
