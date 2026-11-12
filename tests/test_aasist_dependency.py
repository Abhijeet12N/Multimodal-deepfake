"""Source discovery must work with submodules and older exported workspaces."""
from pathlib import Path

import pytest
import torch

from deepfake_detector.upstream import default_aasist_checkpoint, find_aasist_source
from deepfake_detector.audio_detection.detectors import aasist


def make_source(path):
    model = path / "models" / "AASIST.py"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("# source presence fixture\n")


def test_submodule_source_takes_precedence_over_sibling_checkout(tmp_path):
    root = tmp_path / "project"
    sibling = tmp_path / "aasist"
    make_source(sibling)
    assert find_aasist_source(root) == sibling.resolve()
    submodule = root / "third_party" / "aasist"
    submodule.mkdir(parents=True)
    assert find_aasist_source(root) == sibling.resolve()  # Uninitialized directory is insufficient.
    make_source(submodule)
    assert find_aasist_source(root) == submodule.resolve()


def test_older_colab_layout_is_supported(tmp_path):
    source = tmp_path / "project" / "aasist"
    make_source(source)
    assert find_aasist_source(tmp_path / "project") == source.resolve()


def test_checkpoint_lookup_prefers_finetuned_then_submodule_then_sibling(tmp_path):
    root = tmp_path / "project"
    submodule = root / "third_party" / "aasist" / "models" / "weights" / "AASIST.pth"
    sibling = tmp_path / "aasist" / "models" / "weights" / "AASIST.pth"
    assert default_aasist_checkpoint(root, "aasist") == submodule
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"sibling fixture")
    assert default_aasist_checkpoint(root, "aasist") == sibling
    submodule.parent.mkdir(parents=True)
    submodule.write_bytes(b"submodule fixture")
    assert default_aasist_checkpoint(root, "aasist") == submodule
    installed = root / "models" / "aasist_finetuned.pth"
    installed.parent.mkdir()
    installed.write_bytes(b"fine-tuned fixture")
    assert default_aasist_checkpoint(root, "aasist") == installed
    with pytest.raises(ValueError, match="Unsupported"):
        default_aasist_checkpoint(root, "unexpected")


def test_missing_source_error_explains_submodule_initialization(monkeypatch):
    monkeypatch.setattr(aasist, "AASIST_UPSTREAM_DIR", None)
    with pytest.raises(FileNotFoundError, match="git submodule update --init --recursive"):
        aasist._load_upstream_aasist_model({}, torch.device("cpu"))
