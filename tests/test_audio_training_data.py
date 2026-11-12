"""Prevent evaluation leakage and label inversion in the Colab training path."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import pytest
import soundfile as sf

from training.finetune_aasist import TrainWindows, read_splits, sha256


def example_manifest(tmp_path, shared_speaker=False):
    rows = []
    for index, split in enumerate(("train", "val", "dev", "eval")):
        path = tmp_path / f"{split}.wav"
        sf.write(path, np.full(1600, 0.1 * (index + 1), dtype=np.float32), 16000, subtype="FLOAT")
        rows.append({"path": str(path), "label": str(index % 2), "split": split,
                     "speaker_id": "same" if shared_speaker else f"speaker_{index}",
                     "source_id": f"recording_{index}"})
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    return manifest, rows


def test_reserved_clip_is_rejected_before_training(tmp_path):
    manifest, rows = example_manifest(tmp_path)
    with pytest.raises(ValueError, match="Reserved evaluation clip"):
        read_splits(manifest, {sha256(rows[0]["path"])})


def test_same_speaker_in_validation_and_train_is_rejected(tmp_path):
    manifest, _ = example_manifest(tmp_path, shared_speaker=True)
    with pytest.raises(ValueError, match="Split leakage"):
        read_splits(manifest, set())


@pytest.mark.parametrize("label, upstream_target", [(0, 1), (1, 0)])
def test_aasist_training_padding_and_label_mapping(tmp_path, label, upstream_target):
    path = tmp_path / "speech.wav"
    sf.write(path, np.full(1600, 0.25, dtype=np.float32), 16000, subtype="FLOAT")
    wave, target = TrainWindows([{"path": str(path), "label": label}], augment=False)[0]
    assert wave.shape == (64600,)
    assert np.allclose(wave.numpy(), 0.25)  # Do not apply Wav2Vec normalization to AASIST.
    assert target == upstream_target


def test_importing_audio_package_does_not_import_video_dependencies():
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "from deepfake_detector.audio_detection import AASISTDetector; "
        "assert 'facenet_pytorch' not in sys.modules; "
        "assert 'timm' not in sys.modules; "
        "assert 'transformers' not in sys.modules; "
        "assert 'huggingface_hub' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True, timeout=60)


def test_aasist_inference_with_huggingface_imports_blocked(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "third_party" / "aasist"
    checkpoint = source / "models" / "weights" / "AASIST.pth"
    if not checkpoint.is_file():
        pytest.skip("Local official AASIST checkpoint is not installed")
    # Reproduce an isolated clone/bundle: no sibling repository is available.
    runtime = tmp_path / "project"
    shutil.copytree(root / "src", runtime / "src", ignore=shutil.ignore_patterns("__pycache__"))
    isolated_source = runtime / "third_party" / "aasist" / "models"
    isolated_source.mkdir(parents=True)
    shutil.copy2(source / "models" / "AASIST.py", isolated_source / "AASIST.py")
    isolated_checkpoint = isolated_source / "AASIST.pth"
    shutil.copy2(checkpoint, isolated_checkpoint)
    code = '''
import sys, importlib.abc, types
from pathlib import Path
class BlockHF(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'transformers', 'huggingface_hub', 'tokenizers'}:
            raise ImportError('Hugging Face imports intentionally unavailable: ' + fullname)
sys.meta_path.insert(0, BlockHF())
sys.path.insert(0, 'src')
# The old top-level models import would collide with this unrelated package.
sys.modules['models'] = types.ModuleType('models')
import torch
import numpy as np
torch.set_num_threads(2)
from deepfake_detector.audio_detection import get_detector
from deepfake_detector.audio_detection.detectors.aasist import AASIST_UPSTREAM_DIR
assert AASIST_UPSTREAM_DIR == (Path.cwd() / 'third_party' / 'aasist').resolve()
detector = get_detector('aasist', checkpoint_path=sys.argv[1], device='cpu')
wave = np.random.default_rng(42).normal(0, .05, (1, 64600)).astype(np.float32)
result = detector.predict_batch(wave)
assert result['raw_scores'].shape == (1, 2)
assert np.isfinite(result['bonafide_scores']).all()
assert 'transformers' not in sys.modules and 'huggingface_hub' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code, str(isolated_checkpoint)], cwd=runtime, check=True, timeout=60)
