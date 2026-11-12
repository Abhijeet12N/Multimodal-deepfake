"""Package current audio code and generate a Colab notebook without local datasets/secrets."""
from pathlib import Path
import json
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "id": hashlib.sha256(source.encode()).hexdigest()[:12],
              "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    archive = ROOT / "artifacts" / "audio_colab_bundle.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    upstream = ROOT.parent / "aasist"
    paths = list((ROOT / "src").rglob("*.py")) + [
        ROOT / "training" / "finetune_aasist.py", ROOT / "scripts" / "fit_calibration.py",
        ROOT / "scripts" / "download_asvspoof2019.py",
        ROOT / "reports" / "audio_development_readiness.json",
    ]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        for path in paths:
            if "__pycache__" not in path.parts:
                out.write(path, "Multimodal-deepfake/" + path.relative_to(ROOT).as_posix())
        for name in ("models/AASIST.py", "models/weights/AASIST.pth", "models/weights/AASIST-L.pth", "LICENSE", "NOTICE"):
            path = upstream / name
            if path.is_file():
                out.write(path, "aasist/" + name)
    cells = [
        cell("markdown", """# AASIST audio upgrade in Colab

Run cells in order with a **GPU runtime**. The notebook downloads raw training
audio directly onto Colab's local disk; no dataset on Drive is needed.
Upload `audio_colab_bundle.zip` generated from your current project. It contains
audio code, official AASIST checkpoints and the eight local evaluation hashes;
it contains no `.env`, user uploads or local evaluation audio.

This workflow uses AASIST/AASIST-L, with no Fairseq. It trains on `train`, selects
weights on `val`, calibrates recording scores on separate `dev`, and checks `eval`
once. It does not select thresholds using your eight local clips. The notebook
has been generated and syntax checked locally; GPU training must be run here.

Use this updated notebook together with its updated ZIP. The AASIST path does
not import Transformers or Hugging Face Hub, and the setup cell does not change
them. This avoids downgrading Colab's Hub dependency used by Gradio/Diffusers.
"""),
        cell("code", """# AASIST uses raw waveforms; keep Colab's existing Hugging Face packages.
%pip install -q "librosa>=0.10,<1" "soundfile>=0.12,<1" "soxr>=0.3,<2" "scikit-learn>=1.3,<2"
import torch
assert torch.cuda.is_available(), "Choose Runtime > Change runtime type > GPU"
print(torch.__version__, torch.cuda.get_device_name(0))
"""),
        cell("code", """from google.colab import files
from pathlib import Path
import zipfile, json, csv, hashlib, subprocess, sys
from datetime import datetime

uploaded = files.upload()  # Select audio_colab_bundle.zip from your machine.
bundle_name = next(name for name in uploaded if name.endswith('.zip'))
workspace = Path('/content/audio_upgrade')
workspace.mkdir(exist_ok=True)
with zipfile.ZipFile(bundle_name) as bundle:
    for member in bundle.infolist():
        target = (workspace / member.filename).resolve()
        assert target.is_relative_to(workspace.resolve()), "Unsafe archive member"
    bundle.extractall(workspace)
project = workspace / 'Multimodal-deepfake'
# Verify the uploaded ZIP is updated and AASIST has no HF dependency.
check = [sys.executable, '-c',
         "import sys; sys.path.insert(0, 'src'); "
         "from deepfake_detector.audio_detection import get_detector; "
         "assert 'transformers' not in sys.modules; "
         "assert 'huggingface_hub' not in sys.modules; "
         "print('AASIST imports passed without Transformers or Hugging Face Hub')"]
subprocess.run(check, cwd=project, check=True)
"""),
        cell("markdown", """## Download the raw dataset onto Colab

The default downloads only **ASVspoof 2019 Logical Access (LA)**: a 7.64 GB ZIP
from the [publisher's record](https://zenodo.org/records/6906306), containing all
train/dev/eval waveforms and protocols. No Kaggle token or Hugging Face login.

The download resumes partial files, checks the official size and MD5, checks
available disk space, safely extracts and validates all 121,461 waveform paths.
The ZIP is removed after successful validation to reclaim space. Re-running
the cell reuses the validated extracted dataset while this runtime exists.

Colab local storage is temporary. The default keeps results locally for download
at the end; optionally enable Drive for **output backups only** below.
"""),
        cell("code", """DOWNLOAD_ROOT = Path('/content/datasets/asvspoof2019')
CUSTOM_MANIFEST = None  # Advanced: a local waveform CSV instead of the official download.
SAVE_OUTPUTS_TO_DRIVE = False  # True mounts Drive for results; the dataset stays on local disk.
MODELS = ['aasist', 'aasist_l']
EPOCHS = 10
BATCH_SIZE = 16  # Reduce to 8 or 4 if the GPU runs out of memory.

if CUSTOM_MANIFEST is None:
    subprocess.run([sys.executable, '-u', str(project/'scripts/download_asvspoof2019.py'),
                    '--output-dir', str(DOWNLOAD_ROOT)], check=True)
    ready = json.loads((DOWNLOAD_ROOT/'dataset_ready.json').read_text())
    DATA_ROOT = Path(ready['data_root'])
    print('Validated dataset:', DATA_ROOT)
    print(json.dumps(ready['splits'], indent=2))
"""),
        cell("markdown", """## Prepare disjoint training and scoring splits

The official development speakers are split into separate `val` (epoch selection)
and `dev` (calibration) groups using a fixed seed. The original evaluation split
is reserved for final evaluation. Your eight local clips remain excluded.

For an advanced custom manifest, supply `sample_id,path,label,split,speaker_id,
source_id,dataset` with four disjoint splits (`train,val,dev,eval`). Labels are
0 = real and 1 = fake. Derivatives of each speaker/source stay in one split.
"""),
        cell("code", """if SAVE_OUTPUTS_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    results_home = Path('/content/drive/MyDrive/deepfake_audio_upgrade')
else:
    results_home = Path('/content/audio_upgrade_results')
OUTPUT_ROOT = results_home / datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)

if CUSTOM_MANIFEST is not None:
    manifest = Path(CUSTOM_MANIFEST)
    assert manifest.is_file()
else:
    protocol_dir = DATA_ROOT / 'ASVspoof2019_LA_cm_protocols'
    rows = []
    for split, suffix in [('train','train.trn'), ('dev','dev.trl'), ('eval','eval.trl')]:
        protocol = protocol_dir / f'ASVspoof2019.LA.cm.{suffix}.txt'
        assert protocol.is_file(), f'Protocol missing: {protocol}'
        for line in protocol.read_text().splitlines():
            speaker, sample, _, attack, key = line.split()
            assert key in ['bonafide','spoof']
            path = DATA_ROOT / f'ASVspoof2019_LA_{split}' / 'flac' / f'{sample}.flac'
            assert path.is_file(), f'Raw waveform missing: {path}'
            rows.append(dict(sample_id=sample, path=str(path), label=int(key=='spoof'),
                             split=split, speaker_id=speaker, source_id=sample,
                             dataset='asvspoof2019_la', attack_type=attack))
    # Keep val and calibration dev speakers separate; fixed split independent of scores.
    speakers = sorted({r['speaker_id'] for r in rows if r['split']=='dev'},
                      key=lambda s: hashlib.sha256(('42:'+s).encode()).hexdigest())
    assert len(speakers) >= 2
    validation_speakers = set(speakers[:len(speakers)//2])
    for row in rows:
        if row['split']=='dev' and row['speaker_id'] in validation_speakers:
            row['split']='val'
    manifest = project / 'audio_waveform_manifest.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
print('Manifest:', manifest)
print('Exports:', OUTPUT_ROOT)
"""),
        cell("markdown", """## Fine-tune, calibrate and evaluate

Both variants start from their official pretrained weights. Training applies
random crops, gain and noise to both classes. Representative compressed speech
and diverse fake generation methods still need to be included in the dataset;
this starter augmentation does not guarantee generalization to web audio.

Validation, calibration and evaluation use the production recording pipeline.
These stages can take substantial GPU time on the full dataset. Checkpoints and
CSV scores are written to `OUTPUT_ROOT`. Use a fresh output directory for another
run. With local output storage, download the exported ZIP before deleting the
Colab runtime. Optional Drive backups do not store the dataset.
"""),
        cell("code", """for model_id in MODELS:
    checkpoint_name = 'AASIST-L.pth' if model_id=='aasist_l' else 'AASIST.pth'
    command = [sys.executable, '-u', str(project/'training/finetune_aasist.py'),
               '--model', model_id, '--manifest', str(manifest),
               '--checkpoint', str(workspace/'aasist/models/weights'/checkpoint_name),
               '--reserved-hashes', str(project/'reports/audio_development_readiness.json'),
               '--output', str(OUTPUT_ROOT/model_id), '--epochs', str(EPOCHS),
               '--batch-size', str(BATCH_SIZE)]
    subprocess.run(command, check=True)
"""),
        cell("code", """export_path = Path('/content/trained_audio_models.zip')
with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in OUTPUT_ROOT.rglob('*'):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT_ROOT))
for model_id in MODELS:
    print(model_id, (OUTPUT_ROOT/model_id/'evaluation.json').read_text())
files.download(str(export_path))
"""),
        cell("markdown", """## Return to the local studio

Download `trained_audio_models.zip`. Keep both original checkpoints. For the
chosen model, set its checkpoint **path** to the exported fine-tuned `.pth`
(`AASIST_CHECKPOINT_PATH` for AASIST), and place its matching `aasist_platt.json`
or `aasist_l_platt.json` in `configs/calibration/`. Model choice stays in the UI.

The studio verifies model identity, checkpoint/preprocessing/aggregation
signature and development provenance before loading calibration. Select the
matching detector and restart Flask after installing files. AASIST-L checkpoint
path overrides use `AASIST_L_CHECKPOINT_PATH`. Evaluate your eight local clips
again after export; do not refit using their results.
"""),
    ]
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "GPU"}, "cells": cells}
    notebook_path = ROOT / "notebooks" / "AASIST_Colab_Upgrade.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Notebook: {notebook_path}")
    print(f"Source bundle: {archive} ({archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
