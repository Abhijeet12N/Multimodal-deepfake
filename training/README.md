# Model Training Suite

This folder contains the complete, production-grade training, auditing, and evaluation pipelines for the Multimodal Deepfake Detection project.

## Overview of Modules

| Script | Purpose |
|---|---|
| `train_multimodal.py` | Complete multimodal multi-task training pipeline (Xception + Wav2Vec 2.0 + SyncStream + late fusion). |
| `train_audio.py` | Standalone sliding-window audio deepfake classifier training pipeline. |
| `dataset_audit.py` | Integrity auditor for FakeAVCeleb dataset (scans durations, corruption, video/audio stream availability). |
| `smoke_test.py` | Quick 1-batch pre-flight sanity test for the training loader and model forward pass. |
| `syncnet_test.py` | Standalone audiovisual synchronization test across FakeAVCeleb categories. |
| `syncnet_score_testset.py` | Calculates lip-sync confidence scores across the test split. |
| `post_processing.py` | Threshold optimization, test-time augmentation (TTA), and temperature scaling. |
| `cross_attention.py` | Advanced cross-attention multimodal fusion architecture variant. |
| `live_monitor.py` | Generates training curve plots in real-time from `train.log`. |
| `run_train.sh` | Bash script for running training with automatic Google Drive synchronization. |

## Quick Start: Multimodal Training

### 1. Environment Variables

Set the dataset location and your output working directory:

```powershell
# Windows PowerShell
$env:DATASET_ROOT = "D:\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
$env:WORK_DIR = "D:\deepfake-work"
```

```bash
# Linux / macOS / Colab
export DATASET_ROOT="/content/FakeAVCeleb_v1.2"
export WORK_DIR="/content/work"
```

### 2. Dataset Scan and Subject-Disjoint Splitting

Scans the 4 FakeAVCeleb categories (`RealVideo-RealAudio`, `FakeVideo-RealAudio`, `RealVideo-FakeAudio`, `FakeVideo-FakeAudio`), filters corrupt media, and generates subject-disjoint train/val/test splits:

```bash
python training/train_multimodal.py --stage scan
```

Outputs generated in `$WORK_DIR`:
- `meta_train.csv` (~70% subjects)
- `meta_val.csv` (~10% subjects)
- `meta_test.csv` (~20% subjects)

### 3. Model Training

Trains the multi-task model with 3 prediction heads (`video`, `audio`, `any`):

```bash
python training/train_multimodal.py --stage train --epochs 12 --batch 24 --workers 4
```

Checkpoints and metrics are saved to `$WORK_DIR/best_model.pt`.

### 4. Evaluation and Ablations

Evaluate the trained checkpoint on the held-out test set:

```bash
python training/train_multimodal.py --stage eval --batch 24 --workers 4
```

Run ablation studies comparing individual streams (Image-only, Audio-only, Sync-only, Multimodal Fusion):

```bash
python training/train_multimodal.py --stage ablation
```

## Quick Start: Standalone Audio Training

The standalone audio classifier trains on full-duration audio recordings using sliding 3.0-second windows with 1.5-second overlap:

```bash
# Extract Wav2Vec 2.0 embeddings
python training/train_audio.py --stage extract --extract-batch 8

# Train sliding-window classifier
python training/train_audio.py --stage train --epochs 30 --batch 256 --patience 5
```

The resulting model is saved to `$WORK_DIR/audio_model.pt`.

## Guides

- Detailed Google Colab setup: [`colab_quickstart.md`](colab_quickstart.md)
- Standalone audio training walkthrough: [`audio_only_quickstart.md`](audio_only_quickstart.md)

