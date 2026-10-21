# Standalone audio detector quick start

The standalone audio checker is a separate model from the existing video detector:

- `best_model.pt` + `MODEL_PATH`: video, audio, and lip-sync multimodal analysis.
- `audio_model.pt` + `AUDIO_MODEL_PATH`: audio-only analysis.

The audio model uses pretrained Wav2Vec 2.0 as a frozen feature extractor and trains a separate fake/real classifier on FakeAVCeleb labels. It splits each complete recording into overlapping 3-second windows and averages the window scores for one recording-level result. Audio-only scores must be evaluated independently; the multimodal README metrics do not describe this classifier.

## Train in Colab

Use a GPU runtime. Keep the FakeAVCeleb dataset extracted in the runtime and clone this repository. Install the project dependencies. If pip fails on `facenet-pytorch`'s Pillow pin in a Python 3.13 runtime, the audio-only script does not import facenet-pytorch; you only need it if running the existing video dataset scan script.

Set the extracted dataset root to the directory directly containing the four category folders. Generate the subject-disjoint splits using the existing scan stage:

```python
import os
os.environ["DATASET_ROOT"] = "/content/data/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2"
os.environ["WORK_DIR"] = "/content/work"

%cd /content/multimodal-deepfake-detection
!python notebook_v2/deepfake_v3.py --stage scan
```

Then extract frozen Wav2Vec features from the full duration of every recording and train the audio-only classifier:

```python
!python notebook_v2/train_audio_only.py --stage extract --extract-batch 8
!python notebook_v2/train_audio_only.py --stage train --epochs 30 --batch 256 --patience 5
```

The `extract` stage reads `meta_train.csv`, `meta_val.csv`, and `meta_test.csv` from `/content/work`. It writes compact feature caches there. Full-duration extraction takes longer than the original first-3-second version. The `train` stage trains a window classifier, selects a threshold using validation scores aggregated per recording, prints held-out test metrics using the same aggregation, and writes:

```text
/content/work/audio_model.pt
```

Save this checkpoint to Drive before the Colab runtime ends:

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/deepfake-model
!cp /content/work/audio_model.pt /content/drive/MyDrive/deepfake-model/
```

If audio decoding fails for MP4 files, install ffmpeg in the Colab runtime (`!apt-get -qq update && apt-get -qq install -y ffmpeg`) and rerun feature extraction. The updated pipeline writes a different feature-cache format, so rerun `--stage extract` after replacing the script; caches from the old first-window pipeline cannot be reused.

## Use both checkers locally

Download `audio_model.pt` from Drive. Create/activate the project's Python environment, install `requirements.txt`, and start the demo with both checkpoint paths set. On Windows PowerShell:

```powershell
$env:MODEL_PATH = "C:\path\to\best_model.pt"
$env:AUDIO_MODEL_PATH = "C:\path\to\audio_model.pt"
cd ui
python app.py
```

Open `http://127.0.0.1:5000`. Use the video upload for the multimodal model and the separate audio upload for the audio-only model. Supported audio extensions are WAV, MP3, M4A, FLAC, OGG, and AAC. The audio checker analyzes the complete recording in overlapping windows, returns a recording-level mean score, and shows the highest-scoring time ranges. The first run downloads the pretrained Wav2Vec 2.0 weights if they are not cached locally.

## Labels and interpretation

Audio labels are derived from the FakeAVCeleb category names. The script preserves the subject-disjoint train/validation/test split created by `deepfake_v3.py --stage scan`. Inspect the held-out recording-level test AUC/F1 printed by the script before using the audio checker. Threshold calibration is only as representative as the validation recordings and their durations. Dataset-specific metrics do not guarantee performance on other sources or manipulation methods; include representative long real and fake recordings in evaluation before relying on long-audio decisions.
