# System Architecture

## High-Level Pipeline

```text
Input Media (Video / Audio)
  │
  ├─► Frame Sampling (8 frames) ──► MTCNN Face Detection ──► Xception Visual Stream ─────┐
  │                                                                                      │
  ├─► Audio Track (16 kHz Mono) ─► Frozen Wav2Vec 2.0 ────► Audio Projection Stream ────┼─► Late Fusion MLP Trunk
  │                                                                                      │     ├─► Video Manipulation Head
  └─► Lip Crops + Mel Spectrogram ───────────────────────► 3D/2D SyncStream ─────────────┘     ├─► Audio Manipulation Head
                                                                                               └─► Any Manipulation Head (Fusion)
```

## Modular Organization

The codebase is organized into dedicated, decoupled modules:

```text
multimodal-deepfake-detection/
├── src/deepfake_detector/      # Core reusable library
│   ├── config.py              # Centralized configuration & constants
│   ├── model.py               # PyTorch neural network architectures
│   ├── preprocessing.py       # Face detection, lip cropping & audio processing
│   ├── inference.py           # Multimodal video predictor & interpretation
│   ├── audio_inference.py     # Standalone sliding-window audio predictor
│   └── reporting.py           # Structured JSON report generator & risk scoring
├── training/                  # Dedicated model training suite
│   ├── train_multimodal.py    # Multi-task multimodal training pipeline
│   ├── train_audio.py         # Standalone audio classifier training pipeline
│   ├── dataset_audit.py       # Dataset integrity & corruption scanner
│   ├── smoke_test.py          # Fast pre-flight verification script
│   ├── syncnet_test.py        # Lip-sync consistency testing
│   ├── syncnet_score_testset.py# Dataset-wide SyncNet scoring
│   ├── post_processing.py     # Decision threshold calibration & TTA
│   ├── cross_attention.py     # Experimental cross-attention fusion architecture
│   ├── live_monitor.py        # Real-time loss & metric curve visualizer
│   └── run_train.sh           # Shell training automation launcher
├── ui/                        # Web dashboard interface
│   ├── app.py                 # Flask server & REST API
│   ├── templates/index.html   # Interactive UI template
│   ├── static/                # CSS stylesheet & client JavaScript
│   ├── samples/               # Local test media clips
│   └── uploads/               # User-uploaded media storage
├── tests/                     # Automated test suite (unittest & pytest)
│   ├── test_models.py         # Model architecture & forward pass tests
│   ├── test_preprocessing.py  # Face cropping, lip cropping & tensor prep tests
│   ├── test_inference.py      # Predictor guards & decision interpret tests
│   ├── test_ui.py             # Flask route security & API endpoint tests
│   ├── test_cli.py            # CLI argument parsing & report tests
│   ├── test_reporting.py      # Risk scoring & report generation tests
│   └── test_config.py         # Configuration sanity tests
├── scripts/                   # Command-line tools
│   ├── analyze_video.py       # CLI video inference utility
│   └── summarize_results.py   # Markdown evaluation artifact renderer
├── models/                    # Model weights storage (best_model.pt, audio_model.pt)
└── results/                   # Historical evaluation metrics, ROC curves & confusion matrices
```

## Core Components

### 1. Preprocessing (`src/deepfake_detector/preprocessing.py`)
- Samples 8 equidistant frames across the video duration.
- Identifies the facial bounding box on the median frame using MTCNN.
- Generates 299x299 normalized face crops and 112x112 lip crops.
- Extracts 16 kHz mono audio and computes 64-band log-mel spectrograms.

### 2. Multi-Task Deepfake Detector (`src/deepfake_detector/model.py`)
- **ImageStream**: Xception feature extractor with mean and standard deviation temporal pooling.
- **AudioStream**: Pretrained frozen Wav2Vec 2.0 encoder projected to the shared latent dimension.
- **SyncStream**: Joint 3D-CNN on lip sequences and 2D-CNN on spectrogram representations.
- **Late Fusion Trunk**: Multi-layer perceptron with dropout and batch normalization.
- **Three Dedicated Heads**:
  - `video`: Predicts visual face/lip manipulation.
  - `audio`: Predicts voice cloning or synthetic speech.
  - `any`: Fused decision identifying any manipulation across both modalities.

### 3. Standalone Audio Detector (`src/deepfake_detector/audio_inference.py`)
- Evaluates complete audio recordings of arbitrary duration.
- Deconstructs audio into 3.0-second sliding windows with 1.5-second hop intervals.
- Calculates per-window manipulation likelihood and aggregates recording-level probabilities.
