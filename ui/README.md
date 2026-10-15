# Deepfake Detection Studio - Web User Interface

This directory contains the web dashboard for interacting with the multimodal video and audio deepfake detectors.

## Directory Structure

```text
ui/
├── app.py             # Main Flask server and REST API routes
├── templates/         # HTML templates (Jinja2)
│   └── index.html     # Unified dashboard for video and audio analysis
├── static/            # Static client-side assets
│   ├── app.js         # Interactive analysis controls, chart updates, and JSON exporter
│   └── style.css      # Dark-mode responsive interface styles
├── samples/           # Sample media folder for quick evaluation
│   └── README.md      # Instructions on adding local test samples
├── uploads/           # Ephemeral storage for user-submitted media (git-ignored)
└── README.md          # This documentation file
```

## Running the Web UI

From the project root:

```powershell
# Windows PowerShell
python ui/app.py
```

Or using custom host and port:

```powershell
$env:FLASK_HOST = "0.0.0.0"
$env:FLASK_PORT = "8080"
python ui/app.py
```

On Linux / macOS:

```bash
python ui/app.py
```

Open your browser at `http://127.0.0.1:5000` (or your configured port).

## Model Configuration

By default, the application looks for checkpoints in the `models/` directory:
- `models/best_model.pt` for multimodal video analysis.
- `models/audio_model.pt` for standalone audio analysis.

To point to custom model locations:

```powershell
$env:MODEL_PATH = "C:\custom\path\best_model.pt"
$env:AUDIO_MODEL_PATH = "C:\custom\path\audio_model.pt"
python ui/app.py
```

If one model is missing, the server will start gracefully and display an informative status banner indicating which checkpoint needs to be loaded.

## API Endpoints

- `GET /`: Dashboard interface.
- `GET /api/status`: Check loaded models, device, and service health.
- `POST /api/analyze`: Submit a video (file upload or sample name) for multimodal analysis.
- `POST /api/analyze-audio`: Submit an audio recording (WAV, MP3, M4A, FLAC, OGG, AAC) for sliding-window analysis.
- `GET /samples/<filename>`: Serve sample clips.
- `GET /uploads/<filename>`: Serve uploaded media clips.

