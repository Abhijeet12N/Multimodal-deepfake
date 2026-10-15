# Deepfake Detection Studio · Flask Demo

This folder contains the local web demo for single-video multimodal deepfake analysis.

## Run

```bash
cd demo_site

python app.py
```

By default the app loads `../models/best_model.pt` for video and `../models/audio_model.pt` for
standalone audio. Set `MODEL_PATH` or `AUDIO_MODEL_PATH` to override these paths.

Open `http://127.0.0.1:5000`.

If `MODEL_PATH` is missing or invalid, the UI still opens and shows a model warning. Analysis requests return a clear error until a valid checkpoint is configured.

## Checkpoint files

Copy your files into the project root's `models/` directory:

```text
models/best_model.pt
models/audio_model.pt
```

The multimodal video detector and standalone audio detector use different checkpoints.

## Inputs

Supported upload extensions:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`

Sample videos can be placed under `demo_site/samples/`. Files are ignored by Git to avoid redistributing licensed or private video data.

## Output

For each successful analysis, the demo shows:

- video/audio/any scores,
- a final decision label,
- timing and preprocessing metadata,
- a structured explanation report,
- downloadable JSON report.
