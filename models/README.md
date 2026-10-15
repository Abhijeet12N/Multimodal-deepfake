# Local model checkpoints

Copy the checkpoints you trained or downloaded into this folder:

```text
models/
├── best_model.pt    # multimodal video + audio checkpoint
└── audio_model.pt   # standalone audio checkpoint
```

The Flask app loads these paths by default. Override them with `MODEL_PATH` and
`AUDIO_MODEL_PATH` if your files are stored elsewhere. Checkpoint files are ignored by Git,
so they remain local and are not uploaded with the source code.

The two checkpoints are separate models and are not interchangeable. The standalone audio
checkpoint must be trained with the current long-recording audio pipeline.
