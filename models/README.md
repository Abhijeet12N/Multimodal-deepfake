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

## Installed fine-tuned audio model

`aasist_finetuned.pth` is the selected AASIST export from run `20261108_060713`,
epoch 4. The original exports from both supplied ZIPs are preserved under
`finetuned/<run_id>/aasist/`. See
[selection and evaluation](../reports/finetuned_audio_evaluation/report.md).

With this installed file, the Studio defaults to **AASIST (fine-tuned)**. An explicit
`AASIST_CHECKPOINT_PATH` takes precedence; otherwise lookup prefers
`models/aasist_finetuned.pth` and falls back to the upstream pretrained checkpoint.
Other detectors can still be selected from the UI dropdown.

The supplied exports do not include independent development calibration or final
benchmark outputs. This model is uncalibrated and still flags all four real clips
in the reserved local test. Fine-tuning is verified; reliable web-audio accuracy is
not established. Checkpoints, source ZIPs and extracted export folders stay local.

## AASIST source dependency

Initialize `third_party/aasist` with `git submodule update --init --recursive`.
The parent repository pins the tested upstream source and preserves its MIT license
and notice. That upstream revision includes the official baseline checkpoints;
your fine-tuned checkpoints remain separate, ignored files under `models/`.
The Studio prefers the installed fine-tuned file, then the submodule baseline,
and finally older workspace paths. See [dependency details](../third_party/README.md).
