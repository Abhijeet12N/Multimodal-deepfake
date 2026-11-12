# Audio upgrade execution

## Fine-tuned exports installed on 12 November 2026

Two Colab AASIST exports have now been supplied and inspected locally. Run
`20261108_060713`, epoch 4, was selected using its recomputed validation AUC
(0.999649 versus 0.999331 for the other export), before reserved-clip testing.
Its checkpoint is installed at `models/aasist_finetuned.pth` and is the Studio's
default **AASIST (fine-tuned)** option on this machine.

The exports contain checkpoints, training histories and validation scores, but no
separate development calibration or final evaluation outputs. The installed model
therefore remains uncalibrated. It detects 3/4 fake clips but misclassifies 4/4 real
clips in the reserved test. See the [installation and evaluation report](../reports/finetuned_audio_evaluation/report.md)
for hashes, model-selection evidence, per-clip results and limitations. The sections
below describe the original workflow; their earlier local execution limitations
do not imply that the newly supplied training checkpoints are absent.

## Completed locally

- Restored Phase 1 Wav2Vec preprocessing: pad/truncate to 32,000 samples, apply its processor, extract CNN features and mean-pool. Regression checks compare the exact tensors against the training processor output.
- Unified studio audio decoding on soundfile and direct SOXR HQ resampling, with FFmpeg/librosa fallbacks. This avoids the earlier first-use resampling stall and retains the resampling algorithm.
- Added decode/window/model timings and a reproducible four-detector, eight-clip runner. Fresh predictions completed for all 32 model/file combinations. See [evaluation](../reports/audio_pipeline_evaluation/report.md).
- Bound recording calibration to a fingerprint of checkpoint, preprocessing, window sizes, VAD and aggregation. Native-input calibration is not applied to full-recording inference. Recording calibration is applied after raw-score aggregation; timeline flags remain raw window diagnostics.
- Recorded the eight evaluation file hashes in [data readiness](../reports/audio_development_readiness.json). Training and calibration exclude them. The local ASVspoof manifests reference feature caches, not raw audio.

The fixes improve correctness and execution. They do not establish a reliable detector for arbitrary uploaded audio. Current checkpoints still have large false-positive/missed-fake counts on the local clips.

## Continue in Colab

Initialize the source dependency with `git submodule update --init --recursive`,
then run `python scripts/build_audio_colab_bundle.py`. The pinned source, official
starting checkpoints, and upstream license/notice are packaged from
`third_party/aasist` into the same relative path inside the Colab project. No
sibling repository is required. Older bundle layouts remain readable by the loader.

Open [AASIST_Colab_Upgrade.ipynb](../notebooks/AASIST_Colab_Upgrade.ipynb) in a Colab GPU runtime and upload the local `artifacts/audio_colab_bundle.zip` when prompted. Run the cells in order. The notebook downloads the raw dataset directly onto Colab; neither a local dataset nor a dataset on Drive is needed.

The downloader uses the [official ASVspoof 2019 publisher archive](https://zenodo.org/records/6906306): 7,640,952,520 bytes, MD5 `30c98f11d8b2bc21f2c257bfd78bb5c5`. It resumes interrupted downloads, validates size/checksum, checks disk space, extracts safely and verifies every protocol-referenced waveform. It removes the ZIP after successful validation and reuses the extracted data on subsequent runs in that runtime. The resolved `DATA_ROOT` is loaded automatically from `dataset_ready.json`.

The notebook supports the official ASVspoof 2019 LA raw folder layout, or a custom waveform manifest. It splits the ASVspoof development speakers into independent `val` and calibration `dev` groups using a fixed seed. With a custom manifest, provide these four disjoint splits yourself:

| Split | Used for |
|---|---|
| train | Weight updates and gain/noise augmentation |
| val | Choosing the best epoch using recording AUC |
| dev | Fitting recording-level Platt calibration |
| eval | One final held-out check |

Speaker IDs, source IDs and file hashes are checked across splits before training. Each split must have both classes. Calibration requires at least 1,000 independent dev examples. These checks do not infer unknown source provenance or detect every re-encoded derivative.

The runner fine-tunes original AASIST/AASIST-L with their own upstream class mapping (0=spoof, 1=bonafide). It saves weights/reports under `/content/audio_upgrade_results/` by default, then exports `trained_audio_models.zip`. Set `SAVE_OUTPUTS_TO_DRIVE=True` for optional output backups; the dataset still stays on Colab local disk. Download local outputs before deleting the runtime. No Fairseq or SSL-AASIST training is involved. Include representative compression and diverse synthesis methods in the training data; gain/noise augmentation alone is insufficient evidence of web-audio robustness.

**The full dataset download and GPU training have not been executed from this local session.** The official download endpoint was checked using a 32-byte range request, which returned a ZIP signature and the expected full archive size. Resumption, extraction and protocol validation have small-file regression tests. Actual download, training and calibration run when the notebook is executed in Colab.

The Colab setup installs audio/scoring dependencies only and preserves its installed Transformers/Hub packages. AASIST imports and inference are checked with Hugging Face imports unavailable. If you used the older notebook cell that installed Transformers 4, use the updated notebook and source ZIP together; restarting the session clears previously imported modules without deleting your dataset files. This AASIST workflow does not depend on the Hub versions used by Gradio/Diffusers.

## Install Colab outputs

Preserve the original checkpoints and put fine-tuned files under a new local path. Set `AASIST_CHECKPOINT_PATH` or `AASIST_L_CHECKPOINT_PATH` to that file. These settings identify files; detector selection stays in the UI dropdown.

Copy the matching `aasist_platt.json` or `aasist_l_platt.json` into `configs/calibration/`. Restart Flask, choose the corresponding detector and re-evaluate the reserved eight clips. A calibration from another checkpoint or a changed pipeline will be rejected.

## Reproduce local evaluation

```powershell
python scripts/evaluate_local_audio.py --offline --threads 2
python scripts/prepare_audio_development.py
python scripts/build_audio_colab_bundle.py
```

The evaluation runner writes each prediction immediately to JSONL and emits a CSV, summary JSON and Markdown report after completion. Clip labels come from the Real/Fake folders and require independent source verification before making broader accuracy claims.
