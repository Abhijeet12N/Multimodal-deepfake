# Fine-tuned AASIST installation and evaluation

Inspected: 12 November 2026. Both user-provided ZIPs were extracted under `models/finetuned/` after path, existing-file and ZIP CRC checks. The original ZIPs, both exports and upstream pretrained weights are preserved locally.

## Model selection

Both exports contain full AASIST checkpoints, not AASIST-L. Both identify the same parent checkpoint and training-manifest SHA-256. Validation ROC AUC was recomputed from each export's `val_scores.csv` and agrees with the checkpoint metadata.

| Run | Saved epoch | Validation recordings | Validation ROC AUC | Validation EER | Validation real false positives at raw threshold 0 |
|---|---:|---:|---:|---:|---:|
| **20261108_060713 (selected)** | **4** | **10,511** | **0.999649** | **0.648%** | **28 / 1,211** |
| 20261112_074111 | 2 | 10,511 | 0.999331 | 0.990% | 58 / 1,211 |

The selected run has the higher validation AUC. Selection occurred before inference on the reserved clips. The validation set contains 1,211 bona fide and 9,300 spoof recordings with the same sample IDs and labels across both exports. Validation metrics are model-selection diagnostics, not independent deployment accuracy. EER is a validation curve statistic; its associated threshold was not installed.

Active copy: `models/aasist_finetuned.pth`.

SHA-256: `7c7ef265a2d393634202392e33a868c5a7dd4db46b8e39ef58e134c27dca65ba`.

Original export: `models/finetuned/20261108_060713/aasist/aasist_finetuned.pth`.

The Studio selects **AASIST (fine-tuned)** by default when this installed checkpoint is present. The original Wav2Vec and AASIST-L choices remain available. Checkpoint lookup retains explicit path overrides and falls back to upstream weights when no installed fine-tuned file exists. The local AASIST checkpoint path setting was updated from the upstream baseline to the installed fine-tuned copy.

## Missing calibration and final benchmark outputs

Each ZIP contains only the checkpoint, training history and validation score CSV. Neither includes `dev_scores.csv`, `aasist_platt.json`, `pipeline.json`, `eval_scores.csv` or `evaluation.json`. Consequently, independent calibration and final held-out ASVspoof evaluation cannot be verified from these exports.

The model remains **uncalibrated**, using its existing bona fide logit threshold of **0.0**. No new threshold or calibrator was fitted during installation. The earlier six-clip AASIST calibration artifact is rejected by the Studio provenance checks.

The original training manifest was not supplied. Its hash is preserved in the checkpoint, but the full training split and overlap checks cannot be independently reconstructed from validation scores alone.

## Reserved eight-clip test

Inference used the production Studio loader, CPU with two PyTorch threads, the existing 4.0375-second windows and 2-second hop, and the unchanged raw-score decision policy. All eight predictions used the selected checkpoint hash and completed without inference errors.

| Measurement | Fine-tuned AASIST | Historical pretrained AASIST |
|---|---:|---:|
| Correct decisions | 3 / 8 | 3 / 8 |
| Real clips correctly recognized | 0 / 4 | 0 / 4 |
| Real clips predicted fake | 4 / 4 | 4 / 4 |
| Fake clips detected | 3 / 4 | 3 / 4 |
| Fake clips missed | 1 / 4 | 1 / 4 |

The fine-tuned checkpoint produces the same binary decisions as the earlier pretrained run on this small set, although its raw scores differ. It does not resolve the observed false-positive problem. The local folder labels are provisional and eight clips do not establish general accuracy. The historical baseline is included for context rather than as a repeated runtime benchmark.

Detailed new predictions and timings are in `selected/report.md`, `selected/predictions.csv`, `selected/predictions.jsonl`, and `selected/summary.json`. `selection.json` records checkpoint/archive hashes, validation metrics, missing exports, selection criteria and evaluation results.

## Next step

Resume or complete the Colab scoring stages using this saved checkpoint. Produce independent development recording scores, the matching pipeline signature and Platt artifact, then evaluate the untouched ASVspoof evaluation split. Keep the reserved local clips excluded from fitting. Domain-representative development data is still needed before claiming reliable performance on compressed podcast or web audio.

Reproduce the local selected-checkpoint test:

```powershell
python scripts/evaluate_local_audio.py --models aasist --offline --threads 2 --output reports/finetuned_audio_evaluation/selected
```
