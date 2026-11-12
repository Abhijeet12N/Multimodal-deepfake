# Project reports

Start with [PPT_Generation_Report.pdf](PPT_Generation_Report.pdf) for presentation preparation, or copy content from the [editable Markdown report](PPT_Generation_Report.md). It is the current report as of 12 November 2026 and contains the architecture, model inventory, measured results, fine-tuned AASIST status, a 16-slide outline, speaker notes, suggested visuals, and viva questions.

## Supporting evidence

| Location | Contents and intended use |
|---|---|
| [finetuned_audio_evaluation/report.md](finetuned_audio_evaluation/report.md) | Current AASIST export selection, checkpoint provenance, missing calibration outputs, and the latest eight-clip results |
| [finetuned_audio_evaluation/selected/](finetuned_audio_evaluation/selected/) | Raw predictions, timing, summary, and readable output for the installed fine-tuned model |
| [audio_pipeline_evaluation/report.md](audio_pipeline_evaluation/report.md) | Earlier eight-clip comparison across the four original Studio detector choices |
| [baseline_audio_benchmark/](baseline_audio_benchmark/) | Historical Phase 1 Wav2Vec native-feature benchmark on 71,237 ASVspoof evaluation examples; not a raw-upload latency benchmark |
| [calibration_fit_wav2vec_dev/](calibration_fit_wav2vec_dev/) | Canonical native-feature development scores and metrics retained as provenance for the historical Wav2Vec calibrator |
| [audio_development_readiness.json](audio_development_readiness.json) | Reserved eight-clip hashes and the original local raw-data readiness audit; used by Colab packaging |
| [dataset_summary.json](dataset_summary.json), [leakage_audit.json](leakage_audit.json) | Historical manifest/split audits; the older six-clip manifest in these audits is separate from the later eight-clip Studio evaluation |
| [../results/](../results/) | Saved multimodal metrics, category results, training curves, ROC curves, and ablation figures |

The current report distinguishes saved multimodal benchmark accuracy, fine-tuned validation AUC, and exploratory local audio outcomes. The selected fine-tuned AASIST remains uncalibrated and incorrectly flags all four provisionally real clips in the local test.

## Cleanup and regeneration

The October report and previous presentation report were replaced by the current PPT report. Superseded six-clip audio summaries, mixed literature/local comparison documents, duplicate output aliases, and a redundant development benchmark summary were removed. Unique current results and calibration provenance are retained in the folders above.

Regenerate the presentation PDF from its Markdown source:

```powershell
python scripts/build_presentation_report.py
```

The older `python scripts/generate_project_report.py` command now calls the same generator and produces the same current PDF. Edit `PPT_Generation_Report.md` before regeneration; the generator does not refresh model evaluations.
