# Audio model comparison and calibration

## Local clips reserved for evaluation

The eight files in `initial setup/deepfake_phase1/Audios/{Real,Fake}` were treated as evaluation data. None of these files was used to fit the calibrator or select its threshold.

| Clip | Label | Studio AASIST | Previous UI Wav2Vec | Phase 1 Wav2Vec |
|---|---|---|---|---|
| Real Podcast Alia Bhatt | Real | Real | Fake | Fake |
| Real Podcast Jay Shetty quote | Real | Real | Fake | Fake |
| Real Podcast Bill Gates | Real | Fake | Fake | Fake |
| Fake clip | Fake | Real | Real | Fake |
| Fake Donald Trump | Fake | Real | Real | Real |
| Fake audio | Fake | Real | Fake | Fake |
| Real Podcast Elon Musk | Real | Real | Fake | Real |
| Deepfake video of Volodymyr Zelensky | Fake | Fake | Fake | Fake |
| **Accuracy** |  | **4/8** | **2/8** | **4/8** |

The models disagree substantially on these compressed, out-of-domain clips. Results do not establish which model is reliable for arbitrary web audio.

## Phase 1 Wav2Vec calibration

Platt scaling was fit using the Phase 1 Wav2Vec predictions on 24,844 ASVspoof 2019 LA development recordings (2,548 bona fide and 22,296 spoof). It was then evaluated without refitting on the separate 71,237-recording eval split.

| Split | Count | ECE | Brier score | Log loss | Accuracy at calibrated 0.5 |
|---|---:|---:|---:|---:|---:|
| Dev, fit set | 24,844 | 0.00488 | 0.01856 | 0.07988 | 97.87% |
| Eval, held out | 71,237 | 0.05868 | 0.05713 | 0.20185 | 93.04% |

The fit artifact is `configs/calibration/legacy_wav2vec_mlp_platt.json`. Dev fit metrics are in-sample; held-out eval metrics are the better estimate of calibration generalization. The raw classifier's historical eval report records EER 7.13% and accuracy 84.90% at its checkpoint threshold; calibration changes score mapping and the operating point, not the underlying ranking ability.

## Current UI behavior and limitations

- The UI only loads calibration when the artifact identifies the matching model and has dev/val provenance with at least 1,000 fit examples. The earlier AASIST artifact has only six eval examples and is ignored.
- To select this calibrated model, set `AUDIO_DETECTOR=legacy_wav2vec_mlp`. The Phase 1 checkpoint is resolved independently of the older generic `AUDIO_MODEL_PATH`; `LEGACY_AUDIO_CHECKPOINT_PATH` can override it.
- AASIST remains uncalibrated. The available dev cache contains Wav2Vec features, not raw waveforms, so it cannot be used to calibrate AASIST.
- The eight local clips have not been re-scored with the new calibrator: the Wav2Vec frontend is not cached in the current runtime, and network access to download it is blocked. Their comparison above is the prior, uncalibrated evaluation. Re-scoring them with the fitted artifact remains a follow-up evaluation step, with no refitting or threshold changes based on those results.
