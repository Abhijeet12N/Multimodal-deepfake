# Audio pipeline evaluation

Fresh evaluation of eight reserved clips through the studio loaders. No weights, thresholds or calibration were fitted using these clips.

Folder labels are provisional; this small set does not establish general accuracy. The old six-clip AASIST calibrator and native-input Wav2Vec calibrator were not applied.

| Detector | Correct / 8 | Inconclusive | Real clips flagged / 4 | Fake clips detected / 4 | Analysis seconds |
|---|---:|---:|---:|---:|---:|
| aasist | 3 | 0 | 4 | 3 | 54.14 |

Inconclusive results remain unresolved and count against correct/8. Times exclude model loading and are one CPU run with two PyTorch threads, not a repeated latency benchmark.

## Per-clip decisions

| File | Label | aasist |
|---|---|---|
| Real Podcast Bill Gates.mp3 | real | fake |
| Real Podcast Elon Musk.mp3 | real | fake |
| Such a great quote 🔥 #shorts - Jay Shetty.mp3 | real | fake |
| What is the best advice Alia Bhatt has ever received #jayshetty #podcast #aliabhatt - Jay Shetty Podcast.mp3 | real | fake |
| Deepfake video of Volodymyr Zelensky surrendering surfaces on social media - The Telegraph.mp3 | fake | fake |
| Fake audio.mp3 | fake | fake |
| Fake clip.mp3 | fake | real |
| Fake donald trump.mp3 | fake | fake |

## Timing breakdown

| Detector | Decode seconds | Model seconds |
|---|---:|---:|
| aasist | 0.498 | 53.525 |

Raw window scores, timestamps, thresholds, calibration status and file SHA-256 values are saved in `predictions.jsonl`; flat results are in `predictions.csv`.
