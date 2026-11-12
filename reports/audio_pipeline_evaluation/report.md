# Audio pipeline evaluation

Fresh evaluation of eight reserved clips through the studio loaders. No weights, thresholds or calibration were fitted using these clips.

Folder labels are provisional; this small set does not establish general accuracy. The old six-clip AASIST calibrator and native-input Wav2Vec calibrator were not applied.

| Detector | Correct / 8 | Inconclusive | Real clips flagged / 4 | Fake clips detected / 4 | Analysis seconds |
|---|---:|---:|---:|---:|---:|
| aasist | 3 | 0 | 4 | 3 | 65.69 |
| aasist_l | 4 | 0 | 4 | 4 | 54.73 |
| legacy_wav2vec_mlp | 3 | 0 | 3 | 2 | 28.97 |
| legacy_standalone | 2 | 4 | 0 | 2 | 77.08 |

Inconclusive results remain unresolved and count against correct/8. Times exclude model loading and are one CPU run with two PyTorch threads, not a repeated latency benchmark.

## Per-clip decisions

| File | Label | aasist | aasist_l | legacy_wav2vec_mlp | legacy_standalone |
|---|---|---|---|---|---|
| Real Podcast Bill Gates.mp3 | real | fake | fake | fake | uncertain |
| Real Podcast Elon Musk.mp3 | real | fake | fake | real | uncertain |
| Such a great quote 🔥 #shorts - Jay Shetty.mp3 | real | fake | fake | fake | uncertain |
| What is the best advice Alia Bhatt has ever received #jayshetty #podcast #aliabhatt - Jay Shetty Podcast.mp3 | real | fake | fake | fake | uncertain |
| Deepfake video of Volodymyr Zelensky surrendering surfaces on social media - The Telegraph.mp3 | fake | fake | fake | fake | fake |
| Fake audio.mp3 | fake | fake | fake | real | fake |
| Fake clip.mp3 | fake | real | fake | fake | real |
| Fake donald trump.mp3 | fake | fake | fake | real | real |

## Timing breakdown

| Detector | Decode seconds | Model seconds |
|---|---:|---:|
| aasist | 0.614 | 64.989 |
| aasist_l | 0.645 | 54.059 |
| legacy_wav2vec_mlp | 0.654 | 28.293 |
| legacy_standalone | 0.603 | 76.463 |

Raw window scores, timestamps, thresholds, calibration status and file SHA-256 values are saved in `predictions.jsonl`; flat results are in `predictions.csv`.
