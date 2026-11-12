# Multimodal Deepfake Detection Studio
## Project report and presentation notes

Prepared: 12 November 2026

Scope: This report describes the implementation and saved results available in the local project. Historical benchmark measurements, recent audio tests, and planned upgrades are presented separately. Colab training outcomes are not treated as completed without exported checkpoints and evaluation reports.

## 1. Abstract

Multimodal Deepfake Detection Studio is a research application for identifying manipulated video and synthetic or altered speech. Its multimodal model combines facial representations from Xception, speech representations from Wav2Vec 2.0, and audiovisual compatibility features from a joint 3D/2D convolutional network. A shared late-fusion network produces separate predictions for visual manipulation, audio manipulation, and overall manipulation.

The application also provides an independent audio scanner with selectable Studio Wav2Vec, Phase 1 Wav2Vec, AASIST, and AASIST-L detectors. It analyzes recordings through overlapping windows, aggregates scores, and displays timestamped diagnostics in a Flask web interface. Recent development has focused on consistent preprocessing, correct score interpretation, transparent presentation, reproducible evaluation, and a Colab workflow for further training and recording-level calibration.

Saved multimodal benchmark results are promising. However, the recent eight-clip audio evaluation shows substantial false positives and missed detections. The current system is a research prototype whose predictions require validation on the intended deployment data.

## 2. Problem statement and objectives

Manipulated media may contain a fake face, cloned speech, altered lip movements, or combinations of these changes. Inspecting only one modality can miss evidence available in another. A recording may also contain a short suspicious segment within otherwise genuine content.

The project objectives are:

- Analyze visual, audio, and audiovisual evidence within one application.
- Distinguish visual manipulation, audio manipulation, and overall manipulation.
- Support standalone audio analysis without requiring a video.
- Compare audio detectors through a common interface and UI selector.
- Provide window timestamps, model scores, thresholds, and playback for investigation.
- Evaluate models reproducibly while keeping training, model selection, calibration, and testing separate.

## 3. Scope and technology stack

| Component | Implementation and purpose |
|---|---|
| Backend | Python and Flask; uploads, model loading, inference APIs, and result delivery |
| Deep learning | PyTorch; timm Xception; Transformers Wav2Vec 2.0; upstream AASIST models |
| Video processing | OpenCV and MTCNN face detection |
| Audio processing | SoundFile, SOXR resampling, librosa, and FFmpeg support |
| Frontend | HTML, CSS, JavaScript, Canvas timeline, and native media playback |
| Evaluation | NumPy, scikit-learn, CSV/JSON/JSONL reports, and automated checks |
| Training workflow | Existing multimodal/audio training scripts and a Colab AASIST upgrade notebook |

The UI offers a multimodal video studio, an audio forensic scanner, and a benchmark/architecture view. Supported inputs include common video formats and WAV, MP3, M4A, FLAC, OGG, and AAC audio. Local CPU inference is supported; the prepared fine-tuning workflow targets a GPU runtime.

## 4. Multimodal architecture

### 4.1 Visual stream

The current preprocessor samples eight frames across a video. MTCNN detects a face on a middle sampled frame, and the selected bounding box is reused to obtain face and lip crops. Face images are resized to 299 x 299 pixels and normalized. Xception extracts a representation for each face image. Temporal mean and standard deviation pooling summarize the sampled frames, followed by a projection to a 512-dimensional embedding.

These are learned appearance features that can respond to manipulation-related texture, boundary, or temporal irregularities. The implementation does not explicitly calculate a separate eye-blink, skin-color, or facial-boundary test, and it does not prove which artifact caused a particular prediction.

### 4.2 Speech stream

The multimodal path loads up to the first three seconds of audio at 16 kHz mono and pads short input. A frozen Wav2Vec 2.0 encoder generates contextual speech representations. Temporal mean pooling and trainable projection layers produce a 512-dimensional audio embedding.

The network learns statistical acoustic patterns from its training data. It does not establish authenticity by checking whether the spoken content is factually correct or by recognizing the speaker's identity.

### 4.3 Audiovisual compatibility stream

The preprocessor creates 112 x 112 lip-region crops and a normalized 64-band log-mel spectrogram. A 3D CNN processes the lip-image sequence; a 2D CNN processes the spectrogram. Their features are concatenated and projected to a 512-dimensional embedding.

This stream is intended to help model audiovisual relationships. It is a learned compatibility representation, rather than an explicit measurement of lip-sync offset in milliseconds. The present implementation samples frames across the video but uses only the initial audio segment, which limits strict temporal alignment for longer videos.

### 4.4 Late fusion and output heads

The three 512-dimensional embeddings are concatenated into a 1,536-dimensional feature vector. A shared multilayer perceptron reduces it through 512 and 256 units. Three binary heads produce logits for visual, audio, and overall manipulation. Sigmoid converts these logits into model output scores used by the inference code.

All three heads operate on fused features. The video and audio outputs are task-specific predictions, rather than fully independent decisions made only from their named stream. Their numerical outputs should not automatically be interpreted as calibrated real-world confidence.

Architecture flow:

Video -> sampled face crops -> Xception -> visual embedding

Audio -> 16 kHz waveform -> Wav2Vec 2.0 -> speech embedding

Lip crops + log-mel spectrogram -> 3D/2D CNN -> compatibility embedding

Three embeddings -> concatenation -> shared MLP -> video / audio / overall predictions

## 5. Existing and newly integrated audio detectors

| Detector | Main features and classifier | Default window / hop | Current local status |
|---|---|---|---|
| Studio Wav2Vec + MLP | Wav2Vec contextual embeddings and the existing standalone classifier | 3 s / 1.5 s | Integrated; evaluated on reserved clips |
| Phase 1 Wav2Vec + MLP | Normalized waveform, frozen Wav2Vec CNN features, 512-dimensional mean pooling, two-class MLP | 2 s / 1 s | Integrated; training-preprocessing parity corrected |
| AASIST | Raw-waveform sinc filterbank and spectro-temporal graph attention | 4.0375 s / 2 s | Pretrained upstream checkpoint integrated and evaluated |
| AASIST-L | Compact AASIST configuration | 4.0375 s / 2 s | Pretrained upstream checkpoint integrated and evaluated |
| SSL-AASIST | SSL frontend with AASIST-style detection backend | Model dependent | Adapter/workflow exists; runtime/training validation remains incomplete and work is deferred |
| XLSR-MamBo | Optional adapter in the registry | Model dependent | No completed local evaluation is established in this report |

AASIST and AASIST-L were integrated using upstream implementations and checkpoints. They should be described as integrated pretrained models, rather than newly invented architectures or successfully fine-tuned project models. The local evidence does not establish that they are more accurate than Wav2Vec on arbitrary uploads.

The Phase 1 Wav2Vec detector assigns class 0 to bona fide and class 1 to spoof; its higher score means more spoof-like. AASIST uses upstream class 0 for spoof and class 1 for bona fide; its raw bona fide logit increases toward genuine speech. Preserving these different score directions is essential.

## 6. Audio flow from UI to prediction

1. The user selects a detector in the audio-scanner dropdown and chooses a recording.
2. JavaScript submits the file and detector ID to POST /api/analyze-audio.
3. Flask validates the request, stores the upload, and loads or reuses the selected detector.
4. The shared service decodes the recording, converts it to mono, and resamples it to the detector's required rate.
5. Overlapping windows are generated with the detector's required length, hop, and padding. The shared service can exclude low-energy windows using an RMS gate.
6. The detector runs batches of windows and returns scores with explicit class mapping and score orientation.
7. For the current UI service configuration, the recording decision uses the mean of the active windows' relevant scores. A matching recording-level calibrator is applied only when its provenance and pipeline signature are accepted.
8. The backend returns the recording verdict, raw recording score, thresholds, window flags, timestamps, calibration status, warnings, and timing information.
9. JavaScript displays the prediction, score cards, diagnostic timeline, ranked windows, and linked audio playback.

The older Studio standalone predictor has its own sliding-window inference path. It can return an inconclusive result when local spikes conflict with the recording-level mean. Its behavior is not identical to the shared adapter service.

The RMS gate is a simple energy heuristic, not a trained speech recognizer. Quiet speech can be excluded, and energetic background noise can pass it.

### Recording verdict versus window flags

For the Phase 1 detector, an uncalibrated recording is classified as spoof when its mean spoof score reaches the checkpoint threshold. Windows are flagged separately using their own scores. The count of flagged windows is not a majority vote and is not the percentage of audio that is fake.

Example from the saved Elon Musk podcast test:

| Measurement | Saved value |
|---|---:|
| Mean recording score | 0.911344 |
| Decision threshold | 0.984982 |
| Flagged windows | 45 of 63 |
| Recording prediction | Bona fide |

The result follows the implemented mean-score rule even though most windows are flagged. It does not independently prove that the clip is genuine. Current UI changes display the mean and threshold, identify the verdict as a model prediction, distinguish calibration status, and initialize the playback timer before playback begins.

## 7. Datasets and evaluation design

### Multimodal benchmark

The project contains saved FakeAVCeleb evaluation artifacts covering four combinations: real video/real audio, fake video/real audio, real video/fake audio, and fake video/fake audio. The saved test set contains 3,276 examples: 75, 1,485, 75, and 1,641 examples respectively. Training documentation describes subject-disjoint splitting. This report reads the stored results; it does not independently rerun training or re-audit all original data provenance.

The strong category imbalance should be disclosed. Per-category performance and balanced accuracy are useful alongside overall accuracy.

### Audio benchmark and local evaluation

ASVspoof 2019 LA manifests reference 25,380 training, 24,844 development, and 71,237 evaluation examples. Existing local benchmark work includes cached Wav2Vec feature arrays; this is different from evaluating full raw recordings through the Studio pipeline.

The recent Studio evaluation reserves eight local recordings: four provisionally labeled real and four fake. These labels come from their folders and need independent source verification. All eight clips are excluded from training and threshold tuning. Their file hashes support repeatable comparisons.

The prepared Colab workflow downloads the official raw LA archive directly to Colab storage. It separates weight training, best-epoch selection, calibration, and held-out evaluation. Exported fine-tuned model results have not been verified locally at the time of this report.

## 8. Saved results

### 8.1 Historical multimodal results

Source: results/test_results.json. These are saved benchmark results, not newly measured deployment performance.

| Prediction task | Accuracy | F1 score | ROC AUC |
|---|---:|---:|---:|
| Visual manipulation | 99.88% | 0.9994 | 0.9991 |
| Audio manipulation | 98.69% | 0.9875 | 0.9982 |
| Overall manipulation | 99.69% | 0.9984 | 0.9994 |

The overall balanced accuracy is approximately 99.19%. Overall recall on the real-video/fake-audio category is 93.33%, demonstrating that the aggregate result can hide weaker performance on a smaller category.

The saved ablation file reports lower overall AUC when the image or synchronization inputs are removed. It also reports slightly higher accuracy without the audio input than with the full model. Therefore, the presentation should not claim that adding every modality improves every metric. These experiments support examining each stream's contribution under the documented evaluation setup.

### 8.2 Recent eight-clip Studio audio evaluation

Source: reports/audio_pipeline_evaluation/report.md. These results are separate from the multimodal benchmark.

| Detector | Correct / 8 | Inconclusive | Real clips predicted fake / 4 | Fake clips detected / 4 |
|---|---:|---:|---:|---:|
| AASIST | 3 | 0 | 4 | 3 |
| AASIST-L | 4 | 0 | 4 | 4 |
| Phase 1 Wav2Vec + MLP | 3 | 0 | 3 | 2 |
| Studio Wav2Vec + MLP | 2 | 4 | 0 | 2 |

Inconclusive predictions count as unresolved rather than correct. The Studio Wav2Vec detector's four real clips were all inconclusive, so zero real clips predicted fake does not mean it correctly recognized all genuine clips.

AASIST-L detected all four provisionally fake clips but also classified all four real clips as fake. It cannot be described as the best-performing general detector based on those results. Eight examples are too few to establish general accuracy or choose a reliable deployment threshold.

### 8.3 Feature-based audio benchmark

A saved Phase 1 Wav2Vec report on 71,237 ASVspoof evaluation feature examples records ROC AUC 0.9815 and EER approximately 7.13%. This is a native feature benchmark, not evidence of the same performance on uploaded MP3 recordings. Timing from cached-feature classification is not raw-audio end-to-end inference latency.

## 9. Important development work completed

- Introduced a common AudioDetector interface, model registry, and interchangeable adapters.
- Added a UI detector selector with separate checkpoint paths.
- Corrected Phase 1 waveform normalization to match training-time preprocessing.
- Unified decoding and direct SOXR resampling to avoid an earlier first-use resampling stall.
- Added per-file results, model/preprocessing metadata, file hashes, and timing breakdowns.
- Preserved upstream class mapping and score direction for AASIST.
- Separated recording verdicts from window diagnostics and removed misleading confidence interpretations.
- Added recording score/threshold cards, prediction qualification, and playback-timer initialization.
- Restricted calibration use to matching checkpoints, preprocessing, windowing, and aggregation.
- Prepared a resumable dataset downloader and Colab fine-tuning, calibration, evaluation, and export workflow.

The most recent presentation changes passed 17 JavaScript regression tests covering verdicts, score presentation, window flags, and playback timer behavior. This is a software check, not a measurement of detection accuracy.

## 10. Limitations and remaining work

The main unresolved issue is generalization to audio outside the training distribution. Compressed web clips, background music, recording devices, speaking styles, languages, and unfamiliar generation methods can change model behavior. The current small local evaluation cannot identify which factor caused a particular failure.

Other limitations include sparse video-frame sampling, a reused face bounding box, limited audio coverage in the multimodal path, imperfect alignment between sampled video and the initial audio segment, and the absence of a universally validated calibration artifact for arbitrary uploaded recordings. Window flags are model diagnostics and have not been validated as exact manipulation boundaries.

The next development order is to execute and inspect the prepared Colab workflow, evaluate fine-tuned AASIST/AASIST-L with separate development data, calibrate the complete recording pipeline, and compare frozen candidates on the reserved clips and a larger independently labeled test set. Any new deployment decision should consider genuine-audio false positives, fake-audio recall, uncertainty rate, and runtime together.

Further improvements include temporally aligned audiovisual sampling, better face tracking, more representative audio sources and codecs, multilingual testing, stronger speech-region detection, and repeated latency measurements on the intended hardware. SSL-AASIST remains deferred until its dependencies and training can be validated in isolation.

## 11. Main implementation files

| File | Main responsibility |
|---|---|
| ui/templates/index.html | Video/audio upload forms, detector dropdown, result sections |
| ui/static/app.js | API submission, result rendering, Canvas timeline, playback timer |
| ui/static/audio-presentation.js | Safe score interpretation and audio display preparation |
| ui/app.py | Flask routes, detector selection/loading, upload handling |
| src/deepfake_detector/model.py | ImageStream, AudioStream, SyncStream, MultiTaskDetector, AudioClassifier |
| src/deepfake_detector/preprocessing.py | VideoPreprocessor.process, face/lip crops, mel features, audio windows |
| src/deepfake_detector/inference.py | DeepfakePredictor.predict and multimodal decision interpretation |
| src/deepfake_detector/audio_inference.py | Existing Studio standalone audio predictor |
| src/deepfake_detector/audio_detection/service.py | Shared audio pipeline and recording verdict |
| src/deepfake_detector/audio_detection/detectors/legacy_wav2vec.py | Phase 1 Wav2Vec adapter and normalization |
| src/deepfake_detector/audio_detection/detectors/aasist.py | AASIST/AASIST-L upstream loading and inference |
| src/deepfake_detector/audio_detection/aggregation.py | Recording aggregation and merging flagged intervals |
| src/deepfake_detector/audio_detection/calibration.py | Score calibration |
| scripts/evaluate_local_audio.py | Repeatable multi-detector Studio evaluation |
| scripts/download_asvspoof2019.py | Dataset download, checksum, extraction, validation |
| training/finetune_aasist.py | Prepared training, calibration, evaluation, export pipeline |
| notebooks/AASIST_Colab_Upgrade.ipynb | Colab execution workflow |

## 12. Suggested presentation: 15 slides

### Slide 1 - Project title

- Multimodal Deepfake Detection Studio.
- Detection of visual manipulation and synthetic speech using deep learning.
- Add team members, guide, department, and institution.

### Slide 2 - Problem statement

- Deepfakes can manipulate faces, voices, or both.
- Single-modality analysis can miss complementary evidence.
- Short suspicious segments require timestamped inspection.

### Slide 3 - Project objectives

- Combine facial, speech, and audiovisual features.
- Provide separate video, audio, and overall predictions.
- Support selectable audio detectors and diagnostic playback.

### Slide 4 - System architecture

- Three streams: Xception, Wav2Vec 2.0, and joint 3D/2D CNN.
- Three 512-dimensional embeddings feed a shared fusion network.
- Three binary output heads support manipulation analysis.

### Slide 5 - Video preprocessing and visual stream

- Sample eight frames; obtain face and lip crops using MTCNN.
- Resize faces to 299 x 299 and extract Xception features.
- Pool temporal mean and standard deviation.

### Slide 6 - Audio and audiovisual streams

- Normalize input format to 16 kHz mono.
- Use Wav2Vec speech representations and log-mel features.
- Combine lip-image and spectrogram features for audiovisual compatibility.

### Slide 7 - Standalone audio model comparison

- Existing Studio and Phase 1 Wav2Vec + MLP baselines.
- Newly integrated pretrained AASIST and AASIST-L.
- SSL-AASIST deferred; no completed training result claimed.

### Slide 8 - UI-to-backend workflow

- Select detector and upload media.
- Flask validates input and invokes inference.
- Return recording scores, thresholds, windows, and timings.

### Slide 9 - Sliding-window analysis

- Use detector-specific overlapping windows.
- Aggregate active-window scores for the recording decision.
- Window flags support review; their count is not a vote or a fake-audio percentage.

### Slide 10 - Datasets and evaluation

- FakeAVCeleb for the saved multimodal benchmark.
- ASVspoof 2019 LA for audio development and benchmark work.
- Eight reserved local clips for a separate exploratory comparison.

### Slide 11 - Historical multimodal results

- Overall accuracy: 99.69%; overall ROC AUC: 0.9994.
- Overall F1: 0.9984; balanced accuracy: approximately 99.19%.
- Identify these as saved benchmark results and disclose category imbalance.

### Slide 12 - Current audio findings

- AASIST and Wav2Vec still show false positives and missed detections.
- AASIST-L flags all four real clips as fake in the small local test.
- Local exploratory results do not establish general accuracy.

### Slide 13 - Implementation improvements

- Detector selection moved to the UI.
- Corrected preprocessing and score orientation; improved decoding.
- Added transparent score/threshold presentation and playback diagnostics.

### Slide 14 - Limitations and future work

- Validate generalization on representative, independently labeled recordings.
- Execute Colab fine-tuning and recording-level calibration.
- Improve audiovisual alignment, tracking, robustness, and runtime evaluation.

### Slide 15 - Project takeaway

- Implemented an integrated multimodal research studio with modular audio comparison.
- Demonstrated strong saved benchmark results and identified real-world audio limitations.
- The next milestone is independently validated audio improvement.

## 13. Suggested demonstration and viva points

Demonstration: show the UI detector selector, upload a known-source clip, explain the recording score and threshold, inspect a flagged window, and play the corresponding segment. Show one difficult or inconclusive example so that the audience sees the system's limitations.

Q: Why multimodal detection? A: Manipulation can affect different streams, and their joint features provide complementary evidence.

Q: Why use Wav2Vec and AASIST? A: Wav2Vec supplies the existing learned-speech baseline; AASIST supplies a different raw-waveform graph-attention approach for comparison. Accuracy improvement must be measured.

Q: Is a model score of 0.99 equivalent to 99% certainty? A: No. Calibration and validation on representative independent data are required before using that interpretation.

Q: Why can many windows be flagged while the recording is predicted bona fide? A: The shared pipeline uses the mean score and its decision threshold; the window count is a separate diagnostic.

Q: Did this project invent AASIST? A: No. The project integrates upstream pretrained models, implements adapters and the application pipeline, and prepares a separate fine-tuning workflow.

Q: Is the system production ready? A: It is a research prototype. Current local audio errors and incomplete validation prevent a claim of reliable unrestricted verification.

## 14. Evidence and presentation assets

Use results/test_results.json and results/per_category.csv for saved multimodal metrics; results/ablation.csv for ablations; reports/audio_pipeline_evaluation/report.md and predictions.csv for the recent eight-clip comparison; docs/audio_upgrade_execution.md for upgrade status; and the model, preprocessing, service, and UI files above for implementation details.

Existing figures available for slides include results/roc_3tasks.png, results/training_curves.png, results/split_distribution.png, and results/ablation.png. Use the ablation figure with the metric and experimental setup clearly identified. Add a fresh screenshot after restarting the app and loading the updated UI.

For presentation references, the upstream repositories are listed in the project documentation: FakeAVCeleb (DASH-Lab/FakeAVCeleb), AASIST (clovaai/aasist), and Wav2Vec 2.0 (facebookresearch/fairseq). This report does not substitute literature benchmark numbers for measured project results.
