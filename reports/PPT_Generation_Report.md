# Multimodal Deepfake Detection Studio
## Current project report for PPT generation

Prepared: 12 November 2026

Scope: This is the current presentation source as of 12 November 2026. It includes the installed fine-tuned AASIST checkpoint, verified validation exports, the latest reserved-clip tests, and remaining calibration work. Historical multimodal benchmarks, validation results, and exploratory audio tests are separate forms of evidence.

## Instructions for generating the PPT

Create a 16-slide academic project presentation from this report, following Section 12. Use three to five short bullets per slide and put the explanation in speaker notes. Use a clean technical theme, readable tables, and the existing project figures identified in Section 14. Keep team members, guide, college, and department as editable placeholders because those details were not provided.

Use the implemented architecture and measured values in this report. Identify the 99.69% overall accuracy as a saved multimodal benchmark result, the 0.999649 AASIST AUC as validation performance, and the 3/8 result as an exploratory local audio test. Explain that the fine-tuned model still misclassifies all four provisionally real clips. Present SSL-AASIST as deferred and calibration/final ASVspoof evaluation as incomplete. Do not invent performance gains, successful deployment claims, references, names, or completed training stages.

The slide outline supplies content, speaker notes, and suggested visuals. The technical sections provide the detail needed to answer presentation questions.

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

AASIST source is supplied through the pinned `third_party/aasist` Git submodule at
commit `a04c9863f63d44471dde8a6abcb3b082b07cd1d1`. New clones use
`git clone --recurse-submodules`; existing clones use `git submodule update --init --recursive`.
The upstream license and notice remain with its source. Project fine-tuned checkpoints
and datasets are distributed separately and remain excluded from the main Git repository.

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
| AASIST | Raw-waveform sinc filterbank and spectro-temporal graph attention | 4.0375 s / 2 s | Pretrained and supplied fine-tuned checkpoints integrated and evaluated locally |
| AASIST-L | Compact AASIST configuration | 4.0375 s / 2 s | Pretrained upstream checkpoint integrated and evaluated |
| SSL-AASIST | SSL frontend with AASIST-style detection backend | Model dependent | Adapter/workflow exists; runtime/training validation remains incomplete and work is deferred |
| XLSR-MamBo | Optional adapter in the registry | Model dependent | No completed local evaluation is established in this report |

AASIST and AASIST-L use upstream architectures and pretrained checkpoints. Two user-supplied fine-tuned AASIST exports have now been inspected; the selected epoch-4 checkpoint is integrated locally. AASIST-L fine-tuning has not been established from the supplied files. These are integrated and fine-tuned upstream architectures, rather than newly invented model architectures. The local evidence does not establish superior accuracy on arbitrary uploads.

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

The prepared Colab workflow downloads the official raw LA archive directly to Colab storage. It separates weight training, best-epoch selection, calibration, and held-out evaluation. Two supplied fine-tuned AASIST checkpoints and their validation scores have now been verified locally. Their exports do not contain independent development calibration or final held-out ASVspoof evaluation results.

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

Sources: reports/audio_pipeline_evaluation/report.md for the earlier four-model run and reports/finetuned_audio_evaluation/selected/report.md for the selected fine-tuned model. These results are separate from the multimodal benchmark.

| Detector | Correct / 8 | Inconclusive | Real clips predicted fake / 4 | Fake clips detected / 4 |
|---|---:|---:|---:|---:|
| Pretrained AASIST | 3 | 0 | 4 | 3 |
| Fine-tuned AASIST (selected) | 3 | 0 | 4 | 3 |
| AASIST-L | 4 | 0 | 4 | 4 |
| Phase 1 Wav2Vec + MLP | 3 | 0 | 3 | 2 |
| Studio Wav2Vec + MLP | 2 | 4 | 0 | 2 |

Inconclusive predictions count as unresolved rather than correct. The Studio Wav2Vec detector's four real clips were all inconclusive, so zero real clips predicted fake does not mean it correctly recognized all genuine clips.

AASIST-L detected all four provisionally fake clips but also classified all four real clips as fake. It cannot be described as the best-performing general detector based on those results. Eight examples are too few to establish general accuracy or choose a reliable deployment threshold.

### 8.3 Feature-based audio benchmark

A saved Phase 1 Wav2Vec report on 71,237 ASVspoof evaluation feature examples records ROC AUC 0.9815 and EER approximately 7.13%. This is a native feature benchmark, not evidence of the same performance on uploaded MP3 recordings. Timing from cached-feature classification is not raw-audio end-to-end inference latency.

### 8.4 Supplied fine-tuned AASIST checkpoints

Both supplied exports contain validation predictions for the same 10,511 recordings. Run 20261108_060713, epoch 4, has validation ROC AUC 0.999649, compared with 0.999331 for run 20261112_074111, epoch 2. The higher-AUC checkpoint was selected before testing the reserved clips and installed as the default AASIST (fine-tuned) option.

| Export | Selected epoch | Validation AUC | Validation EER | Real false positives at threshold 0 |
|---|---:|---:|---:|---:|
| 8 November run (selected) | 4 | 0.999649 | 0.648% | 28 / 1,211 |
| 12 November run | 2 | 0.999331 | 0.990% | 58 / 1,211 |

The validation set contains 1,211 bona fide and 9,300 spoof examples. The installed copy is models/aasist_finetuned.pth. Its provenance and checksum are recorded in reports/finetuned_audio_evaluation/selection.json. Validation EER is a curve statistic; its associated threshold was not installed. Both ZIPs lack independent development calibration and final ASVspoof evaluation outputs.

Its reserved-clip results remain 3/8 correct: 3/4 fake clips detected and all 4/4 real clips incorrectly flagged. This is the same binary outcome as the historical pretrained AASIST run. Fine-tuning and local integration are established; improved web-audio accuracy is not. No calibration or threshold adjustment was fitted during installation. See reports/finetuned_audio_evaluation/report.md for provenance and limitations.

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
- Inspected two supplied fine-tuned AASIST exports, selected by validation AUC, installed the chosen checkpoint, and evaluated all eight reserved clips.

The fine-tuned deployment changes passed 24 relevant Python regression tests. Earlier presentation changes passed 17 JavaScript regression tests covering verdicts, scores, flags, and playback. Two actual multipart API uploads also matched the selected checkpoint results. Software checks establish execution behavior, not detection accuracy.

## 10. Limitations and remaining work

The main unresolved issue is generalization to audio outside the training distribution. Compressed web clips, background music, recording devices, speaking styles, languages, and unfamiliar generation methods can change model behavior. The current small local evaluation cannot identify which factor caused a particular failure.

Other limitations include sparse video-frame sampling, a reused face bounding box, limited audio coverage in the multimodal path, imperfect alignment between sampled video and the initial audio segment, and the absence of a universally validated calibration artifact for arbitrary uploaded recordings. Window flags are model diagnostics and have not been validated as exact manipulation boundaries.

The next development order is to complete the independent development-calibration and held-out evaluation stages for the supplied AASIST checkpoint, validate the complete recording pipeline on representative audio, and compare frozen candidates on a larger independently labeled test set. No fine-tuned AASIST-L export has been supplied. Any deployment decision should consider genuine-audio false positives, fake-audio recall, uncertainty rate, and runtime together.

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

## 12. Slide-ready presentation: 16 slides

Suggested length: 10-15 minutes. Adjust the delivery time to your institution's requirements.

### Slide 1 - Project title

- Multimodal Deepfake Detection Studio.
- Deep learning for visual manipulation and synthetic-speech analysis.
- Team: [names]; guide: [name]; department and institution: [details].

Speaker notes: Introduce this as an implemented research prototype with a web interface, multimodal analysis, and interchangeable audio models. The latest addition is a supplied fine-tuned AASIST checkpoint.

Suggested visual: A current screenshot of the Studio homepage.

### Slide 2 - Problem statement

- Media manipulation can change the face, voice, or both.
- Single-modality analysis can miss complementary evidence.
- Long recordings need timestamped inspection of suspicious segments.

Speaker notes: Explain why a genuine video can contain fake audio and why genuine speech can accompany a manipulated face. Detection supports review; a model prediction alone does not establish provenance.

Suggested visual: A simple four-case grid: real/real, fake/real, real/fake, fake/fake.

### Slide 3 - Objectives and project scope

- Combine visual, speech, and audiovisual representations.
- Predict video manipulation, audio manipulation, and overall manipulation.
- Support standalone audio, detector selection, and interactive diagnostics.
- Evaluate reproducibly with separate fitting and testing stages.

Speaker notes: The project covers the application pipeline and model integration. It also records errors and uncertain outcomes so that accuracy limitations remain visible.

Suggested visual: A short checklist of the implemented modules.

### Slide 4 - Existing approach and proposed system

- Baseline: Wav2Vec features with MLP audio classifiers.
- Multimodal path: Xception + Wav2Vec + 3D/2D compatibility stream.
- Audio upgrade: common adapters for pretrained and fine-tuned AASIST.
- Flask Studio presents recording scores and window diagnostics.

Speaker notes: The contribution is the integrated workflow, modular detector comparison, preprocessing corrections, and transparent presentation. AASIST and Wav2Vec are upstream architectures rather than inventions of this project.

Suggested visual: A baseline-to-integrated-system comparison table.

### Slide 5 - System architecture

- Visual branch: sampled faces -> Xception -> 512-dimensional embedding.
- Speech branch: waveform -> Wav2Vec -> 512-dimensional embedding.
- Compatibility branch: lip images + log-mel -> 3D/2D CNN -> 512 dimensions.
- Concatenate 1,536 features -> MLP -> three binary heads.

Speaker notes: Late fusion combines the learned embeddings before classification. The output heads are task-specific, but all receive shared fused features; they are not completely independent modality decisions.

Suggested visual: A three-branch architecture diagram with a shared fusion trunk.

### Slide 6 - Visual preprocessing and features

- Sample eight frames across the video.
- MTCNN selects a face; reuse its box for face and lip crops.
- Resize faces to 299 x 299 pixels; run Xception.
- Pool temporal mean and standard deviation.

Speaker notes: The network learns appearance patterns associated with its training labels. It does not implement a separate explicit eye-blink or facial-boundary authenticity test. Sparse sampling and box reuse are current limitations.

Suggested visual: The Studio face/lip crop gallery.

### Slide 7 - Speech and audiovisual features

- Multimodal audio: initial three seconds, 16 kHz mono.
- Frozen Wav2Vec contextual features with trainable projection.
- 112 x 112 lip crops and 64-band log-mel spectrogram.
- Joint 3D/2D CNN learns audiovisual compatibility.

Speaker notes: The compatibility branch is a learned representation, not a measured lip-sync delay. Sampling video over the full clip while using initial audio limits strict alignment in longer clips.

Suggested visual: Lip crops beside a spectrogram, connected to the two CNN branches.

### Slide 8 - Standalone audio model inventory

- Studio Wav2Vec + MLP: 3-second windows, 1.5-second hop.
- Phase 1 Wav2Vec + MLP: 2-second windows, 1-second hop.
- AASIST/AASIST-L: raw waveform, 4.0375-second windows, 2-second hop.
- Selected fine-tuned AASIST is the current local default.

Speaker notes: AASIST learns spectro-temporal relationships through graph attention. AASIST-L is its compact alternative. Only AASIST fine-tuned exports were supplied; SSL-AASIST is deferred and XLSR-MamBo has no completed local evaluation here.

Suggested visual: The detector table from Section 5 and a screenshot of the dropdown.

### Slide 9 - Audio flow: UI to backend

- Select detector and upload a recording.
- POST /api/analyze-audio -> validation and checkpoint loader.
- Decode, resample, generate windows, and run batched inference.
- Aggregate recording score; return verdict, thresholds, windows, and timings.

Speaker notes: Name ui/app.py as the API entry point and audio_detection/service.py as the shared inference service. app.js and audio-presentation.js render the returned result. The older standalone predictor uses its own compatible path.

Suggested visual: A sequence diagram: browser -> Flask -> audio service -> detector -> browser.

### Slide 10 - Scores, window flags, and uncertainty

- Current shared UI service averages relevant active-window scores.
- Phase 1 score rises toward spoof; AASIST bona fide logit rises toward real.
- Window flags support review and are not a majority vote.
- Scores remain uncalibrated unless a matching artifact is accepted.

Speaker notes: Use the saved podcast example: average 0.911344 below threshold 0.984982 despite 45/63 flags. Explain that a flagged-window ratio is not a fake-audio percentage. The updated UI displays the recording score and threshold.

Suggested visual: A result screenshot containing both score cards and the timeline.

### Slide 11 - Datasets and evaluation separation

- FakeAVCeleb: four real/fake audio-video combinations.
- Saved multimodal test artifacts contain 3,276 examples.
- ASVspoof LA: 25,380 train, 24,844 dev, 71,237 eval examples.
- Reserve eight local clips separately; do not fit to them.

Speaker notes: The saved multimodal dataset is imbalanced. Audio feature-cache benchmarks differ from end-to-end uploaded-recording tests. The planned raw-audio workflow separates training, validation, calibration development, and final evaluation.

Suggested visual: results/split_distribution.png with split sizes and a train/val/dev/eval flow.

### Slide 12 - Historical multimodal results

- Overall accuracy: 99.69%; overall F1: 0.9984.
- Overall ROC AUC: 0.9994; balanced accuracy: about 99.19%.
- Visual ROC AUC: 0.9991; audio ROC AUC: 0.9982.
- These are saved benchmark measurements, not unrestricted deployment accuracy.

Speaker notes: Cite results/test_results.json. Category imbalance matters: overall recall on real-video/fake-audio examples is 93.33%. The ablation results also do not show that every added modality improves every metric.

Suggested visual: results/roc_3tasks.png and the three-task results table.

### Slide 13 - Fine-tuned AASIST selection

- Two supplied AASIST exports use the same 10,511 validation recordings.
- 8 November, epoch 4: validation AUC 0.999649.
- 12 November, epoch 2: validation AUC 0.999331.
- Select the higher validation AUC before reserved-clip testing.

Speaker notes: The selected model is installed as models/aasist_finetuned.pth. Strong validation performance is not proof of transfer to podcast recordings. Neither export contains independent development calibration or final ASVspoof evaluation outputs.

Suggested visual: The two-row validation comparison in Section 8.4.

### Slide 14 - Current audio findings and limitations

- Fine-tuned AASIST: 3/4 fake clips detected; 1/4 missed.
- All 4/4 provisionally real clips were incorrectly flagged.
- Overall correct: 3/8; same binary outcome as pretrained AASIST.
- Generalization and recording-level calibration remain unresolved.

Speaker notes: Show the errors openly. These eight folder-labeled clips are an exploratory check, not an accuracy estimate for all audio. No threshold was adjusted to make them pass, and fine-tuning has not solved the observed false positives.

Suggested visual: A confusion-count table from the latest fine-tuned evaluation.

### Slide 15 - Demonstration and implementation contribution

- Show the model selector and upload a known-source clip.
- Explain the recording score, threshold, and calibration status.
- Click a window and play the corresponding segment.
- Highlight preprocessing fixes, provenance checks, API flow, and error reporting.

Speaker notes: Include a difficult or incorrect result so the demonstration reflects current behavior. Report that 24 relevant Python tests and 17 earlier UI tests passed, while distinguishing software behavior from detection accuracy.

Suggested visual: A sequence of three current UI screenshots and a short file/module map.

### Slide 16 - Next steps and project takeaway

- Complete independent development calibration and final ASVspoof evaluation.
- Validate on representative compressed, multilingual, and diverse-source audio.
- Improve audiovisual alignment, tracking, and runtime measurements.
- Delivered an integrated research studio; reliable arbitrary-audio verification is still a research goal.

Speaker notes: End with the concrete outcome: multimodal inference, modular audio comparison, fine-tuned checkpoint integration, and repeatable evidence. The next milestone is independently validated improvement rather than a higher score on the same eight clips.

Suggested visual: A completed / remaining checklist followed by questions.

## 13. Suggested demonstration and viva points

Demonstration: show the UI detector selector, upload a known-source clip, explain the recording score and threshold, inspect a flagged window, and play the corresponding segment. Show one difficult or inconclusive example so that the audience sees the system's limitations.

Q: Why multimodal detection? A: Manipulation can affect different streams, and their joint features provide complementary evidence.

Q: Why use Wav2Vec and AASIST? A: Wav2Vec supplies the existing learned-speech baseline; AASIST supplies a different raw-waveform graph-attention approach for comparison. Accuracy improvement must be measured.

Q: Is a model score of 0.99 equivalent to 99% certainty? A: No. Calibration and validation on representative independent data are required before using that interpretation.

Q: Why can many windows be flagged while the recording is predicted bona fide? A: The shared pipeline uses the mean score and its decision threshold; the window count is a separate diagnostic.

Q: Did this project invent AASIST? A: No. The project integrates upstream models, implements adapters and the application pipeline, and provides a fine-tuning workflow. Two fine-tuned AASIST exports were supplied and the selected checkpoint was tested locally.

Q: Is the system production ready? A: It is a research prototype. Current local audio errors and incomplete validation prevent a claim of reliable unrestricted verification.

## 14. Evidence and presentation assets

Use results/test_results.json and results/per_category.csv for historical multimodal metrics; results/ablation.csv for ablations; reports/audio_pipeline_evaluation/ for the earlier eight-clip comparison; reports/finetuned_audio_evaluation/selection.json and selected/ for the new fine-tuned results; reports/baseline_audio_benchmark/ for the native Wav2Vec evaluation; docs/audio_upgrade_execution.md for upgrade status; and the implementation files listed above for architecture details. reports/README.md indexes the retained evidence folders.

Existing figures available for slides include results/roc_3tasks.png, results/training_curves.png, results/split_distribution.png, and results/ablation.png. Use the ablation figure with the metric and experimental setup clearly identified. Add a fresh screenshot after restarting the app and loading the updated UI.

For presentation references, the upstream repositories are listed in the project documentation: FakeAVCeleb (DASH-Lab/FakeAVCeleb), AASIST (clovaai/aasist), and Wav2Vec 2.0 (facebookresearch/fairseq). This report does not substitute literature benchmark numbers for measured project results.
