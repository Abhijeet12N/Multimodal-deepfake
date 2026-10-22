"""Build a source-grounded PDF report for the Multimodal Deepfake project.

The report intentionally distinguishes checked-in measurements from published
reference results and from adapters that are implemented but not runnable in
the present CPU-only environment.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "Multimodal_Deepfake_Project_Report_October_2026.pdf"
RESULTS = ROOT / "results"
PAGE_W, PAGE_H = A4

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1864AB")
TEAL = colors.HexColor("#087F5B")
AMBER = colors.HexColor("#D97706")
RED = colors.HexColor("#B42318")
LIGHT_BLUE = colors.HexColor("#E7F5FF")
LIGHT_GREY = colors.HexColor("#F4F7FA")
MID_GREY = colors.HexColor("#52606D")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def footer(canvas: Canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(1.6 * cm, 1.45 * cm, PAGE_W - 1.6 * cm, 1.45 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(1.6 * cm, 0.95 * cm, "Multimodal Deepfake Detection Studio | Project report")
    page = str(doc.page)
    canvas.drawRightString(PAGE_W - 1.6 * cm, 0.95 * cm, f"Page {page}")
    canvas.restoreState()


def add_table(rows, widths, styles, header=True):
    table = LongTable(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
    for i in range(1 if header else 0, len(rows)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), LIGHT_GREY))
    table.setStyle(TableStyle(commands))
    return table


def figure(path: Path, width=16.8 * cm, max_height=10.5 * cm):
    if not path.exists():
        return None
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    h = min(width * ratio, max_height)
    img.drawWidth, img.drawHeight = h / ratio, h
    return img


def build() -> None:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=26,
                           leading=31, textColor=NAVY, spaceAfter=10)
    subtitle = ParagraphStyle("Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=12,
                              leading=17, textColor=MID_GREY, spaceAfter=12)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17,
                        leading=22, textColor=NAVY, spaceBefore=10, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                        leading=15, textColor=BLUE, spaceBefore=8, spaceAfter=5)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.3,
                          leading=13.2, textColor=colors.HexColor("#243B53"), spaceAfter=6)
    small = ParagraphStyle("Small", parent=body, fontSize=7.6, leading=10.0, spaceAfter=0)
    cell = ParagraphStyle("Cell", parent=body, fontSize=7.7, leading=9.7, spaceAfter=0)
    header_cell = ParagraphStyle("HeaderCell", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    callout = ParagraphStyle("Callout", parent=body, fontSize=9.2, leading=13.2,
                             leftIndent=8, rightIndent=8, borderColor=BLUE,
                             borderWidth=0.8, borderPadding=8, backColor=LIGHT_BLUE, spaceBefore=4, spaceAfter=10)
    caution = ParagraphStyle("Caution", parent=callout, borderColor=AMBER,
                             backColor=colors.HexColor("#FFF7E6"))

    story = []
    # Cover
    story += [Spacer(1, 2.2 * cm), p("MULTIMODAL DEEPFAKE DETECTION STUDIO", subtitle),
              p("Detailed Project Report", title),
              p("Architecture, model inventory, evidence, evaluation, deployment readiness, and roadmap", subtitle),
              Spacer(1, 0.9 * cm)]
    cover_rows = [
        [p("Report scope", header_cell), p("Source-grounded review of the checked-in Multimodal-deepfake repository", header_cell)],
        [p("Prepared", cell), p("22 October 2026", cell)],
        [p("Primary system", cell), p("Xception + Wav2Vec 2.0 + audiovisual SyncStream with multi-task late fusion", cell)],
        [p("Audio modernization", cell), p("Legacy Wav2Vec baseline; AASIST and AASIST-L verified; SSL-AASIST and XLSR-MamBo adapters staged for GPU environments", cell)],
        [p("Evidence basis", cell), p("Source code, configurations, JSON/CSV evaluation artifacts, model cards, audit documents, and existing result figures", cell)],
    ]
    story.append(add_table(cover_rows, [4.2 * cm, 12.6 * cm], styles))
    story += [Spacer(1, 1.0 * cm), p("Purpose", h2),
              p("This report consolidates the project as it is implemented. It explicitly separates (1) measurements stored in this repository, (2) published/upstream reference values reproduced in project documentation, and (3) integrations whose interface is coded but whose checkpoint or runtime is unavailable locally. This distinction is essential for a forensic system: high benchmark scores must not be interpreted as an unconditional guarantee on new media.", body),
              p("Decision summary: retain the multimodal fusion model for video-level analysis, retain the legacy Wav2Vec checkpoint as a reproducible baseline, and use AASIST/AASIST-L as the currently verified standalone audio upgrade. Run SSL-AASIST only in its prescribed Colab/Linux GPU environment and calibrate it on a representative development set before production use.", callout),
              Spacer(1, 3.2 * cm), p("Classification: research and decision-support system; not an autonomous forensic determination service.", small)]
    story.append(PageBreak())

    # Executive summary
    story += [p("1. Executive summary", h1),
              p("The repository implements two complementary detection paths. The multimodal path analyses video, speech, and audiovisual synchrony, emitting three task-specific logits: visual manipulation, audio manipulation, and any manipulation. The standalone path decodes an audio recording to 16 kHz mono, slices it into overlapping windows, applies voice-activity gating, scores windows with a selected detector, and merges suspicious intervals for review.", body)]
    summary_rows = [
        [p("Area", header_cell), p("Current state", header_cell), p("Evidence / interpretation", header_cell)],
        [p("Multimodal detector", cell), p("Implemented; historical evaluation artifact present", cell), p("Three-stream late fusion. Stored FakeAVCeleb test artifact reports AUC 0.9994 for the overall head across 3,276 held-out videos. The checkpoint itself is not currently present in models/.", cell)],
        [p("Existing Wav2Vec baseline", cell), p("Implemented and checkpoint verified", cell), p("Actual legacy checkpoint uses 512-dim CNN acoustic features, a 512 -> 256 -> 2 MLP, 2.0 s windows, 1.0 s hop, and a dev-calibrated spoof threshold of 0.984982.", cell)],
        [p("AASIST / AASIST-L", cell), p("Integrated, checkpointed, parity-tested", cell), p("Raw-waveform SincNet + graph-attention alternatives. AASIST-L trades some upstream benchmark accuracy for lower CPU latency.", cell)],
        [p("SSL-AASIST", cell), p("Adapter and Colab training path implemented; local execution blocked", cell), p("Requires Fairseq-compatible Linux/CUDA runtime, XLSR-53 frontend and upstream parity validation. No local measured score is claimed.", cell)],
        [p("XLSR-MamBo", cell), p("Optional adapter scaffold only", cell), p("Dedicated CUDA 12.8 / FlashAttention / Mamba environment and a compatible checkpoint are required; unsupported on this Windows CPU environment.", cell)],
    ]
    story.append(add_table(summary_rows, [3.0 * cm, 4.3 * cm, 9.5 * cm], styles))
    story += [Spacer(1, 0.4 * cm), p("Key qualification", h2),
              p("The repository contains both a legacy checkpoint contract and a newer generic Wav2Vec adapter contract. They are not interchangeable. The verified legacy wav2vec_best.pt expects 512-dimensional CNN features and two logits, whereas src/deepfake_detector/model.py defines a newer 768-dimensional transformer-hidden-state AudioClassifier with one logit. Any deployment must choose the detector/configuration pair deliberately rather than mixing these interfaces.", caution)]

    # Architecture
    story += [p("2. End-to-end architecture", h1),
              p("The multimodal model deliberately separates independently manipulable signals. This lets the verdict identify whether a face/video stream, speech/audio stream, or cross-modal synchrony is responsible rather than treating every clip as a single opaque binary class.", body)]
    flow_rows = [
        [p("Input media", header_cell), p("Preprocessing", header_cell), p("Feature streams", header_cell), p("Decision", header_cell)],
        [p("Video: MP4/MOV/AVI<br/>Audio: native track or standalone file", cell),
         p("8 sampled frames; MTCNN face crop (299 x 299); lip crop (112 x 112); 16 kHz mono; 64-band log-mel", cell),
         p("Image: Xception + temporal mean/std pooling<br/>Audio: frozen Wav2Vec 2.0 + projection<br/>Sync: 3D lip CNN + 2D mel CNN", cell),
         p("3 x 512-d concatenation -> MLP trunk -> video / audio / any heads", cell)],
    ]
    story.append(add_table(flow_rows, [4.0 * cm, 4.4 * cm, 5.2 * cm, 3.2 * cm], styles))
    story += [p("Multimodal neural components", h2)]
    comp_rows = [
        [p("Component", header_cell), p("Implementation", header_cell), p("Function", header_cell)],
        [p("ImageStream", cell), p("timm Xception; frame features are pooled by mean and standard deviation, then projected to 512-d", cell), p("Detects spatial/temporal facial manipulation evidence.", cell)],
        [p("AudioStream", cell), p("Frozen facebook/wav2vec2-base-960h; 768-d mean pooled transformer state; 768 -> 512 -> 512 projection", cell), p("Carries learned speech/acoustic evidence into fused video analysis.", cell)],
        [p("SyncStream", cell), p("3D CNN on 8 lip crops + 2D CNN on log-mel; 128 -> 512 projection", cell), p("Tests audiovisual consistency, including lip-motion/speech mismatch signals.", cell)],
        [p("Fusion / heads", cell), p("1536 -> 512 -> 256 MLP with dropout; 3 binary output heads", cell), p("Returns distinct logits for visual fake, audio fake, and overall manipulation.", cell)],
    ]
    story.append(add_table(comp_rows, [3.0 * cm, 8.0 * cm, 5.8 * cm], styles))
    story += [p("Experimental fusion extension", h2),
              p("training/cross_attention.py implements a v4 CrossAttentionDetector. It loads/freeze-warm-starts image, audio, and sync streams from v3, then applies three four-head cross-attention blocks (each modality attends to the other two), residual LayerNorm, a new fusion trunk, and the same three heads. It is an implemented training experiment; no checked-in v4 checkpoint or result artifact establishes a deployed performance claim.", body)]

    # Models detailed
    story += [p("3. Model inventory and status", h1),
              p("This inventory includes the requested existing Wav2Vec model and all newly implemented model paths visible in the detector registry and training suite. 'Verified' means code/checkpoint/evidence exists in the repository; it does not mean the model is universally calibrated for arbitrary media.", body)]
    inv_rows = [
        [p("Model", header_cell), p("Architecture / input", header_cell), p("Status", header_cell), p("Deployment position", header_cell)],
        [p("Legacy Wav2Vec 2.0 + MLP", cell), p("Frozen Wav2Vec CNN extractor; mean-pool 512-d features; MLP 512 -> 256 -> 2; 2.0 s windows", cell), p("Verified checkpoint and native/sliding-window artifacts", cell), p("Baseline and compatibility reference", cell)],
        [p("Multimodal late fusion", cell), p("Xception + Wav2Vec 2.0 transformer + 3D/2D SyncStream; three heads", cell), p("Implemented; test/ablation artifacts present; current weight file absent from models/", cell), p("Video and audiovisual analysis", cell)],
        [p("AASIST", cell), p("Raw 16 kHz waveform; 70-filter Sinc frontend; spectro-temporal graph attention + master node; 4.04 s windows", cell), p("Integrated, official checkpoint available, upstream parity reported", cell), p("Active verified standalone audio upgrade", cell)],
        [p("AASIST-L", cell), p("Compact AASIST raw-waveform graph-attention variant; 4.04 s windows", cell), p("Integrated, official lightweight checkpoint available", cell), p("Lower-latency standalone alternative", cell)],
        [p("SSL-AASIST", cell), p("XLSR-53 (Wav2Vec 2.0 multilingual SSL) + weighted layer aggregation + AASIST backend; RawBoost training", cell), p("Registry adapter, config and Colab training script implemented; no compatible local runtime/checkpoint measurement", cell), p("Primary GPU fine-tuning candidate", cell)],
        [p("XLSR-MamBo", cell), p("XLSR SSL frontend + bidirectional Mamba state-space backend", cell), p("Optional adapter implemented; CUDA-specific dependencies missing", cell), p("Research challenger only", cell)],
        [p("Cross-attention v4", cell), p("Cross-modal attention over frozen image/audio/sync embeddings", cell), p("Training implementation only; no checked-in trained output", cell), p("Multimodal research experiment", cell)],
    ]
    story.append(add_table(inv_rows, [2.8 * cm, 5.5 * cm, 4.4 * cm, 4.1 * cm], styles))
    story += [p("3.1 Existing Wav2Vec baseline: exact operational contract", h2),
              p("The legacy wav2vec_best.pt is the reproducible baseline, not merely a conceptual Wav2Vec model. Audio is resampled to 16 kHz mono and framed into 32,000 samples (2.0 seconds) with a 16,000-sample hop. Windows below RMS 0.015 are gated as non-speech. The backbone feature extractor is mean-pooled to a 512-dimensional vector, classified by Linear(512,256), BatchNorm, ReLU, Dropout(0.3), Linear(256,2), and scored as softmax class index 1 (spoof). The calibrated operating threshold is 0.9849822521, selected on ASVspoof 2019 LA development data.", body),
              p("The newer generic AudioClassifier source is materially different: it uses mean-pooled 768-dimensional transformer hidden states, LayerNorm, GELU layers, and one logit. It must be paired only with an aligned audio_model.pt-type checkpoint, not with wav2vec_best.pt.", caution)]
    story += [p("3.2 Newly integrated graph-attention models", h2),
              p("AASIST and AASIST-L receive raw 16 kHz waveform windows of exactly 64,600 samples (4.0375 seconds), repeat-pad shorter recordings, advance at a 32,000-sample (2.0 second) hop, and use a two-logit output convention: spoof index 0 and bonafide index 1. Their native score orientation is higher_is_bonafide, so a fake verdict arises below the bonafide threshold. The service intentionally does not invent a fake probability from uncalibrated logits.", body)]
    story.append(PageBreak())

    # Audio operational pipeline
    story += [p("4. Standalone audio detection service", h1),
              p("The detector registry exposes legacy_wav2vec_mlp, aasist, aasist_l, ssl_aasist, and xlsr_mambo behind an AudioDetector interface. This isolates model-specific score conventions and input contracts from the user-facing pipeline.", body)]
    steps_rows = [
        [p("Step", header_cell), p("Operational behavior", header_cell), p("Why it matters", header_cell)],
        [p("1. Decode", cell), p("soundfile first; imageio-ffmpeg and librosa fallbacks; convert to 16 kHz mono normalized waveform", cell), p("Supports common uploaded formats without requiring torchaudio.", cell)],
        [p("2. Window", cell), p("Generate overlapping fixed-length windows using detector requirements; zero padding for legacy, repeat padding for AASIST-family", cell), p("Preserves the original model's input contract.", cell)],
        [p("3. Gate", cell), p("RMS VAD normally excludes low-energy windows; falls back to all windows when whole recording is quiet", cell), p("Avoids silence dominating recording-level evidence.", cell)],
        [p("4. Score", cell), p("Batched detector inference returns raw scores, orientation, per-window fake score, and optional bonafide score", cell), p("Prevents class-index/score-direction errors between models.", cell)],
        [p("5. Aggregate", cell), p("Mean recording score; top suspicious windows; merge consecutive flagged intervals", cell), p("Produces an analyst-readable timeline rather than a single unexplained label.", cell)],
        [p("6. Report", cell), p("Verdict, threshold, duration, windows, intervals, warnings, model identity/hash prefix, timing", cell), p("Enables traceability and UI/CLI/JSON consumption.", cell)],
    ]
    story.append(add_table(steps_rows, [2.0 * cm, 8.0 * cm, 6.8 * cm], styles))
    story += [p("Score semantics are a safety-critical boundary", h2),
              p("Legacy Wav2Vec produces a spoof probability where higher means more fake. AASIST, AASIST-L, SSL-AASIST, and XLSR-MamBo expose bonafide-oriented logits where lower means more fake. The service reads score_orientation on every prediction and reverses the decision rule accordingly. This design is tested to protect against a common but severe integration error: treating a bonafide score as a fake probability.", callout),
              p("User-facing integration: Flask provides status plus upload endpoints for multimodal video and standalone audio; scripts/analyze_video.py provides command-line analysis and JSON export. If the relevant checkpoint is missing, the UI is designed to report availability rather than silently return a fabricated result.", body)]

    # Data and protocol
    story += [p("5. Data governance and evaluation protocol", h1),
              p("The audio evaluation protocol uses a nine-column manifest schema: sample_id, path, label, dataset, split, speaker_id, source_id, attack_type, and language. The canonical label convention is 0 = genuine and 1 = fake. The project distinguishes native benchmark inference from end-to-end sliding-window recording inference, preventing application-processing artifacts from being confused with raw model discrimination.", body)]
    data_rows = [
        [p("Dataset evidence", header_cell), p("Count / split", header_cell), p("Notes", header_cell)],
        [p("ASVspoof 2019 LA", cell), p("121,461 total: 25,380 train; 24,844 dev; 71,237 eval", cell), p("12,483 genuine / 108,978 fake in stored summary; eval has unseen A07-A19 attacks.", cell)],
        [p("Local evaluation", cell), p("6 recordings: 4 genuine, 2 fake", cell), p("Small, in-the-wild compressed-media sanity set only; unsuitable for firm population metrics.", cell)],
        [p("Leakage audit", cell), p("PASS", cell), p("Stored audit reports no cross-split speaker, source, or exact-duplicate overlap in the audited manifests.", cell)],
        [p("Multimodal FakeAVCeleb test", cell), p("3,276 videos / 75 unique identities", cell), p("Reported in README/result artifact as subject-disjoint held-out evaluation.", cell)],
    ]
    story.append(add_table(data_rows, [4.0 * cm, 5.2 * cm, 7.6 * cm], styles))
    story += [p("Protocol controls", h2),
              p("Threshold selection is limited to the development split and is then applied to evaluation data without label peeking. The leakage checker also records model training provenance. These practices preserve a meaningful separation between fitting, calibration, and evaluation. The data are heavily spoof-skewed (approximately 89.72% fake in the stored summary), so ROC AUC, PR AUC, EER, genuine false-positive rate, and per-attack recall should be read alongside accuracy.", body)]
    story.append(PageBreak())

    # Results
    story += [p("6. Measured evaluation results", h1),
              p("Numbers in this section are reproduced from checked-in JSON/CSV artifacts. They are not fresh reruns performed for this report. Published reference values are called out separately.", body)]
    multi_rows = [
        [p("Head", header_cell), p("Accuracy", header_cell), p("Balanced accuracy", header_cell), p("ROC AUC", header_cell), p("F1", header_cell), p("Recall", header_cell)],
        [p("Video", cell), p("0.9988", cell), p("0.9994", cell), p("0.9991", cell), p("0.9994", cell), p("0.9987", cell)],
        [p("Audio", cell), p("0.9869", cell), p("0.9869", cell), p("0.9982", cell), p("0.9875", cell), p("0.9860", cell)],
        [p("Any / overall", cell), p("0.9969", cell), p("0.9919", cell), p("0.9994", cell), p("0.9984", cell), p("0.9972", cell)],
    ]
    story += [p("6.1 Multimodal historical test artifact (FakeAVCeleb)", h2), add_table(multi_rows, [3.5 * cm, 2.5 * cm, 3.2 * cm, 2.5 * cm, 2.3 * cm, 2.3 * cm], styles),
              p("The artifact is associated with epoch 3. The overall head performs strongly across categories, but RealVideo-FakeAudio is the weakest stored category for overall accuracy (0.9333, n=75); this is consistent with audio-only manipulation being harder when visual evidence remains genuine. Interpret the very high scores in the context of the dataset and split, not as a claim of universal robustness.", body)]
    fig = figure(RESULTS / "roc_3tasks.png")
    if fig:
        story += [fig, p("Figure 1. Stored ROC curves for video, audio, and overall tasks.", small)]
    story += [p("6.2 Multimodal ablation artifact", h2)]
    abl_rows = [
        [p("Configuration", header_cell), p("Overall AUC", header_cell), p("Overall F1", header_cell), p("Interpretation", header_cell)],
        [p("Full multimodal", cell), p("0.9994", cell), p("0.9984", cell), p("Best stored complete configuration.", cell)],
        [p("No image", cell), p("0.7618", cell), p("0.7070", cell), p("Largest measured degradation; visual stream carries strong signal in this dataset.", cell)],
        [p("No audio", cell), p("0.9993", cell), p("0.9987", cell), p("Near-parity on this data; does not establish audio is unnecessary for audio-only attacks outside the split.", cell)],
        [p("No sync", cell), p("0.9931", cell), p("0.9877", cell), p("Synchrony stream adds measurable support.", cell)],
        [p("Only audio", cell), p("0.6393", cell), p("0.4493", cell), p("Audio alone is insufficient for the predominant video-fake test composition.", cell)],
    ]
    story.append(add_table(abl_rows, [3.6 * cm, 2.8 * cm, 2.6 * cm, 7.3 * cm], styles))
    story.append(PageBreak())

    # results audio
    story += [p("7. Audio model results and interpretation", h1),
              p("The legacy Wav2Vec evaluation used its native 512-dimensional feature benchmark path and a threshold selected on the ASVspoof development partition. It is therefore the clearest comparable local baseline. Its 7.13% EER on the held-out ASVspoof 2019 LA evaluation split is paired with high AUC (0.9815) but meaningful weak-attack failure modes.", body)]
    legacy_rows = [
        [p("Metric (ASVspoof 2019 LA eval, n=71,237)", header_cell), p("Legacy Wav2Vec + MLP", header_cell)],
        [p("EER", cell), p("7.13%", cell)],
        [p("Accuracy at dev-calibrated threshold", cell), p("84.90%", cell)],
        [p("ROC AUC / PR AUC", cell), p("0.9815 / 0.9978", cell)],
        [p("Spoof recall / genuine FPR", cell), p("83.33% / 1.47%", cell)],
        [p("Confusion matrix", cell), p("TN 7,247; FP 108; FN 10,647; TP 53,235", cell)],
    ]
    story.append(add_table(legacy_rows, [9.0 * cm, 7.3 * cm], styles))
    weak_rows = [
        [p("Unseen attack", header_cell), p("Spoof recall", header_cell), p("Attack EER", header_cell), p("Reading", header_cell)],
        [p("A07-A14", cell), p("96.8%-99.9% (except none below 96.8%)", cell), p("0.37%-2.07%", cell), p("Strong on several vocoder/TTS/VC variants.", cell)],
        [p("A15", cell), p("88.0%", cell), p("4.55%", cell), p("Performance begins to decline.", cell)],
        [p("A16", cell), p("88.2%", cell), p("5.29%", cell), p("Waveform-concatenation vulnerability.", cell)],
        [p("A17", cell), p("27.1%", cell), p("18.93%", cell), p("Major failure mode; direct concatenation.", cell)],
        [p("A18", cell), p("38.1%", cell), p("13.42%", cell), p("Major voice-conversion failure mode.", cell)],
        [p("A19", cell), p("52.8%", cell), p("12.67%", cell), p("Major voice-conversion failure mode.", cell)],
    ]
    story += [p("Legacy per-attack diagnostic", h2), add_table(weak_rows, [3.0 * cm, 3.8 * cm, 2.8 * cm, 6.7 * cm], styles)]
    story += [p("AASIST comparison: evidence levels", h2)]
    audio_rows = [
        [p("Measure", header_cell), p("Legacy", header_cell), p("AASIST", header_cell), p("AASIST-L", header_cell)],
        [p("ASVspoof 2019 LA eval EER", cell), p("7.13% (stored local artifact)", cell), p("0.83% (published/upstream reference)", cell), p("0.99% (published/upstream reference)", cell)],
        [p("Local 6-recording ROC AUC", cell), p("0.5000", cell), p("0.7500", cell), p("0.7500", cell)],
        [p("Local 6-recording EER", cell), p("75.00%", cell), p("50.00%", cell), p("50.00%", cell)],
        [p("Local CPU latency / recording", cell), p("1,368 ms", cell), p("6,278 ms", cell), p("3,112 ms", cell)],
    ]
    story.append(add_table(audio_rows, [4.0 * cm, 4.1 * cm, 4.1 * cm, 4.1 * cm], styles))
    story += [p("The local set has only six recordings, so its AUC/EER values are diagnostic—not deployable benchmark estimates. It does, however, reveal a real domain-shift issue: the legacy score ranking collapses on compressed web audio, while AASIST-family scores provide better ranking separation but have a downward bonafide-logit shift at the standard 0.0 operating threshold. A calibrated development set from the target media domain is required before hard decisions are trusted.", caution)]
    story.append(PageBreak())

    # deployment
    story += [p("8. Runtime, checkpoints, and deployment readiness", h1)]
    runtime_rows = [
        [p("Asset", header_cell), p("Evidence", header_cell), p("Readiness", header_cell)],
        [p("wav2vec_best.pt", cell), p("666,067 bytes; SHA-256 efba2b1fadcb602b928b49496d6bd0ce9786fc10f8d28f76a6de77dfd378159f", cell), p("Verified legacy baseline checkpoint under initial setup/deepfake_phase1/checkpoints/.", cell)],
        [p("AASIST.pth", cell), p("1,281,532 bytes; SHA-256 51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0", cell), p("Verified standalone upgraded backend.", cell)],
        [p("AASIST-L.pth", cell), p("426,428 bytes; SHA-256 814331d088032bb4c3fa61cc014789eadeed464209dd094ab3a2dd6ffbdce27a", cell), p("Verified low-latency standalone backend.", cell)],
        [p("models/best_model.pt and audio_model.pt", cell), p("Documented as expected paths but models/ currently holds only README.md", cell), p("Multimodal and generic standalone Wav2Vec service needs checkpoint placement/configuration before live deployment.", cell)],
        [p("SSL-AASIST / XLSR-MamBo", cell), p("No local compatible runtime/checkpoint assertion", cell), p("Use isolated GPU environment; do not expose as a ready production choice locally.", cell)],
    ]
    story.append(add_table(runtime_rows, [3.6 * cm, 7.0 * cm, 5.6 * cm], styles))
    story += [p("Environment constraints", h2),
              p("The audited local environment is Windows 11, Python 3.13, CPU-only PyTorch, with no CUDA device. Legacy Wav2Vec and AASIST-family CPU inference are feasible, though AASIST is slower. SSL-AASIST requires the upstream Fairseq/XLSR lineage and is blocked locally by Python/Fairseq packaging and C++ extension constraints; its designated execution target is Linux Google Colab with CUDA and at least 12 GB VRAM. XLSR-MamBo additionally needs CUDA 12.8, FlashAttention, Mamba-SSM, causal-conv1d, and Hydra.", body),
              p("Production recommendation", h2),
              p("For current deployment, wire checkpoint paths explicitly and surface model status in the UI. Use AASIST-L where CPU latency matters and AASIST where its full model is acceptable. Keep the legacy Wav2Vec result available for baseline comparison, but do not treat it as robust to all voice-conversion/concatenation attacks. Escalate uncertain or high-impact cases to analyst review with the interval report and multimodal evidence.", callout)]

    # limitations roadmap
    story += [p("9. Risks, limitations, and prioritized roadmap", h1)]
    risk_rows = [
        [p("Risk / limitation", header_cell), p("Observed implication", header_cell), p("Mitigation", header_cell)],
        [p("Distribution shift", cell), p("Compressed social/web audio shifts AASIST logit distributions; local set is tiny.", cell), p("Collect representative dev audio; calibrate thresholds; evaluate by codec, language, device, and attack family.", cell)],
        [p("Legacy blind spots", cell), p("Low recall on A17-A19 attack types.", cell), p("Adopt AASIST as current alternative; execute SSL-AASIST/RawBoost evaluation on GPU.", cell)],
        [p("Missing checkpoints", cell), p("UI can be unavailable for multimodal/generic audio paths.", cell), p("Version and hash approved weights; configure MODEL_PATH and AUDIO_MODEL_PATH; add startup verification.", cell)],
        [p("Score comparability", cell), p("Softmax probability and raw bonafide logits are not directly comparable.", cell), p("Preserve orientation metadata; calibrate each backend independently; never average incompatible scores.", cell)],
        [p("Forensic misuse", cell), p("False positives/negatives can affect people and decisions.", cell), p("Use as decision support; retain provenance; provide uncertainty and human review.", cell)],
    ]
    story.append(add_table(risk_rows, [4.0 * cm, 6.0 * cm, 6.2 * cm], styles))
    story += [p("Recommended next sequence", h2),
              p("1. Make the existing checkpoint inventory deployable: configure absolute/approved paths, perform a startup load test, and retain checkpoint hashes in every report. 2. Create a larger representative local-development set separated into calibration and final test partitions. 3. Calibrate AASIST/AASIST-L operating points on development data, including codec-aware validation. 4. Run the supplied SSL-AASIST Colab workflow with upstream Fairseq parity checks, RawBoost training, and held-out ASVspoof 2021 / in-the-wild evaluation. 5. Benchmark latency/throughput and uncertainty by audio length. 6. Train/evaluate v4 cross-attention only after the baseline multimodal checkpoint and dataset split are reproducibly packaged.", body)]
    story += [p("10. Conclusion", h1),
              p("The project is a well-structured multimodal deepfake-detection research system with a usable separation between media preprocessing, model backends, scoring semantics, reporting, and UI/CLI integration. Its strongest present engineering achievement is the evidence-aware modernization of standalone audio detection: the legacy Wav2Vec model is preserved with its true contract, AASIST and AASIST-L are integrated with parity and provenance evidence, and advanced SSL-based candidates are staged without being misrepresented as locally complete. The highest-value remaining work is deployment packaging, domain-specific calibration, and GPU evaluation of the SSL-AASIST path.", body),
              p("Appendix: primary repository evidence consulted", h2)]
    evidence_rows = [
        [p("Area", header_cell), p("Key files", header_cell)],
        [p("Core implementation", cell), p("src/deepfake_detector/model.py; audio_detection/service.py; audio_detection/registry.py; detector adapters", cell)],
        [p("Configuration", cell), p("configs/audio/legacy.yaml; aasist.yaml; ssl_aasist.yaml", cell)],
        [p("Measured artifacts", cell), p("results/test_results.json; results/ablation.csv; reports/legacy_eval_report.json; sliding-window JSON reports", cell)],
        [p("Governance/audits", cell), p("docs/current_system_audit.md; docs/evaluation_protocol.md; docs/model_card.md; reports/leakage_audit.json", cell)],
        [p("Training extensions", cell), p("training/cross_attention.py; training/train_ssl_aasist_colab.py; notebook_v2/colab_ssl_aasist_quickstart.md", cell)],
    ]
    story.append(add_table(evidence_rows, [4.0 * cm, 12.2 * cm], styles))

    doc = __import__("reportlab.platypus", fromlist=["SimpleDocTemplate"]).SimpleDocTemplate(
        str(OUT), pagesize=A4, rightMargin=1.6 * cm, leftMargin=1.6 * cm,
        topMargin=1.5 * cm, bottomMargin=1.9 * cm, title="Multimodal Deepfake Detection Studio - Project Report",
        author="Multimodal Deepfake Detection Studio",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    build()
