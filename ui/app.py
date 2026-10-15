"""Flask Web Application for Multimodal Deepfake Detection Studio."""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# Ensure the core src package is accessible
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepfake_detector import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    DeepfakePredictor,
    StandaloneAudioPredictor,
    build_report,
    category_from_path,
    interpret,
)

# Configuration from environment or defaults
MODEL_PATH = os.environ.get("MODEL_PATH", "").strip() or str(PROJECT_ROOT / "models" / "best_model.pt")
AUDIO_MODEL_PATH = os.environ.get("AUDIO_MODEL_PATH", "").strip() or str(PROJECT_ROOT / "models" / "audio_model.pt")
SAMPLES_DIR = HERE / "samples"
UPLOADS_DIR = HERE / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Historical multimodal test set evaluation benchmark metrics
AGGREGATE_METRICS = {
    "test_n": 3276,
    "video_auc": 0.9991,
    "video_f1": 0.9994,
    "audio_auc": 0.9982,
    "audio_f1": 0.9875,
    "any_auc": 0.9994,
    "any_f1": 0.9984,
    "per_category": {
        "R-R": {"n": 75, "any_recall": None, "label": "Real video and real audio (control)"},
        "F-R": {"n": 1485, "any_recall": 0.9973, "label": "Fake video, real audio (Wav2Lip)"},
        "R-F": {"n": 75, "any_recall": 0.9333, "label": "Real video, fake audio (TTS)"},
        "F-F": {"n": 1641, "any_recall": 1.0000, "label": "Both video and audio are fake"},
    },
}

app = Flask(
    __name__,
    template_folder=str(HERE / "templates"),
    static_folder=str(HERE / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB upload limit

predictor: DeepfakePredictor | None = None
predictor_error: str | None = None
audio_predictor: StandaloneAudioPredictor | None = None
audio_predictor_error: str | None = None


def allowed_video(filename: str) -> bool:
    """Validate video file extension."""
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_audio(filename: str) -> bool:
    """Validate audio file extension."""
    return Path(filename).suffix.lower() in ALLOWED_AUDIO_EXTENSIONS


def safe_child(base: Path, user_path: str | Path) -> Path | None:
    """Resolve a user-controlled path and ensure it stays under base directory."""
    candidate = (base / user_path).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def load_predictor() -> DeepfakePredictor | None:
    """Lazy-load the multimodal video predictor singleton."""
    global predictor, predictor_error
    if predictor is not None or predictor_error is not None:
        return predictor

    model_file = Path(MODEL_PATH).expanduser()
    if not model_file.exists():
        predictor_error = f"Model checkpoint not found: {model_file}"
        return None

    try:
        print("[app] Loading multimodal detector checkpoint (~10-15s)...")
        predictor = DeepfakePredictor(str(model_file))
        print("[app] Multimodal detector ready.")
    except Exception as exc:
        predictor_error = f"Model load failed: {exc}"
        print(f"[app] {predictor_error}")
    return predictor


def load_audio_predictor() -> StandaloneAudioPredictor | None:
    """Lazy-load the standalone audio predictor singleton."""
    global audio_predictor, audio_predictor_error
    if audio_predictor is not None or audio_predictor_error is not None:
        return audio_predictor

    model_file = Path(AUDIO_MODEL_PATH).expanduser()
    if not model_file.exists():
        audio_predictor_error = f"Audio checkpoint not found: {model_file}"
        return None

    try:
        print("[app] Loading standalone audio checkpoint...")
        audio_predictor = StandaloneAudioPredictor(str(model_file))
        print("[app] Standalone audio detector ready.")
    except Exception as exc:
        audio_predictor_error = f"Audio model load failed: {exc}"
        print(f"[app] {audio_predictor_error}")
    return audio_predictor


def list_samples() -> list[dict]:
    """Scan samples directory and extract ground-truth categories from file paths."""
    out = []
    for p in sorted(SAMPLES_DIR.glob("**/*")):
        if not p.is_file() or p.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
            continue
        rel = p.relative_to(SAMPLES_DIR).as_posix()
        cat, vfake, afake, any_fake = category_from_path(str(p))
        out.append({
            "name": p.name,
            "rel": rel,
            "category": cat,
            "video_truth": vfake,
            "audio_truth": afake,
            "any_truth": any_fake,
        })
    return out


@app.route("/")
def index():
    """Main dashboard page."""
    pred = load_predictor()
    audio_pred = load_audio_predictor()
    return render_template(
        "index.html",
        samples=list_samples(),
        aggregate=AGGREGATE_METRICS,
        model_ready=pred is not None,
        model_error=predictor_error,
        model_path=MODEL_PATH,
        audio_model_ready=audio_pred is not None,
        audio_model_error=audio_predictor_error,
        audio_model_path=AUDIO_MODEL_PATH,
        model_device=str(pred.device) if pred else "not loaded",
    )


@app.route("/api/status")
def status():
    """System and model status API endpoint."""
    pred = load_predictor()
    audio_pred = load_audio_predictor()
    return jsonify({
        "status": "online",
        "multimodal_model": {
            "loaded": pred is not None,
            "path": MODEL_PATH,
            "error": predictor_error,
            "device": str(pred.device) if pred else None,
        },
        "audio_model": {
            "loaded": audio_pred is not None,
            "path": AUDIO_MODEL_PATH,
            "error": audio_predictor_error,
        },
    })


@app.route("/samples/<path:filename>")
def serve_sample(filename: str):
    """Serve sample media clip."""
    path = safe_child(SAMPLES_DIR, filename)
    if path is None or not path.exists() or path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({"ok": False, "error": "Sample not found"}), 404
    return send_from_directory(SAMPLES_DIR, path.relative_to(SAMPLES_DIR).as_posix())


@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    """Serve user-uploaded media clip."""
    path = safe_child(UPLOADS_DIR, filename)
    if path is None or not path.exists() or path.suffix.lower() not in (ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({"ok": False, "error": "Upload not found"}), 404
    return send_from_directory(UPLOADS_DIR, path.relative_to(UPLOADS_DIR).as_posix())


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analyze a chosen sample or an uploaded video file."""
    pred = load_predictor()
    if pred is None:
        return jsonify({"ok": False, "error": predictor_error or "Model is not available"}), 503

    mode = request.form.get("mode", "sample")
    if mode == "sample":
        sample_name = request.form.get("sample", "").strip()
        if not allowed_video(sample_name):
            return jsonify({"ok": False, "error": "Unsupported sample video format"}), 400
        video_path = safe_child(SAMPLES_DIR, sample_name)
        if video_path is None or not video_path.exists():
            return jsonify({"ok": False, "error": f"Sample not found: {sample_name}"}), 404
        ground = category_from_path(str(video_path))
        video_url = f"/samples/{sample_name}"
    elif mode == "upload":
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"ok": False, "error": "No video file was uploaded"}), 400
        if not allowed_video(f.filename):
            return jsonify({"ok": False, "error": "Unsupported video format. Allowed: MP4, MOV, AVI, MKV"}), 400
        clean_name = secure_filename(Path(f.filename).name)
        if not clean_name:
            return jsonify({"ok": False, "error": "Invalid filename"}), 400
        safe = f"{uuid.uuid4().hex[:8]}_{clean_name}"
        video_path = UPLOADS_DIR / safe
        f.save(video_path)
        ground = category_from_path(f.filename)
        video_url = f"/uploads/{safe}"
    else:
        return jsonify({"ok": False, "error": f"Unknown mode: {mode}"}), 400

    t0 = time.time()
    scores, info = pred.predict(str(video_path))
    if scores is None:
        return jsonify({
            "ok": False,
            "error": "Preprocessing failed: " + str(info.get("fail_reason", "unknown")),
            "info": info,
        }), 422

    decision = interpret(scores)

    # Match ground truth if identifiable
    truth = None
    correct = None
    if ground[0]:
        truth = {
            "category": ground[0],
            "video_fake": bool(ground[3]) and (ground[1] == "fake"),
            "audio_fake": bool(ground[3]) and (ground[2] == "fake"),
            "any_fake": bool(ground[3]),
        }
        v_pred = scores["video"] > 0.5
        a_pred = scores["audio"] > 0.5
        y_pred = scores["any"] > 0.5
        correct = {
            "video": v_pred == truth["video_fake"],
            "audio": a_pred == truth["audio_fake"],
            "any": y_pred == truth["any_fake"],
        }

    report = build_report(scores, decision, info, truth)

    return jsonify({
        "ok": True,
        "video_url": video_url,
        "scores": scores,
        "decision": decision,
        "report": report,
        "info": info,
        "ground_truth": truth,
        "correct": correct,
        "elapsed_total": round(time.time() - t0, 2),
    })


@app.route("/api/analyze-audio", methods=["POST"])
def analyze_audio():
    """Analyze an uploaded audio recording with the standalone sliding-window audio detector."""
    pred = load_audio_predictor()
    if pred is None:
        return jsonify({"ok": False, "error": audio_predictor_error or "Audio model is not available"}), 503

    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"ok": False, "error": "Please select an audio file to analyze"}), 400

    extension = Path(f.filename).suffix.lower()
    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        return jsonify({"ok": False, "error": "Supported audio formats: WAV, MP3, M4A, FLAC, OGG, AAC"}), 400

    clean_name = secure_filename(Path(f.filename).name)
    if not clean_name:
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    saved_name = f"audio_{uuid.uuid4().hex[:8]}_{clean_name}"
    audio_path = UPLOADS_DIR / saved_name
    f.save(audio_path)

    try:
        result = pred.predict(str(audio_path))
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Audio analysis failed: {exc}"}), 422

    return jsonify({
        "ok": True,
        "filename": clean_name,
        "result": result,
    })


def main():
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    print(f"Starting Deepfake Detection Studio at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()

