"""Flask Web Application for Multimodal Deepfake Detection Studio."""
from __future__ import annotations

import os
import sys
import threading
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


def load_dotenv_fallback(env_path: Path | None = None) -> None:
    """Load key-value pairs from .env into os.environ if not already set."""
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                # Clean windows escaped backslashes if present
                v = v.replace("\\\\", "\\")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as exc:
        print(f"[app] Note: Could not read .env file: {exc}")


# Load environment variables from .env
load_dotenv_fallback()

from deepfake_detector import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    DeepfakePredictor,
    StandaloneAudioPredictor,
    build_report,
    category_from_path,
    interpret,
)
from deepfake_detector.audio_detection import StandaloneAudioService, get_detector
from deepfake_detector.upstream import default_aasist_checkpoint


def resolve_model_path(env_key: str, default_rel: str) -> str:
    """Resolve model path from environment or default, falling back safely if configured path is missing."""
    val = os.environ.get(env_key, "").strip().strip("\"'")
    if val:
        candidate = Path(val).expanduser()
        if candidate.exists():
            return str(candidate)
        print(f"[app] Configured {env_key}='{val}' not found on disk. Checking default project location...")

    default_path = PROJECT_ROOT / default_rel
    if default_path.exists():
        return str(default_path)
    return val or str(default_path)


# Configuration from environment or defaults
MODEL_PATH = resolve_model_path("MODEL_PATH", "models/best_model.pt")
DEFAULT_AUDIO_DETECTOR = "legacy_standalone"
AUDIO_MODEL_PATH = resolve_model_path("AUDIO_MODEL_PATH", "models/audio_model.pt")
LEGACY_WAV2VEC_PATH = os.environ.get("LEGACY_AUDIO_CHECKPOINT_PATH", "").strip().strip("'\"")
if LEGACY_WAV2VEC_PATH:
    _legacy_path = Path(LEGACY_WAV2VEC_PATH).expanduser()
    LEGACY_WAV2VEC_PATH = str(_legacy_path if _legacy_path.is_absolute() else PROJECT_ROOT / _legacy_path)
else:
    LEGACY_WAV2VEC_PATH = str(
        PROJECT_ROOT.parent / "initial setup" / "deepfake_phase1" / "checkpoints" / "wav2vec_best.pt"
    )
def resolve_aasist_path(project_root: Path, model_id: str, override: str = "") -> Path:
    """Prefer an installed fine-tuned checkpoint while retaining explicit overrides."""
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else project_root / path
    return default_aasist_checkpoint(project_root, model_id)


AASIST_PATHS = {}
for _model_id, _filename in (("aasist", "AASIST.pth"), ("aasist_l", "AASIST-L.pth")):
    _key = "AASIST_CHECKPOINT_PATH" if _model_id == "aasist" else "AASIST_L_CHECKPOINT_PATH"
    _override = os.environ.get(_key, "").strip().strip("'\"")
    AASIST_PATHS[_model_id] = str(resolve_aasist_path(PROJECT_ROOT, _model_id, _override))
if (Path(AASIST_PATHS["aasist"]).is_file()
        and Path(AASIST_PATHS["aasist"]).stem.endswith("_finetuned")):
    DEFAULT_AUDIO_DETECTOR = "aasist"
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
audio_predictor: StandaloneAudioService | StandaloneAudioPredictor | None = None
audio_predictor_error: str | None = None
audio_predictor_id: str | None = None
predictor_load_lock = threading.Lock()
audio_predictor_load_lock = threading.Lock()
audio_inference_lock = threading.Lock()


def prune_old_uploads(max_age_seconds: int = 7200, max_files: int = 40) -> None:
    """Prune uploads older than max_age_seconds or when total count exceeds max_files."""
    try:
        files = sorted(UPLOADS_DIR.glob("*"), key=lambda p: p.stat().st_mtime)
        now = time.time()
        for p in files:
            if not p.is_file():
                continue
            if (now - p.stat().st_mtime) > max_age_seconds or len(files) > max_files:
                try:
                    p.unlink(missing_ok=True)
                    files.remove(p)
                except OSError:
                    pass
    except Exception:
        pass


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


def load_predictor(force_retry: bool = False) -> DeepfakePredictor | None:
    """Lazy-load the multimodal video predictor singleton with graceful retry."""
    global predictor, predictor_error
    if predictor is not None and not force_retry:
        return predictor

    # The index page and status endpoint can arrive together. Serializing
    # initialization prevents duplicate 476 MB model loads on CPU.
    with predictor_load_lock:
        if predictor is not None and not force_retry:
            return predictor

        model_file = Path(MODEL_PATH).expanduser()
        if not model_file.exists():
            fallback = PROJECT_ROOT / "models" / "best_model.pt"
            if fallback.exists():
                model_file = fallback

        if not model_file.exists():
            predictor_error = f"Model checkpoint not found: {model_file}"
            return None

        try:
            print("[app] Loading multimodal detector checkpoint (~10-15s)...")
            predictor = DeepfakePredictor(str(model_file))
            predictor_error = None
            print("[app] Multimodal detector ready.")
        except Exception as exc:
            predictor_error = f"Model load failed: {exc}"
            print(f"[app] {predictor_error}")
    return predictor


def audio_detector_options() -> list[dict[str, str | bool]]:
    """Return UI detector choices and whether their checkpoint files are present."""
    choices = [
        ("legacy_standalone", "Wav2Vec 2.0 + MLP (Studio)", AUDIO_MODEL_PATH),
        ("legacy_wav2vec_mlp", "Wav2Vec 2.0 + MLP (Phase 1)", LEGACY_WAV2VEC_PATH),
        ("aasist", "AASIST" + (" (fine-tuned)" if Path(AASIST_PATHS["aasist"]).stem.endswith("_finetuned") else ""), AASIST_PATHS["aasist"]),
        ("aasist_l", "AASIST-L" + (" (fine-tuned)" if Path(AASIST_PATHS["aasist_l"]).stem.endswith("_finetuned") else ""), AASIST_PATHS["aasist_l"]),
    ]
    return [
        {"id": model_id, "label": label, "available": Path(path).expanduser().is_file()}
        for model_id, label, path in choices
    ]


def audio_checkpoint_path(detector_id: str) -> str:
    return {
        "legacy_standalone": AUDIO_MODEL_PATH,
        "legacy_wav2vec_mlp": LEGACY_WAV2VEC_PATH,
        "aasist": AASIST_PATHS["aasist"],
        "aasist_l": AASIST_PATHS["aasist_l"],
    }[detector_id]


def load_audio_predictor(
    detector_id: str = DEFAULT_AUDIO_DETECTOR,
    force_retry: bool = False,
) -> StandaloneAudioService | StandaloneAudioPredictor | None:
    """Load the selected allowlisted audio detector, replacing the active model if needed."""
    global audio_predictor, audio_predictor_error, audio_predictor_id
    if detector_id not in {"legacy_standalone", "legacy_wav2vec_mlp", "aasist", "aasist_l"}:
        audio_predictor_error = f"Unknown audio detector: {detector_id}"
        return None
    if audio_predictor is not None and audio_predictor_id == detector_id and not force_retry:
        return audio_predictor

    with audio_predictor_load_lock:
        if audio_predictor is not None and audio_predictor_id == detector_id and not force_retry:
            return audio_predictor

        audio_predictor = None
        audio_predictor_id = None
        audio_predictor_error = None
        model_path = audio_checkpoint_path(detector_id)
        model_file = Path(model_path).expanduser()

        if not model_file.exists():
            audio_predictor_error = f"Audio checkpoint not found: {model_file}"
            return None

        try:
            print(f"[app] Loading {detector_id} audio detector checkpoint...")
            if detector_id == "legacy_standalone":
                audio_predictor = StandaloneAudioPredictor(str(model_file))
            else:
                detector = get_detector(
                    detector_id,
                    checkpoint_path=str(model_file),
                    device=os.environ.get("AUDIO_DEVICE", "cpu"),
                )
                cal_path = PROJECT_ROOT / "configs" / "calibration" / f"{detector_id}_platt.json"
                service = StandaloneAudioService(detector)
                calibrator = None
                if cal_path.is_file():
                    try:
                        from deepfake_detector.audio_detection.calibration import ScoreCalibrator
                        candidate = ScoreCalibrator.load(cal_path)
                        provenance = candidate.fit_data
                        sample_count = int(provenance.get("sample_count", 0))
                        if not service.accepts_calibrator(candidate):
                            print(
                                f"[app] Ignoring calibration artifact: requires >=1000 dev/val recording scores "
                                f"from the matching studio pipeline: {cal_path}"
                            )
                        else:
                            calibrator = candidate
                            print(
                                f"[app] Loaded {provenance.get('split')} calibration "
                                f"({sample_count} samples) from {cal_path}"
                            )
                    except Exception as cal_exc:
                        print(f"[app] Warning: Could not load calibration artifact: {cal_exc}")
                audio_predictor = StandaloneAudioService(
                    detector,
                    calibrator=calibrator,
                    uncertain_band=(0.40, 0.60) if calibrator else None,
                )
            audio_predictor_id = detector_id
            audio_predictor_error = None
            print("[app] Standalone audio detector ready.")
        except Exception as exc:
            audio_predictor_error = f"Audio model load failed: {exc}"
            print(f"[app] {audio_predictor_error}")
    return audio_predictor if audio_predictor_id == detector_id else None


def list_samples() -> list[dict]:
    """Scan samples directory and extract ground-truth categories from file paths."""
    out = []
    for p in sorted(SAMPLES_DIR.glob("**/*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        is_video = ext in ALLOWED_VIDEO_EXTENSIONS
        is_audio = ext in ALLOWED_AUDIO_EXTENSIONS
        if not (is_video or is_audio):
            continue
        rel = p.relative_to(SAMPLES_DIR).as_posix()
        cat, vfake, afake, any_fake = category_from_path(str(p))
        out.append({
            "name": p.name,
            "rel": rel,
            "media_type": "video" if is_video else "audio",
            "category": cat,
            "video_truth": vfake,
            "audio_truth": afake,
            "any_truth": any_fake,
        })
    return out


@app.route("/favicon.ico")
def favicon():
    """Serve an inline SVG shield favicon to prevent 404 logs."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="8" fill="#0f172a"/>'
        '<path d="M16 4 L26 8 V16 C26 22 21 27 16 29 C11 27 6 22 6 16 V8 Z" fill="none" stroke="#00e5ff" stroke-width="2"/>'
        '<circle cx="16" cy="16" r="4" fill="#00e5ff"/>'
        '</svg>'
    )
    return app.response_class(
        svg,
        mimetype="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.route("/")
def index():
    """Main dashboard page."""
    pred = load_predictor()
    audio_pred = load_audio_predictor(DEFAULT_AUDIO_DETECTOR)
    return render_template(
        "index.html",
        samples=list_samples(),
        aggregate=AGGREGATE_METRICS,
        model_ready=pred is not None,
        model_error=predictor_error,
        model_path=MODEL_PATH,
        audio_model_ready=audio_pred is not None,
        audio_model_error=audio_predictor_error,
        audio_model_path=audio_checkpoint_path(DEFAULT_AUDIO_DETECTOR),
        audio_model_id=audio_predictor_id or DEFAULT_AUDIO_DETECTOR,
        audio_detector_options=audio_detector_options(),
        model_device=str(pred.device) if pred else "not loaded",
    )


@app.route("/api/status")
def status():
    """System and model status API endpoint."""
    pred = load_predictor()
    requested_detector = request.args.get("audio_detector", DEFAULT_AUDIO_DETECTOR)
    if requested_detector not in {"legacy_standalone", "legacy_wav2vec_mlp", "aasist", "aasist_l"}:
        requested_detector = DEFAULT_AUDIO_DETECTOR
    audio_pred = load_audio_predictor(requested_detector)
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
            "path": audio_checkpoint_path(requested_detector),
            "model_id": audio_predictor_id or requested_detector,
            "error": audio_predictor_error,
        },
    })


@app.route("/samples/<path:filename>")
def serve_sample(filename: str):
    """Serve sample media clip (video or audio)."""
    path = safe_child(SAMPLES_DIR, filename)
    if path is None or not path.exists() or path.suffix.lower() not in (ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS):
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
    mode = request.form.get("mode", "sample")
    if mode not in ("sample", "upload"):
        return jsonify({"ok": False, "error": f"Unknown mode: {mode}"}), 400

    if mode == "upload":
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"ok": False, "error": "No video file was uploaded"}), 400
        if not allowed_video(f.filename):
            return jsonify({"ok": False, "error": "Unsupported video format. Allowed: MP4, MOV, AVI, MKV"}), 400

    pred = load_predictor()
    if pred is None:
        return jsonify({"ok": False, "error": predictor_error or "Model is not available"}), 503

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
        clean_name = secure_filename(Path(f.filename).name)
        if not clean_name:
            return jsonify({"ok": False, "error": "Invalid filename"}), 400
        safe = f"{uuid.uuid4().hex[:8]}_{clean_name}"
        video_path = UPLOADS_DIR / safe
        f.save(video_path)
        ground = category_from_path(f.filename)
        video_url = f"/uploads/{safe}"
        prune_old_uploads()

    t0 = time.time()
    scores, info = pred.predict(str(video_path))
    if scores is None:
        fail_msg = info.get("fail_reason", "unknown")
        friendly = {
            "no_faces_detected": "No human faces could be detected in the video frames. Ensure the subject is clearly visible and well-lit.",
            "audio_extraction_failed": "Failed to extract or decode audio from the video clip.",
            "video_read_failed": "Unable to read video frames from the uploaded file.",
        }.get(fail_msg, str(fail_msg))
        return jsonify({
            "ok": False,
            "error": f"Preprocessing failed: {friendly}",
            "fail_reason": fail_msg,
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
    """Analyze an uploaded audio recording or sample clip with the standalone sliding-window audio detector."""
    mode = request.form.get("mode", "sample" if "sample" in request.form else "upload")

    if mode == "sample":
        sample_name = request.form.get("sample", "").strip()
        if not sample_name:
            return jsonify({"ok": False, "error": "Please select an audio sample to analyze"}), 400
        if not allowed_audio(sample_name):
            return jsonify({"ok": False, "error": "Unsupported sample audio format"}), 400
        audio_path = safe_child(SAMPLES_DIR, sample_name)
        if audio_path is None or not audio_path.exists():
            return jsonify({"ok": False, "error": f"Sample audio not found: {sample_name}"}), 404
        clean_name = sample_name
        audio_url = f"/samples/{sample_name}"
    else:
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
        audio_url = f"/uploads/{saved_name}"
        prune_old_uploads()

    detector_id = request.form.get("detector", DEFAULT_AUDIO_DETECTOR).strip().lower()
    if detector_id not in {"legacy_standalone", "legacy_wav2vec_mlp", "aasist", "aasist_l"}:
        return jsonify({"ok": False, "error": "Unknown audio detector selection"}), 400
    # Wav2Vec 2.0 CPU inference is memory intensive. Reject concurrent work
    # explicitly instead of making every browser tab appear to hang.
    if not audio_inference_lock.acquire(blocking=False):
        return jsonify({
            "ok": False,
            "error": "An audio analysis is already running. Please wait for it to finish before starting another one.",
        }), 429

    try:
        pred = load_audio_predictor(detector_id)
        if pred is None:
            return jsonify({"ok": False, "error": audio_predictor_error or "Audio model is not available"}), 503
        result = pred.predict(str(audio_path))
        result.setdefault("model_id", detector_id)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Audio analysis failed: {exc}"}), 422
    finally:
        audio_inference_lock.release()

    return jsonify({
        "ok": True,
        "filename": clean_name,
        "audio_url": audio_url,
        "result": result,
    })



def main():
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    print(f"Starting Deepfake Detection Studio at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
