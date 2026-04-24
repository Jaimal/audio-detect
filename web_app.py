#!/usr/bin/env python3
"""Flask backend for the deepfake audio detector SPA.

Loads the bundled Keras model once at startup and exposes:
  GET  /               -> SPA (static/index.html)
  GET  /health         -> liveness
  GET  /samples        -> list of bundled sample filenames
  GET  /sample/<name>  -> download a bundled sample
  POST /predict        -> classify an uploaded file or named bundled sample
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).parent.resolve()
MODEL_PATH = ROOT / "model" / "model-1.keras"
AUDIO_DIR = ROOT / "audio"
STATIC_DIR = ROOT / "static"
N_MELS = 91
MAX_TIME_STEPS = 150
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTS = {".flac", ".m4a", ".wav", ".mp3", ".aac", ".ogg"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("audio-detect")


def _load_model():
    """The bundled file uses the legacy HDF5 format despite its .keras
    extension, so we copy it to a .h5 path so Keras picks the right loader."""
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False).name
    shutil.copy(MODEL_PATH, tmp)
    log.info("Loading model from %s", MODEL_PATH)
    return tf.keras.models.load_model(tmp)


MODEL = _load_model()


def _predict(audio_path: str):
    audio, sr = librosa.load(audio_path)
    if len(audio) == 0:
        raise ValueError("Audio file is empty or could not be decoded")
    duration = float(len(audio) / sr) if sr else 0.0

    mel = librosa.feature.melspectrogram(y=audio, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max)
    if mel.shape[1] < MAX_TIME_STEPS:
        mel = np.pad(mel, ((0, 0), (0, MAX_TIME_STEPS - mel.shape[1])), mode="constant")
    else:
        mel = mel[:, :MAX_TIME_STEPS]

    preds = MODEL.predict(np.array([mel]), verbose=0)[0]
    ai_pct = float(preds[0]) * 100.0
    human_pct = float(preds[1]) * 100.0
    verdict = "Human" if human_pct >= ai_pct else "AI"
    return {
        "verdict": verdict,
        "ai_percent": round(ai_pct, 2),
        "human_percent": round(human_pct, 2),
        "duration_seconds": round(duration, 2),
        "sample_rate": int(sr),
    }


app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", model=MODEL_PATH.name)


@app.get("/samples")
def samples():
    items = []
    for p in sorted(AUDIO_DIR.iterdir()):
        if p.suffix.lower() in ALLOWED_EXTS:
            items.append({"name": p.name, "size": p.stat().st_size})
    return jsonify(items)


@app.get("/sample/<path:name>")
def sample(name):
    return send_from_directory(AUDIO_DIR, name, as_attachment=False)


@app.post("/predict")
def predict_route():
    if "sample" in request.form:
        name = request.form["sample"]
        candidate = (AUDIO_DIR / name).resolve()
        if AUDIO_DIR.resolve() not in candidate.parents or not candidate.exists():
            return jsonify(error="Unknown sample"), 400
        try:
            return jsonify(_predict(str(candidate)))
        except Exception as e:
            log.exception("sample predict failed")
            return jsonify(error=str(e)), 500

    if "file" not in request.files:
        return jsonify(error="No file uploaded"), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify(error="Empty filename"), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify(error=f"Unsupported format: {ext}. Allowed: {sorted(ALLOWED_EXTS)}"), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        f.save(tmp.name)
        path = tmp.name
    try:
        return jsonify(_predict(path))
    except Exception as e:
        log.exception("upload predict failed")
        return jsonify(error=str(e)), 500
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error=f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit"), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False, threaded=True)
