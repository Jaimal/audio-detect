#!/usr/bin/env python3
"""Simple Flask web UI for the deepfake audio detector.

Loads the bundled Keras model once at startup, then accepts audio
uploads via a small browser UI and returns Human/AI probabilities.
"""

import io
import os
import shutil
import tempfile
import logging
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, jsonify, render_template_string, request, send_from_directory

ROOT = Path(__file__).parent.resolve()
MODEL_PATH = ROOT / "model" / "model-1.keras"
AUDIO_DIR = ROOT / "audio"
N_MELS = 91
MAX_TIME_STEPS = 150
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTS = {".flac", ".m4a", ".wav", ".mp3", ".aac", ".ogg"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("audio-detect")


def _load_model():
    """Load the bundled HDF5 model. The file uses the legacy .h5 format
    despite its .keras extension, so we copy to a temp .h5 path so
    Keras picks the right loader."""
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


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

INDEX_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>Deepfake Audio Detector</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
    h1 { margin-bottom: 0.25rem; }
    .sub { color: #666; margin-top: 0; }
    .card { border: 1px solid #ccc4; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; }
    button { font-size: 1rem; padding: 0.5rem 1rem; border-radius: 6px; border: 1px solid #888; background: #f6f6f6; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: progress; }
    input[type=file] { font-size: 1rem; }
    .result { font-size: 1.1rem; }
    .verdict { font-weight: 700; font-size: 1.5rem; }
    .ai { color: #c0392b; }
    .human { color: #27ae60; }
    .bar { height: 12px; background: #eee; border-radius: 6px; overflow: hidden; margin: 0.4rem 0 0.6rem; }
    .bar > span { display: block; height: 100%; }
    .meta { color: #666; font-size: 0.9rem; }
    ul.samples { padding-left: 1.2rem; }
    code { background: #0001; padding: 1px 4px; border-radius: 3px; }
    .err { color: #c0392b; }
  </style>
</head>
<body>
  <h1>Deepfake Audio Detector</h1>
  <p class=\"sub\">Upload a clip of someone speaking. The model classifies it as Human or AI-generated.</p>

  <div class=\"card\">
    <form id=\"f\">
      <label>Audio file (flac, m4a, wav, mp3, aac, ogg — up to 50MB):<br>
        <input type=\"file\" name=\"file\" id=\"file\" accept=\"audio/*,.flac,.m4a,.wav,.mp3,.aac,.ogg\" required>
      </label>
      <div style=\"margin-top: 0.75rem;\">
        <button type=\"submit\" id=\"go\">Analyze</button>
      </div>
    </form>
    <div id=\"out\" class=\"result\" style=\"margin-top: 1rem;\"></div>
  </div>

  <div class=\"card\">
    <strong>Try the bundled samples:</strong>
    <ul class=\"samples\">
      {% for s in samples %}
        <li>
          <a href=\"/sample/{{ s }}\" download>{{ s }}</a>
          &middot; <a href=\"#\" data-sample=\"{{ s }}\" class=\"try\">try this sample</a>
        </li>
      {% endfor %}
    </ul>
    <p class=\"meta\">Tip: the two ElevenLabs clips should classify as AI; <code>real_human.m4a</code> as Human.</p>
  </div>

  <script>
    const out = document.getElementById('out');
    const form = document.getElementById('f');
    const go = document.getElementById('go');

    function render(d) {
      const cls = d.verdict === 'Human' ? 'human' : 'ai';
      out.innerHTML = `
        <div class=\"verdict ${cls}\">${d.verdict}</div>
        <div>AI: ${d.ai_percent}%</div>
        <div class=\"bar\"><span style=\"width:${d.ai_percent}%; background:#e74c3c\"></span></div>
        <div>Human: ${d.human_percent}%</div>
        <div class=\"bar\"><span style=\"width:${d.human_percent}%; background:#2ecc71\"></span></div>
        <div class=\"meta\">duration: ${d.duration_seconds}s &middot; sample rate: ${d.sample_rate} Hz</div>
      `;
    }

    async function analyze(formData) {
      go.disabled = true;
      out.innerHTML = 'Analyzing…';
      try {
        const res = await fetch('/predict', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
        render(data);
      } catch (e) {
        out.innerHTML = '<span class=\"err\">Error: ' + e.message + '</span>';
      } finally {
        go.disabled = false;
      }
    }

    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const file = document.getElementById('file').files[0];
      if (!file) return;
      const fd = new FormData();
      fd.append('file', file);
      analyze(fd);
    });

    document.querySelectorAll('a.try').forEach(a => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const fd = new FormData();
        fd.append('sample', a.dataset.sample);
        analyze(fd);
      });
    });
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    samples = sorted(p.name for p in AUDIO_DIR.iterdir() if p.suffix.lower() in ALLOWED_EXTS)
    return render_template_string(INDEX_HTML, samples=samples)


@app.get("/sample/<path:name>")
def sample(name):
    return send_from_directory(AUDIO_DIR, name, as_attachment=True)


@app.get("/health")
def health():
    return jsonify(status="ok", model=str(MODEL_PATH.name))


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port, debug=False)
