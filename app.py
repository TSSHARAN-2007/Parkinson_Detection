"""Serves the website and runs the notebook's two trained models.

Run locally:   python app.py
Production:    gunicorn app:app  (see Dockerfile)
"""
import io
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
import tensorflow as tf

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE, "models"))
PATHS = {"spiral": "Spiral_Model.h5", "wave": "Wave_model.h5"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB per request

MODELS, LOAD_ERRORS = {}, {}
for name, fname in PATHS.items():
    full = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(full):
        LOAD_ERRORS[name] = f"{fname} not found in {MODEL_DIR}"
        print(f"[models] {LOAD_ERRORS[name]}: the {name} model is disabled")
        continue
    try:
        # compile=False: we only need inference, and it avoids optimizer-state issues
        MODELS[name] = tf.keras.models.load_model(full, compile=False)
        print(f"[models] Loaded {name} model from {fname}")
    except Exception as exc:  # version mismatch, corrupt file, etc.
        LOAD_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        print(f"[models] Could not load {fname}: {LOAD_ERRORS[name]}")


def prep(file_storage):
    """Same input the notebook used: RGB, 224x224, raw 0-255 pixels.
    (EfficientNetB0 rescales internally; the augmentation layers are inactive at inference.)"""
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    return np.asarray(img, dtype="float32")[None]


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/health")
def health():
    return jsonify(
        ready=bool(MODELS),
        models={n: (n in MODELS) for n in PATHS},
        errors=LOAD_ERRORS,
    )


@app.post("/predict")
def predict():
    out = {"spiral": None, "wave": None, "ensemble": None}
    for name in PATHS:
        f = request.files.get(name)
        if not f or name not in MODELS:
            continue
        try:
            x = prep(f)
        except (UnidentifiedImageError, OSError):
            return jsonify(error=f"The {name} image could not be read."), 400
        # Sigmoid output = probability of class 1 ("Parkinson"; class_names = [Healthy, Parkinson])
        out[name] = float(MODELS[name].predict(x, verbose=0)[0][0])
    if out["spiral"] is None and out["wave"] is None:
        return jsonify(error="No model could score this input. Check that the .h5 files are in models/ (see /health)."), 503
    if out["spiral"] is not None and out["wave"] is not None:
        out["ensemble"] = (out["spiral"] + out["wave"]) / 2
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
