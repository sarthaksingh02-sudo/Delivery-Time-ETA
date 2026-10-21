from __future__ import annotations

import json
import io
import os
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_file, send_from_directory

from eta_model import FEATURES, demo_model, predict, train
from model_package import build_package

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "eta_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
PACKAGE_PATH = Path(os.environ.get("ETA_MODEL_PACKAGE", str(ROOT / "deliverables" / "eta_model_package.zip")))
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 120 * 1024 * 1024

if MODEL_PATH.exists():
    model_bundle = joblib.load(MODEL_PATH)
    if model_bundle.get("mode") == "real_dataset" and not PACKAGE_PATH.exists():
        build_package(MODEL_PATH, METRICS_PATH, PACKAGE_PATH)
elif PACKAGE_PATH.exists():
    with zipfile.ZipFile(PACKAGE_PATH) as package:
        model_bundle = joblib.load(io.BytesIO(package.read("eta_model.joblib")))
else:
    model_bundle = demo_model()

def current_metrics():
    if not METRICS_PATH.exists():
        if PACKAGE_PATH.exists():
            try:
                with zipfile.ZipFile(PACKAGE_PATH) as package:
                    return json.loads(package.read("metrics.json"))
            except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
                pass
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.get("/api/status")
def status():
    return jsonify({
        "mode": model_bundle.get("mode", "unknown"),
        "winner": model_bundle.get("winner", "ETA model"),
        "training_rows": model_bundle.get("training_rows", 0),
        "metrics": current_metrics() if model_bundle.get("mode") == "real_dataset" else None,
        "package_ready": model_bundle.get("mode") == "real_dataset" and PACKAGE_PATH.is_file(),
        "features": model_bundle.get("features", FEATURES),
    })

@app.post("/api/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    try:
        required = set(model_bundle.get("features", FEATURES))
        missing = sorted(required - body.keys())
        if missing:
            return jsonify({"error": "Missing inputs: " + ", ".join(missing)}), 400
        row = pd.DataFrame([{**body, "City": body.get("City", body["Area"])}])
        minutes = float(predict(model_bundle, row)[0])
        return jsonify({"minutes": round(minutes, 1), "mode": model_bundle.get("mode"), "model": model_bundle.get("winner")})
    except (TypeError, ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/api/train")
def api_train():
    global model_bundle
    uploaded = request.files.get("dataset")
    if not uploaded or not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"error": "Choose a CSV file from the Kaggle dataset."}), 400
    try:
        frame = pd.read_csv(uploaded)
        report = train(frame, ROOT / "models")
        model_bundle = joblib.load(MODEL_PATH)
        build_package(MODEL_PATH, METRICS_PATH, PACKAGE_PATH)
        return jsonify({"message": "Training, held-out evaluation and model packaging completed.", "report": report, "package_ready": True, "features": model_bundle.get("features", FEATURES)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

@app.get("/api/model-package")
def download_model_package():
    if model_bundle.get("mode") != "real_dataset" or not PACKAGE_PATH.is_file():
        return jsonify({"error": "Train the model on the real Kaggle CSV before downloading a model package."}), 409
    return send_file(PACKAGE_PATH, as_attachment=True, download_name="eta_model_package.zip", mimetype="application/zip")

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "CSV is larger than the 120 MB upload limit."}), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
