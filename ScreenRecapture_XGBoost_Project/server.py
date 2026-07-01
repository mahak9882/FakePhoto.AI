"""
server.py - Local prediction server for the live demo.

Run:  python server.py
Then open demo.html in your browser.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import io
from PIL import Image
import tempfile

from features import extract_features

app = Flask(__name__)
CORS(app)  # allow the HTML page to call this from file:// or localhost

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

# Load model once at startup - not per request
print("Loading model...")
bundle = joblib.load(MODEL_PATH)
PIPELINE = bundle["pipeline"]
THRESHOLD = bundle.get("threshold", 0.5)
print(f"Model loaded. Threshold = {THRESHOLD}")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image field"}), 400

    file = request.files["image"]
    img_bytes = file.read()

    # Save to temp file (features.py expects a path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(tmp_path, "JPEG")

    try:
        feats = extract_features(tmp_path).reshape(1, -1)
        proba = PIPELINE.predict_proba(feats)[0]
        classes = PIPELINE.named_steps["clf"].classes_
        screen_idx = int(np.where(classes == 1)[0][0])
        score = float(proba[screen_idx])

        # Calibrate around threshold
        if score >= THRESHOLD:
            calibrated = 0.5 + 0.5 * (score - THRESHOLD) / (1.0 - THRESHOLD + 1e-8)
        else:
            calibrated = 0.5 * score / (THRESHOLD + 1e-8)

        label = "SCREEN (recapture)" if score >= THRESHOLD else "REAL photo"
        return jsonify({
            "score": round(calibrated, 4),
            "raw_score": round(score, 4),
            "threshold": THRESHOLD,
            "label": label
        })
    finally:
        os.unlink(tmp_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    print("Open demo.html in your browser")
    app.run(host="localhost", port=5000, debug=False)
