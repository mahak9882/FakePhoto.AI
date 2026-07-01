#!/usr/bin/env python3
"""
predict.py - Spot the Fake Photo

Usage:
    python predict.py some_image.jpg
    -> prints a single float in [0, 1]: 0 = REAL photo, 1 = SCREEN recapture

Loads the pre-trained model (model.joblib) and the same feature pipeline
used for training (features.py). No training happens here - this is the
fast, one-line predictor the assignment asks for.
"""

import sys
import os
import joblib
import numpy as np

from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")


def predict(image_path: str) -> float:
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    threshold = bundle.get("threshold", 0.5)

    feats = extract_features(image_path).reshape(1, -1)
    proba = pipeline.predict_proba(feats)[0]

    classes = pipeline.named_steps["clf"].classes_
    screen_idx = int(np.where(classes == 1)[0][0])
    raw_score = float(proba[screen_idx])

    # Return the raw probability (for the caller to use), but the
    # threshold-adjusted decision can be inferred: score >= threshold -> SCREEN
    return raw_score


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.isfile(img_path):
        print(f"Error: file not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    score = predict(img_path)
    print(f"{score:.4f}")
