"""
Train the real-vs-screen classifier.

Usage:
    python train.py --real_dir DATA/REAL --screen_dir DATA/SCREEN
"""

import argparse
import os
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from features import extract_features, FEATURE_NAMES

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp"}


def collect_paths(folder):
    paths = []
    for f in sorted(os.listdir(folder)):
        ext = os.path.splitext(f)[1].lower()
        if ext in VALID_EXT:
            paths.append(os.path.join(folder, f))
    return paths


def build_dataset(real_dir, screen_dir):
    real_paths = collect_paths(real_dir)
    screen_paths = collect_paths(screen_dir)
    print(f"REAL images:   {len(real_paths)}")
    print(f"SCREEN images: {len(screen_paths)}")

    X, y, used_paths = [], [], []
    for p in real_paths:
        try:
            X.append(extract_features(p))
            y.append(0)
            used_paths.append(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}")
    for p in screen_paths:
        try:
            X.append(extract_features(p))
            y.append(1)
            used_paths.append(p)
        except Exception as e:
            print(f"  [skip] {p}: {e}")

    return np.array(X), np.array(y), used_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", default="../data/DATA/REAL")
    ap.add_argument("--screen_dir", default="../data/DATA/SCREEN")
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    print("Extracting features...")
    t0 = time.time()
    X, y, paths = build_dataset(args.real_dir, args.screen_dir)
    print(f"Feature extraction done in {time.time()-t0:.1f}s. "
          f"X shape={X.shape}")

    # Two candidate models, compared via 5-fold stratified CV
    candidates = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=2,
                max_features="sqrt", class_weight="balanced", random_state=42)),
        ]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_acc, best_preds = None, -1, None
    for name, pipe in candidates.items():
        preds = cross_val_predict(pipe, X, y, cv=skf, method="predict")
        acc = accuracy_score(y, preds)
        print(f"\n[{name}] 5-fold CV accuracy: {acc:.4f}")
        print(classification_report(y, preds, target_names=["REAL", "SCREEN"]))
        print("Confusion matrix [[TN FP] [FN TP]]:")
        print(confusion_matrix(y, preds))
        if acc > best_acc:
            best_name, best_acc, best_preds = name, acc, preds

    print(f"\n>> Best model: {best_name} (CV accuracy {best_acc:.4f})")

    # Threshold tuning: find the cutoff that maximises CV accuracy
    # (default 0.5 is rarely optimal on imbalanced classes)
    best_pipe_for_proba = candidates[best_name]
    from sklearn.model_selection import cross_val_predict as cvp
    probas = cvp(best_pipe_for_proba, X, y, cv=skf, method="predict_proba")[:, 1]
    best_thresh, best_thresh_acc = 0.5, 0.0
    for thresh in np.arange(0.30, 0.71, 0.02):
        preds_t = (probas >= thresh).astype(int)
        acc_t = accuracy_score(y, preds_t)
        if acc_t > best_thresh_acc:
            best_thresh_acc, best_thresh = acc_t, thresh
    print(f"   Threshold tuning: best threshold={best_thresh:.2f}, "
          f"accuracy={best_thresh_acc:.4f} (vs {best_acc:.4f} at 0.5)")
    best_acc = best_thresh_acc

    # Misclassified files - useful for the note / sanity-check of data quality
    misclassified = [(p, yt, yp) for p, yt, yp in zip(paths, y, best_preds) if yt != yp]
    if misclassified:
        print(f"\nMisclassified ({len(misclassified)}):")
        for p, yt, yp in misclassified:
            print(f"  true={'SCREEN' if yt else 'REAL'} pred={'SCREEN' if yp else 'REAL'}  {p}")

    # Refit best model on ALL data for the final shipped model
    final_pipe = candidates[best_name]
    final_pipe.fit(X, y)
    joblib.dump({"pipeline": final_pipe, "feature_names": FEATURE_NAMES,
                 "cv_accuracy": best_acc, "model_name": best_name,
                 "threshold": best_thresh}, args.out)
    print(f"\nSaved final model to {args.out}")

    if best_name == "rf":
        importances = final_pipe.named_steps["clf"].feature_importances_
        order = np.argsort(importances)[::-1][:10]
        print("\nTop 10 most important features:")
        for i in order:
            print(f"  {FEATURE_NAMES[i]}: {importances[i]:.4f}")


if __name__ == "__main__":
    main()
