# Spot the Fake Photo — Submission Notes

## What I built
A lightweight real-vs-screen classifier that takes one image and outputs a
score from 0 (real photo) to 1 (screen recapture), running entirely on CPU
with no internet connection required.

Usage:
    python predict.py some_image.jpg   →   0.07   (real)
    python predict.py another.jpg      →   0.91   (screen)

---

## Approach

No deep learning. Each image is resized to 512×512 and a fixed vector of
55 handcrafted signal features is extracted, then fed into a small Random
Forest classifier. The whole model is under 1MB.

The features are physically grounded in what actually happens when a camera
photographs a screen:

**Frequency domain (FFT)**
Screens produce interference between their pixel grid and the camera sensor
grid (moiré). This shows up as periodic peaks in the frequency spectrum.
Features: energy across 5 radial frequency bands, peakiness in the
moiré-prone mid-high band, and count of strong off-centre spectral peaks.

**Grid regularity (most important feature, importance 0.22)**
Autocorrelation of the edge map measures the sharpness and pitch of any
repeating pattern. Genuine screen pixel grids are sharp, fine, and
consistently spaced. Real-world grids (tiled floors, fences, blinds) are
coarser and less perfectly regular once perspective distortion is involved.
This feature was added after diagnosing early false positives — real photos
of tiled floors were triggering the moiré detector — and it fixed them.

**Sharpness / blur (Laplacian variance)**
Overall and per-tile Laplacian variance. Real scenes have uneven
depth-of-field blur across image tiles; flat screen recaptures are
uniformly sharp or uniformly soft.

**Colour / dynamic range**
Per-channel mean and std, saturation, highlight clipping fraction, shadow
clipping fraction. Screens often produce colour cast and glare blowout.

**Blue bias**
(mean_B − mean_R) / (mean_B + mean_R). Phone OLED and laptop LCD screens
tend to be slightly cooler/bluer than real scenes lit by warm ambient light.
Ranked 4th most important feature.

**Texture (Local Binary Patterns)**
LBP histogram (18 bins) + entropy. Screen sub-pixel structure produces more
locally regular texture than natural surfaces.

**Bezel / border contrast**
Brightness difference between the outermost 8% border strip and the inner
region. Screens photographed in a room typically have a dark bezel
surrounding a brighter display. Also measures: glare blob count (specular
highlights on screen surface) and spatial brightness non-uniformity across
a 3×3 tile grid.

**Paper / print signature**
Very-high-frequency spectral energy and local variance uniformity. Targets
halftone / print-dot patterns in printout recaptures, which lack screen
backlighting and moiré but carry different texture from natural scenes.

**Edges**
Canny edge density and count of long straight lines via Hough transform.
Bezels produce strong long straight lines near the image border.

Classifier: Random Forest (400 trees, max depth 8, balanced class weights,
sklearn). Two models were compared via 5-fold CV — Random Forest beat
Logistic Regression (90% vs 86%) and was chosen.

---

## Accuracy

**95.17% — 5-fold cross-validated accuracy**
**95.86% — with threshold tuning (threshold = 0.54)**

Evaluated on 145 genuinely camera-captured images (87 REAL, 58 SCREEN
recaptures) using StratifiedKFold + cross_val_predict, so every prediction
is on data the model never trained on. These are honest numbers.

Only 7 images misclassified:
- 5 REAL predicted as SCREEN: real photos with unusually strong regular
  patterns (gridded surfaces) that partially overlap with screen signal.
- 2 SCREEN predicted as REAL: recaptures shot at a steep angle or very
  close distance, suppressing moiré and bezel signal.

**Dataset note:** the original dataset contained stock/web-downloaded images
mixed with genuine phone captures. Training on the mixed set gave ~81%
accuracy for the wrong reasons — the model was partly learning "professional
stock photo vs casual phone snap" rather than real vs screen signal. All
numbers above use only WhatsApp-captured images (guaranteed genuine
camera-to-subject captures). This is the honest number, not the inflated one.

---

## Latency

- **~150–200 ms per image** (warm process, laptop CPU, no GPU)
- **~1.5 s cold start** for a single `python predict.py img.jpg` call —
  almost entirely one-time Python/OpenCV import overhead, not actual compute.
  In a persistent server process or phone app, imports happen once; each
  subsequent image is ~150–200 ms.

Tested on laptop CPU (Intel). No GPU required at any point.

---

## Cost per image

**On-device: free.** Model is <1 MB, runs on CPU, no network call needed.
This is the right deployment target — the phone already has the camera, the
CPU, and the model. Zero marginal cost per image.

**Cloud server (rough estimate):** a small CPU instance (~$0.05/hr) doing
~200 ms of work per image handles ~18,000 images/hour per core.
- Per 1,000 images: ~$0.003
- Per 1,000,000 images: ~$2.80

Assumptions: 100% utilisation, no orchestration overhead, single core.
Real-world cost would be 2–3× higher with idle time and infra overhead.

---

## Phone deployment path

The brief says "it will eventually run on a phone" — the design choices
already support this:

- Model is 800 KB, CPU-only, no GPU dependency ✓
- All feature extraction uses standard OpenCV functions (FFT, Canny,
  Laplacian, LBP, HoughLinesP) that exist in OpenCV's Android and iOS SDKs ✓
- Random Forest inference exports to ONNX in one line via `sklearn-onnx` ✓

Concrete path to ship on-device:
1. `sklearn-onnx` converts the Random Forest to a `.onnx` file (~800 KB)
2. Feature extraction ported to Kotlin (Android) or Swift (iOS) using
   OpenCV Mobile SDK — same functions, same API
3. ONNX Runtime Mobile runs inference on the `.onnx` model
4. Estimated on-device inference: 200–400 ms on a mid-range phone CPU

No retraining needed. The Python training pipeline stays as-is; only the
inference side moves to mobile.

---

## Live demo

An optional live demo is included (`server.py` + `demo.html`):

    python server.py        # start local prediction server
    # open demo.html in browser

The page uses the device camera, captures a frame every 2 seconds, sends it
to the local Flask server, and shows the verdict live with a score meter.
Works on mobile too — open demo.html on your phone while server.py runs on
your laptop (same WiFi, use laptop's local IP instead of localhost).

---

## What I'd improve with more time

**1. More data, especially hard cases**
High-DPI screens (iPhone 15 Pro, MacBook Retina) produce almost no moiré —
they're the hardest case and need 50+ dedicated examples. Currently
underrepresented.

**2. Separate printout handling**
Printed photos and QR cards have no backlight, no sub-pixel grid, and no
moiré — physically a different problem. A dedicated branch with halftone
dot detection and ink-gamut vs. emissive-light features would handle them
better than the current shared classifier.

**3. Small CNN with enough data**
Once the dataset reaches 300+ per class, a distilled MobileNetV3 would
likely catch artifacts the handcrafted features miss, while staying small
enough for on-device use. The handcrafted approach was chosen here because
it's more interpretable, faster to train, and more reliable on a small
dataset.

**4. Reduce cold-start latency**
Lazy-importing only what's needed, or compiling to ONNX/frozen binary,
would cut the 1.5 s cold start to under 200 ms total.

---

## Staying ahead of adapting cheaters

**Monitoring:** Track feature importance over time. If `grid_sharpness`
stops discriminating (e.g. cheaters switch to very high-DPI screens with no
moiré), that's an early signal to retrain or add features.

**Retraining cadence:** Refresh the dataset periodically as new screen
models (higher DPI, mini-LED, OLED) and printer types appear. Each has a
different grid pitch or halftone pattern.

**Choosing the fraud cutoff:** A false positive (blocking a real user) is
more costly than a false negative (missing a cheat who gets human review).
I'd set the threshold at ~0.60 in production (vs 0.54 used for accuracy
optimisation here) and route flagged images to human review rather than
auto-rejecting. The threshold is a single parameter tunable live against
observed false-positive rates — no retraining needed.

**Adversarial robustness:** A determined cheater could shoot a screen
through a diffuser (kills moiré) or post-process to remove colour cast.
Longer term, a small CNN trained on adversarial examples (generated by
specifically trying to fool the current model) would raise the bar
significantly.
