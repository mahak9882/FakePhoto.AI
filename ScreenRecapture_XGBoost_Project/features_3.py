"""
Feature extraction for REAL vs SCREEN (recapture) classification.

Design goal: capture signal that genuinely distinguishes "photo of a real
scene" from "photo of a screen displaying an image", not artifacts that
correlate with file source (stock photo vs phone photo). To that end:
  - Every image is resized to a fixed size before any feature is computed.
  - Color/format-specific quirks (e.g. webp vs jpeg quantization) are
    avoided in favor of structural / frequency / texture signals that are
    physically grounded in what happens when a camera photographs a screen:
        * Moire interference between the screen's pixel grid and the
          camera's sensor grid -> periodic peaks in the frequency spectrum.
        * Screens emit light through a regular sub-pixel grid -> elevated
          high-frequency energy in specific bands even after resize.
        * Screens have a backlight -> different local contrast / dynamic
          range behavior, slight color cast, more uniform glare patterns.
        * Real-world scenes have continuous depth -> natural blur/sharpness
          gradients; recaptures are uniformly flat / often have edge frame
          cutoff artifacts (bezel edges) or reflections.
"""

import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 512  # fixed working resolution


def load_image(path):
    """Load any supported format via PIL, convert to RGB numpy array."""
    im = Image.open(path)
    im = im.convert("RGB")
    arr = np.array(im)
    return arr


def preprocess(arr):
    """Resize to fixed square size (ignores aspect ratio - we care about
    texture/frequency statistics, not composition)."""
    img = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def fft_features(gray):
    """Frequency-domain features tuned to catch moire / pixel-grid signal."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    # radial bins: low / mid / high frequency energy ratios
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = np.sqrt(cy ** 2 + cx ** 2)

    bands = []
    edges = [0, 0.1, 0.25, 0.45, 0.7, 1.01]
    for i in range(len(edges) - 1):
        lo, hi = edges[i] * max_r, edges[i + 1] * max_r
        mask = (r >= lo) & (r < hi)
        bands.append(mag[mask].mean() if mask.any() else 0.0)

    total_energy = mag.sum() + 1e-8
    band_ratios = [b / (mag.mean() + 1e-8) for b in bands]

    # Peak-iness in mid/high band: moire shows up as sharp narrow peaks,
    # not smooth energy -> compare max to mean in that band.
    mid_high_mask = (r >= 0.25 * max_r) & (r < 0.7 * max_r)
    region = mag[mid_high_mask]
    peakiness = (region.max() - region.mean()) / (region.std() + 1e-8) if region.size else 0.0

    # count of strong local maxima off-axis (grid harmonics)
    norm_mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    thresh = norm_mag > 0.85
    # exclude center DC blob
    thresh[cy - 10:cy + 10, cx - 10:cx + 10] = False
    n_peaks = int(thresh.sum())

    return band_ratios + [peakiness, n_peaks / 1000.0]


def laplacian_features(gray):
    """Sharpness / blur statistics."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()
    # local sharpness variation across tiles (real scenes: uneven depth of
    # field -> high variance across tiles; flat screen capture -> uniform)
    tiles = []
    n = 4
    th, tw = gray.shape[0] // n, gray.shape[1] // n
    for i in range(n):
        for j in range(n):
            tile = gray[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            tiles.append(cv2.Laplacian(tile, cv2.CV_64F).var())
    tile_var = np.var(tiles)
    tile_mean = np.mean(tiles)
    return [var, tile_var, tile_mean]


def color_features(img):
    """Color cast / saturation / dynamic range statistics."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    feats = []
    # per-channel mean/std (RGB) - color cast indicator
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        feats.append(ch.mean())
        feats.append(ch.std())

    feats.append(s.mean())
    feats.append(s.std())
    feats.append(v.mean())
    feats.append(v.std())

    # dynamic range (1st-99th percentile spread of luminance)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    p1, p99 = np.percentile(gray, [1, 99])
    feats.append(p99 - p1)

    # clipped highlight fraction (glare / screen brightness blowout)
    feats.append(float((gray > 250).mean()))
    feats.append(float((gray < 5).mean()))

    return feats


def texture_features(gray):
    """Local Binary Pattern histogram - texture regularity.
    Screen sub-pixel grids and moire create more regular/periodic local
    texture than natural surfaces."""
    from skimage.feature import local_binary_pattern
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    # entropy of LBP histogram - lower entropy = more regular/periodic texture
    ent = -np.sum(hist * np.log2(hist + 1e-8))
    return list(hist) + [ent]


def edge_features(gray):
    """Edge density and straightness - bezels / screen frame edges produce
    strong long straight lines near image borders."""
    edges = cv2.Canny(gray, 50, 150)
    density = edges.mean() / 255.0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=IMG_SIZE // 4, maxLineGap=10)
    n_long_lines = 0 if lines is None else len(lines)

    return [density, n_long_lines / 50.0]


def grid_regularity_features(gray):
    """Distinguish fine, regular screen pixel/sub-pixel grids (and moire)
    from coarser, less perfectly periodic real-world grids (tiles, fences,
    blinds). Uses autocorrelation of the edge map: screens produce a sharp,
    closely-spaced repeating peak structure at a fairly fine, consistent
    pitch; real-world grids tend to have wider, less uniform spacing and
    weaker periodicity once perspective is involved."""
    edges = cv2.Canny(gray, 50, 150).astype(np.float32)
    edges -= edges.mean()

    f = np.fft.fft2(edges)
    ac = np.fft.ifft2(f * np.conj(f)).real
    ac = np.fft.fftshift(ac)
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    ac_norm = ac / (ac[cy, cx] + 1e-8)

    strip = 40
    row = ac_norm[cy, cx + 3:cx + 3 + strip]
    col = ac_norm[cy + 3:cy + 3 + strip, cx]

    def first_peak_stats(arr):
        if arr.size < 5:
            return 0.0, 0.0
        peak_val = arr.max()
        peak_pos = arr.argmax() + 3
        return float(peak_val), float(peak_pos)

    rp_val, rp_pos = first_peak_stats(row)
    cp_val, cp_pos = first_peak_stats(col)

    sharpness = (rp_val + cp_val) / 2.0
    pitch = (rp_pos + cp_pos) / 2.0

    return [sharpness, pitch / strip]


def paper_texture_features(gray):
    """Halftone / paper-print signature: printed photos (inkjet/laser/
    magazine) often show a fine dot/raster pattern and a flatter local
    variance profile than a continuous-tone digital screen or a real scene
    with natural micro-texture."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))
    hh, ww = mag.shape
    cy, cx = hh // 2, ww // 2
    Y, X = np.ogrid[:hh, :ww]
    r = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    very_high = (r >= 0.85 * max_r)
    vh_energy = mag[very_high].mean() if very_high.any() else 0.0

    mean_local = cv2.blur(gray.astype(np.float32), (9, 9))
    sqmean_local = cv2.blur((gray.astype(np.float32)) ** 2, (9, 9))
    var_local = np.clip(sqmean_local - mean_local ** 2, 0, None)
    var_of_var = var_local.std()

    return [vh_energy, var_of_var / 1000.0]


def extract_features(path):
    arr = load_image(path)
    img = preprocess(arr)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    feats = []
    feats += fft_features(gray)
    feats += laplacian_features(gray)
    feats += color_features(img)
    feats += texture_features(gray)
    feats += edge_features(gray)
    feats += grid_regularity_features(gray)
    feats += paper_texture_features(gray)

    return np.array(feats, dtype=np.float32)


FEATURE_NAMES = (
    [f"fft_band_{i}" for i in range(5)] + ["fft_peakiness", "fft_npeaks"] +
    ["lap_var", "lap_tile_var", "lap_tile_mean"] +
    ["r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
     "sat_mean", "sat_std", "val_mean", "val_std",
     "dynamic_range", "highlight_frac", "shadow_frac"] +
    [f"lbp_bin_{i}" for i in range(18)] + ["lbp_entropy"] +
    ["edge_density", "n_long_lines"] +
    ["grid_sharpness", "grid_pitch"] +
    ["paper_vh_energy", "paper_var_of_var"]
)
