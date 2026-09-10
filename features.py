"""
features.py
------------
Turns a raw RGB satellite-like image into a numeric feature vector the
ML model can learn from. This mirrors what the CNN layers of a deep model
would learn to detect on their own, but here we compute them explicitly:

    1. Vegetation ratio     - fraction of green-dominant pixels
    2. Built-up ratio       - fraction of gray/man-made colored pixels
    3. Edge density         - how much "structure" (roads/building edges)
                               is in the image, via a Sobel-style gradient
    4. Color variance        - texture/heterogeneity of the scene
    5. Brightness stats      - mean & std of pixel intensity

These 5 categories expand into a fixed-length vector (see FEATURE_NAMES).
"""

import numpy as np
from PIL import Image

FEATURE_NAMES = [
    "vegetation_ratio",
    "builtup_ratio",
    "edge_density",
    "color_variance",
    "brightness_mean",
    "brightness_std",
]


def _to_array(img_or_path):
    if isinstance(img_or_path, str):
        img = Image.open(img_or_path).convert("RGB")
    else:
        img = img_or_path.convert("RGB")
    return np.array(img).astype(np.float32)


def _sobel_edges(gray):
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag


def extract_features(img_or_path):
    arr = _to_array(img_or_path)  # H,W,3
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Vegetation: green channel clearly dominant
    veg_mask = (g > r + 8) & (g > b + 8)
    vegetation_ratio = veg_mask.mean()

    # Built-up (roofs/roads): grayish, r≈g≈b, and not too dark (roads) or
    # mid-bright (rooftops)
    max_c = np.max(arr, axis=2)
    min_c = np.min(arr, axis=2)
    saturation = (max_c - min_c) / (max_c + 1e-6)
    builtup_mask = (saturation < 0.15) & (max_c > 60)
    builtup_ratio = builtup_mask.mean()

    gray = arr.mean(axis=2)
    edges = _sobel_edges(gray)
    edge_density = (edges > 25).mean()

    color_variance = arr.std(axis=(0, 1)).mean()
    brightness_mean = gray.mean()
    brightness_std = gray.std()

    return np.array([
        vegetation_ratio,
        builtup_ratio,
        edge_density,
        color_variance,
        brightness_mean,
        brightness_std,
    ], dtype=np.float32)


def extract_features_batch(paths):
    return np.stack([extract_features(p) for p in paths])
