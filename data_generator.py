"""
data_generator.py
------------------
Generates a synthetic dataset of "satellite-like" images for the poverty
prediction project.

WHY SYNTHETIC DATA?
Real satellite imagery (Sentinel-2 / Maxar) + real poverty ground-truth
labels (DHS / World Bank LSMS surveys) require registration, API keys and
multi-GB downloads that aren't available in this environment. To give you
a complete, runnable, end-to-end project today, this script *procedurally
draws* images that mimic the key visual signals researchers actually use
for this task:

    - Building density / size       -> more & bigger rectangles
    - Road network density          -> more & straighter lines
    - Vegetation cover               -> green patches
    - Settlement pattern regularity -> grid-like vs. sparse/random layout

Each image is generated FROM a target poverty class (LOW / MEDIUM / HIGH),
with the visual features controlled to correlate with that class, exactly
like how real developed areas show dense infrastructure and real
under-developed areas show sparse infrastructure + more open land.

>>> HOW TO SWAP IN REAL DATA LATER <<<
Replace this script with a script that:
  1. Downloads Sentinel-2 / Maxar tiles for GPS coordinates in a DHS survey
  2. Downloads the DHS "wealth index" for each coordinate
  3. Buckets the wealth index into LOW / MEDIUM / HIGH (or keep it continuous
     and switch to a regression model)
  4. Saves tiles into data/train/<class>/ and data/test/<class>/ exactly like
     this script does -- the rest of the pipeline (features.py, train.py,
     app.py) does not need to change.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 128
CLASSES = ["low", "medium", "high"]

# Feature ranges per class: (num_buildings, num_roads, vegetation_ratio, jitter)
# jitter = how randomly scattered buildings are (0 = neat grid, 1 = chaotic)
CLASS_PROFILES = {
    "low": dict(buildings=(2, 10), roads=(0, 2), veg=(0.55, 0.85), jitter=(0.7, 1.0)),
    "medium": dict(buildings=(15, 35), roads=(2, 5), veg=(0.30, 0.55), jitter=(0.3, 0.6)),
    "high": dict(buildings=(40, 80), roads=(5, 10), veg=(0.05, 0.25), jitter=(0.0, 0.25)),
}


def _base_terrain(veg_ratio):
    """Create a base ground texture: mix of soil/brown and vegetation/green."""
    soil = np.array([150, 120, 90], dtype=np.float32)
    veg = np.array([60, 130, 60], dtype=np.float32)
    h, w = IMG_SIZE, IMG_SIZE

    # Perlin-ish noise via random blobs for natural-looking patches
    mask = np.zeros((h, w), dtype=np.float32)
    n_blobs = random.randint(15, 30)
    for _ in range(n_blobs):
        cx, cy = random.randint(0, w), random.randint(0, h)
        r = random.randint(8, 30)
        yy, xx = np.ogrid[:h, :w]
        dist = (xx - cx) ** 2 + (yy - cy) ** 2
        mask += np.exp(-dist / (2 * r * r))
    mask = mask / mask.max()
    mask = (mask > (1 - veg_ratio)).astype(np.float32)  # threshold to hit target ratio approx

    img = soil[None, None, :] * (1 - mask[:, :, None]) + veg[None, None, :] * mask[:, :, None]
    noise = np.random.normal(0, 6, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def _draw_roads(draw, n_roads):
    for _ in range(n_roads):
        if random.random() < 0.5:
            y = random.randint(0, IMG_SIZE)
            draw.line([(0, y), (IMG_SIZE, y + random.randint(-10, 10))],
                      fill=(90, 90, 90), width=random.randint(2, 4))
        else:
            x = random.randint(0, IMG_SIZE)
            draw.line([(x, 0), (x + random.randint(-10, 10), IMG_SIZE)],
                      fill=(90, 90, 90), width=random.randint(2, 4))


def _draw_buildings(draw, n_buildings, jitter):
    """Higher jitter = scattered irregular buildings (informal settlement look).
    Lower jitter = neat grid-aligned buildings (planned urban look)."""
    grid_cols = max(1, int(np.sqrt(n_buildings)))
    cell = IMG_SIZE // max(grid_cols, 1)
    placed = 0
    i = 0
    while placed < n_buildings:
        gx = (i % grid_cols) * cell
        gy = (i // grid_cols) * cell
        i += 1
        if gy >= IMG_SIZE:
            gx = random.randint(0, IMG_SIZE - 10)
            gy = random.randint(0, IMG_SIZE - 10)

        jx = int((random.random() - 0.5) * jitter * cell * 1.5)
        jy = int((random.random() - 0.5) * jitter * cell * 1.5)
        x0 = np.clip(gx + jx, 0, IMG_SIZE - 6)
        y0 = np.clip(gy + jy, 0, IMG_SIZE - 6)
        bw = random.randint(4, 10) if jitter > 0.5 else random.randint(6, 14)
        bh = random.randint(4, 10) if jitter > 0.5 else random.randint(6, 14)
        x1 = min(IMG_SIZE, x0 + bw)
        y1 = min(IMG_SIZE, y0 + bh)

        roof_gray = random.randint(150, 220)
        color = (roof_gray, roof_gray - random.randint(0, 20), roof_gray - random.randint(10, 40))
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=(60, 60, 60))
        placed += 1


def generate_image(cls):
    profile = CLASS_PROFILES[cls]
    veg_ratio = random.uniform(*profile["veg"])
    img = _base_terrain(veg_ratio)
    draw = ImageDraw.Draw(img)

    n_roads = random.randint(*profile["roads"])
    _draw_roads(draw, n_roads)

    n_buildings = random.randint(*profile["buildings"])
    jitter = random.uniform(*profile["jitter"])
    _draw_buildings(draw, n_buildings, jitter)

    return img


def build_dataset(root="data", n_per_class_train=200, n_per_class_test=40, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    for split, n in [("train", n_per_class_train), ("test", n_per_class_test)]:
        for cls in CLASSES:
            out_dir = os.path.join(root, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(n):
                img = generate_image(cls)
                img.save(os.path.join(out_dir, f"{cls}_{i:04d}.png"))
        print(f"[{split}] generated {n} images per class -> {root}/{split}/<class>/")


if __name__ == "__main__":
    build_dataset()
    print("Done. Dataset created under ./data/train and ./data/test")
