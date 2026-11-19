"""Datasets: a synthetic MRI-like generator (for the demo) and a folder loader (real data)."""
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from .features import load_gray

SYN_CLASSES = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
_PRETTY = {"notumor": "No Tumor", "no_tumor": "No Tumor", "no": "No Tumor", "glioma": "Glioma",
           "meningioma": "Meningioma", "pituitary": "Pituitary", "yes": "Tumor", "tumor": "Tumor"}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _smooth_noise(rng, size, cells):
    lo = rng.normal(size=(cells, cells))
    im = Image.fromarray(lo.astype(np.float32), mode="F").resize((size, size), Image.BICUBIC)
    return np.asarray(im)


def make_mri(kind: int, rng: np.random.Generator, size: int = 128) -> np.ndarray:
    """Synthetic axial brain slice. kind: 0 none, 1 glioma, 2 meningioma, 3 pituitary. -> uint8."""
    yy, xx = np.mgrid[0:size, 0:size]
    x, y = (xx - size / 2) / (size / 2), (yy - size / 2) / (size / 2)
    a = rng.uniform(-0.12, 0.12)                              # small head rotation
    x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    rx, ry = rng.uniform(0.78, 0.88), rng.uniform(0.88, 0.97)
    r_head = np.sqrt((x / rx) ** 2 + (y / ry) ** 2)
    r_brain = np.sqrt((x / (rx * 0.86)) ** 2 + (y / (ry * 0.88)) ** 2)
    img = np.zeros((size, size))
    img[r_head < 1] = 0.50                                    # scalp / skull ring
    img[r_brain < 1.0] = 0.36
    brain = r_brain < 1.0
    img += brain * (0.05 * _smooth_noise(rng, size, 10) + 0.025 * _smooth_noise(rng, size, 28))
    img -= brain * 0.09 * np.exp(-(x / 0.03) ** 2)            # inter-hemispheric fissure
    for s in (-1, 1):                                         # lateral ventricles
        v = ((x - s * 0.15) / 0.09) ** 2 + ((y + 0.05) / 0.22) ** 2
        img -= 0.22 * np.exp(-v * 2.0) * brain
    if rng.random() < 0.35:                                   # distractor: small benign bright spot
        cx, cy = rng.uniform(-0.5, 0.5, 2)
        img += brain * 0.10 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.003)

    def blob(cx, cy, R0, irr, soft):
        th = np.arctan2(y - cy, x - cx)
        R = R0 * (1 + sum(irr * rng.uniform(0.3, 1) * np.cos(k * th + rng.uniform(0, 6.28))
                          for k in (2, 3, 5)) / 2)
        r = np.hypot(x - cx, y - cy)
        return 1 / (1 + np.exp(-(R - r) / soft))

    if kind == 1:                                             # glioma: large, irregular, fuzzy
        ang, rad = rng.uniform(0, 6.28), rng.uniform(0.05, 0.45)
        cx, cy = rad * np.cos(ang), rad * np.sin(ang) - 0.05
        m = blob(cx, cy, rng.uniform(0.17, 0.27), 0.45, 0.06)
        halo = blob(cx, cy, rng.uniform(0.30, 0.40), 0.3, 0.10)
        img += brain * (halo * 0.07 + m * (0.30 + 0.10 * _smooth_noise(rng, size, 12)))
    elif kind == 2:                                           # meningioma: round, sharp, peripheral
        ang, rad = rng.uniform(0, 6.28), rng.uniform(0.60, 0.72)
        cx, cy = rad * rx * np.cos(ang), rad * ry * np.sin(ang)
        m = blob(cx, cy, rng.uniform(0.11, 0.17), 0.05, 0.015)
        img += (r_brain < 1.04) * m * rng.uniform(0.38, 0.5)
    elif kind == 3:                                           # pituitary: small, midline, low
        cx, cy = rng.normal(0, 0.025), 0.26 + rng.normal(0, 0.03)
        m = blob(cx, cy, rng.uniform(0.06, 0.10), 0.06, 0.012)
        img += brain * m * rng.uniform(0.38, 0.5)
    img *= rng.uniform(0.85, 1.15)
    img += rng.normal(0, 0.025, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def synthetic_dataset(n_per_class: int, n_classes: int, seed: int):
    """Returns images (list of 64x64 float arrays), labels, class names."""
    rng = np.random.default_rng(seed)
    imgs, ys = [], []
    if n_classes == 2:
        names = ["No Tumor", "Tumor"]
        for _ in range(n_per_class):
            imgs.append(load_gray(make_mri(0, rng))); ys.append(0)
            imgs.append(load_gray(make_mri(int(rng.integers(1, 4)), rng))); ys.append(1)
    else:
        names = SYN_CLASSES
        for c in range(4):
            for _ in range(n_per_class):
                imgs.append(load_gray(make_mri(c, rng))); ys.append(c)
    return imgs, np.array(ys), names


def _scan(root, max_per_class, rng):
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    imgs, ys = [], []
    for ci, c in enumerate(classes):
        files = sorted(f for f in os.listdir(os.path.join(root, c)) if f.lower().endswith(IMG_EXT))
        rng.shuffle(files)
        for f in files[:max_per_class]:
            try:
                imgs.append(load_gray(os.path.join(root, c, f))); ys.append(ci)
            except Exception:
                pass
    return imgs, np.array(ys), classes


def folder_dataset(root: str, max_per_class: int, seed: int):
    """Class-per-subfolder dataset. Supports the Kaggle layout with Training/ and Testing/.
    Returns (train_imgs, train_y, test_imgs, test_y, class_names)."""
    rng = np.random.default_rng(seed)
    lower = {d.lower(): d for d in os.listdir(root)}
    if "training" in lower and "testing" in lower:
        tr_i, tr_y, names = _scan(os.path.join(root, lower["training"]), max_per_class, rng)
        te_i, te_y, _ = _scan(os.path.join(root, lower["testing"]), max(1, max_per_class // 4), rng)
        return tr_i, tr_y, te_i, te_y, [_PRETTY.get(n.lower(), n) for n in names]
    imgs, ys, names = _scan(root, max_per_class, rng)
    if len(imgs) < 10:
        raise ValueError("Found fewer than 10 images – expected one sub-folder per class.")
    p = rng.permutation(len(imgs)); cut = int(0.8 * len(p))
    tr, te = p[:cut], p[cut:]
    return ([imgs[i] for i in tr], ys[tr], [imgs[i] for i in te], ys[te],
            [_PRETTY.get(n.lower(), n) for n in names])
