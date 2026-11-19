"""Pre-processing, hand-crafted radiomics-style features, PCA and angle scaling."""
import numpy as np
from PIL import Image

SIZE = 64
_yy, _xx = np.mgrid[0:SIZE, 0:SIZE]
_xn, _yn = (_xx - (SIZE - 1) / 2) / (SIZE / 2), (_yy - (SIZE - 1) / 2) / (SIZE / 2)
INTERIOR = (_xn ** 2 / 0.70 ** 2 + _yn ** 2 / 0.80 ** 2) < 1.0      # approx. brain region


def load_gray(src) -> np.ndarray:
    """Path / PIL image / array -> float32 64x64 in [0,1] with percentile contrast stretch."""
    if isinstance(src, np.ndarray):
        a = src.astype(np.float32)
        im = Image.fromarray(np.clip(a * 255 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8))
    elif isinstance(src, Image.Image):
        im = src
    else:
        im = Image.open(src)
    im = im.convert("L").resize((SIZE, SIZE), Image.BICUBIC)
    a = np.asarray(im, dtype=np.float32) / 255.0
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1e-3), 0, 1)


def box_blur(a, k=5):
    p = k // 2
    c = np.cumsum(np.cumsum(np.pad(a, p, mode="edge"), 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, W = a.shape
    return (c[k:k + H, k:k + W] - c[:H, k:k + W] - c[k:k + H, :W] + c[:H, :W]) / (k * k)


def sobel(a):
    p = np.pad(a, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy) / 4.0


def equalize(a):
    h, _ = np.histogram(a, bins=256, range=(0, 1))
    c = np.cumsum(h).astype(float)
    c = (c - c.min()) / max(c.max() - c.min(), 1)
    return c[np.clip((a * 255).astype(int), 0, 255)]


def _hot(g):
    bl = box_blur(g, 5)
    v = bl[INTERIOR]
    thr = np.median(v) + 1.5 * v.std()
    return bl, (bl > thr) & INTERIOR, thr


def stages(g):
    """Images for the preprocessing visualisation."""
    bl, hot, thr = _hot(g)
    heat = np.clip((bl - thr) / max(bl.max() - thr, 1e-3), 0, 1) * INTERIOR
    return {"Gray 64x64": g, "Enhanced": equalize(g), "Edges (Sobel)": np.clip(sobel(g) * 2.5, 0, 1),
            "Hot-spot map": heat}


FEATURE_NAMES = (["mean", "std", "skew", "kurtosis", "entropy", "p95", "p99-mean", "L/R asym",
                  "hot area", "hot cx", "hot cy", "hot radius", "hot spread", "peak contrast",
                  "grad mean", "grad p95", "edge@hot", "hot asym"]
                 + [f"gridI{i}" for i in range(16)] + [f"gridH{i}" for i in range(16)])


def extract(g) -> np.ndarray:
    m = INTERIOR
    v = g[m]
    mu, sd = v.mean(), v.std() + 1e-6
    skew, kurt = ((v - mu) ** 3).mean() / sd ** 3, ((v - mu) ** 4).mean() / sd ** 4
    hist, _ = np.histogram(v, bins=32, range=(0, 1))
    pr = hist / max(hist.sum(), 1)
    ent = -(pr[pr > 0] * np.log2(pr[pr > 0])).sum()
    p95, p99 = np.percentile(v, 95), np.percentile(v, 99)
    asym = np.abs(g - g[:, ::-1])[m].mean()
    bl, hot, _ = _hot(g)
    area = hot.sum() / m.sum()
    if hot.any():
        ys, xs = np.nonzero(hot)
        cx, cy = (xs.mean() - 31.5) / 32, (ys.mean() - 31.5) / 32
        rad, spread = np.hypot(cx, cy), np.sqrt(xs.var() + ys.var()) / 32
    else:
        cx = cy = rad = spread = 0.0
    peak = bl[m].max() - np.median(bl[m])
    mag = sobel(g)
    edge_hot = mag[hot].mean() if hot.any() else 0.0
    hot_asym = abs(int(hot[:, :32].sum()) - int(hot[:, 32:].sum())) / (hot.sum() + 1)
    gi = g.reshape(4, 16, 4, 16).mean(axis=(1, 3)).ravel()
    gh = hot.astype(float).reshape(4, 16, 4, 16).mean(axis=(1, 3)).ravel()
    f = [mu, sd, skew, kurt, ent, p95, p99 - mu, asym, area, cx, cy, rad, spread, peak,
         mag[m].mean(), np.percentile(mag[m], 95), edge_hot, hot_asym]
    return np.concatenate([np.array(f, dtype=float), gi, gh])


class Preprocessor:
    """standardise -> PCA(k) -> min/max scaling to rotation angles in [0, pi]."""

    def fit(self, F, k):
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-8
        Z = (F - self.mu) / self.sd
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        self.comp = Vt[:k]
        P = Z @ self.comp.T
        self.lo, self.hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
        return self

    def standardize(self, F):
        return np.clip((F - self.mu) / self.sd, -6, 6)

    def project(self, F):
        return self.standardize(F) @ self.comp.T

    def angles(self, F):
        P = self.project(F)
        return np.clip((P - self.lo) / np.maximum(self.hi - self.lo, 1e-8), 0, 1) * np.pi

    def state(self):
        return dict(p_mu=self.mu, p_sd=self.sd, p_comp=self.comp, p_lo=self.lo, p_hi=self.hi)

    @classmethod
    def from_state(cls, d):
        o = cls()
        o.mu, o.sd, o.comp, o.lo, o.hi = d["p_mu"], d["p_sd"], d["p_comp"], d["p_lo"], d["p_hi"]
        return o
