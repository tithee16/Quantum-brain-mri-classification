"""Training / evaluation of the hybrid quantum model and the classical baselines."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, Optional

import numpy as np

from . import data as D
from .classical import MLP, matched_hidden
from .features import Preprocessor, extract, load_gray, stages
from .quantum import Adam, HybridQNN

MODEL_KEYS = ["quantum", "mlp_matched", "mlp_full"]
MODEL_NAMES = {"quantum": "Hybrid Quantum (VQC)", "mlp_matched": "Classical MLP (matched params)",
               "mlp_full": "Classical MLP (all features)"}


@dataclass
class Config:
    source: str = "synthetic"          # 'synthetic' | 'folder'
    folder: str = ""
    n_classes: int = 2
    n_per_class: int = 120
    n_qubits: int = 4
    layers: int = 3
    fmap: str = "zz"
    reps: int = 1
    epochs: int = 30
    lr: float = 0.08
    batch: int = 32
    seed: int = 7


def metrics(y, p, C):
    cm = np.zeros((C, C), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    tp = np.diag(cm).astype(float)
    prec = np.where(cm.sum(0) > 0, tp / np.maximum(cm.sum(0), 1), 0)
    rec = np.where(cm.sum(1) > 0, tp / np.maximum(cm.sum(1), 1), 0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return dict(acc=float(tp.sum() / max(cm.sum(), 1)), precision=float(prec.mean()),
                recall=float(rec.mean()), f1=float(f1.mean()), cm=cm)


class Experiment:
    """Everything needed to train, save, load and run inference."""

    def __init__(self, cfg: Config, class_names):
        self.cfg, self.class_names = cfg, list(class_names)
        self.prep: Optional[Preprocessor] = None
        self.qnn: Optional[HybridQNN] = None
        self.mlp_matched: Optional[MLP] = None
        self.mlp_full: Optional[MLP] = None
        self.history: Dict[str, Dict[str, list]] = {k: dict(loss=[], acc=[], val_loss=[], val_acc=[])
                                                    for k in MODEL_KEYS}
        self.results: Dict[str, dict] = {}

    # ---- inputs for each model
    def inputs(self, F):
        ang = self.prep.angles(F)
        return {"quantum": ang, "mlp_matched": ang / np.pi * 2 - 1, "mlp_full": self.prep.standardize(F) / 3}

    def model(self, k):
        return {"quantum": self.qnn, "mlp_matched": self.mlp_matched, "mlp_full": self.mlp_full}[k]

    # ---- inference
    def predict_image(self, src):
        g = load_gray(src)
        F = extract(g)[None]
        X = self.inputs(F)
        out = {"gray": g, "stages": stages(g), "angles": X["quantum"][0]}
        for k in MODEL_KEYS:
            out[k] = self.model(k).predict_proba(X[k])[0]
        out["z"] = self.qnn.expectations(X["quantum"])[0]
        return out

    # ---- persistence
    def save(self, path):
        d = dict(cfg=json.dumps(asdict(self.cfg)), classes=json.dumps(self.class_names),
                 history=json.dumps(self.history),
                 results=json.dumps({k: {**v, "cm": v["cm"].tolist()} for k, v in self.results.items()}),
                 q_theta=self.qnn.theta, q_W=self.qnn.W, q_b=self.qnn.b,
                 m_hidden=self.mlp_matched.hidden, f_hidden=self.mlp_full.hidden,
                 f_nin=self.mlp_full.n_in, **self.prep.state())
        for tag, m in (("mm", self.mlp_matched), ("mf", self.mlp_full)):
            for nm, p in zip(("W1", "b1", "W2", "b2"), m.params()):
                d[f"{tag}_{nm}"] = p
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        cfg = Config(**json.loads(str(z["cfg"])))
        e = cls(cfg, json.loads(str(z["classes"])))
        e.history = json.loads(str(z["history"]))
        e.results = {k: {**v, "cm": np.array(v["cm"])} for k, v in json.loads(str(z["results"])).items()}
        e.prep = Preprocessor.from_state({k: z[k] for k in ("p_mu", "p_sd", "p_comp", "p_lo", "p_hi")})
        C = len(e.class_names)
        e.qnn = HybridQNN(cfg.n_qubits, cfg.layers, cfg.fmap, cfg.reps, C, cfg.seed)
        e.qnn.theta[:], e.qnn.W[:], e.qnn.b[:] = z["q_theta"], z["q_W"], z["q_b"]
        e.mlp_matched = MLP(cfg.n_qubits, int(z["m_hidden"]), C)
        e.mlp_full = MLP(int(z["f_nin"]), int(z["f_hidden"]), C)
        for tag, m in (("mm", e.mlp_matched), ("mf", e.mlp_full)):
            for nm, p in zip(("W1", "b1", "W2", "b2"), m.params()):
                p[:] = z[f"{tag}_{nm}"]
        return e


def train(cfg: Config, on_epoch: Optional[Callable] = None, on_status: Optional[Callable] = None,
          stop=None) -> Experiment:
    say = on_status or (lambda s: None)
    say("Loading dataset …")
    if cfg.source == "folder":
        tr_i, tr_y, te_i, te_y, names = D.folder_dataset(cfg.folder, cfg.n_per_class, cfg.seed)
    else:
        imgs, ys, names = D.synthetic_dataset(cfg.n_per_class, cfg.n_classes, cfg.seed)
        p = np.random.default_rng(cfg.seed).permutation(len(imgs)); cut = int(0.8 * len(p))
        tr_i, tr_y = [imgs[i] for i in p[:cut]], ys[p[:cut]]
        te_i, te_y = [imgs[i] for i in p[cut:]], ys[p[cut:]]
    C = len(names)
    cfg = Config(**{**asdict(cfg), "n_classes": C})
    say(f"Extracting features from {len(tr_i) + len(te_i)} images …")
    Ftr, Fte = np.array([extract(g) for g in tr_i]), np.array([extract(g) for g in te_i])

    ex = Experiment(cfg, names)
    ex.prep = Preprocessor().fit(Ftr, cfg.n_qubits)
    Xtr, Xte = ex.inputs(Ftr), ex.inputs(Fte)

    ex.qnn = HybridQNN(cfg.n_qubits, cfg.layers, cfg.fmap, cfg.reps, C, cfg.seed)
    ex.mlp_matched = MLP(cfg.n_qubits, matched_hidden(cfg.n_qubits, C, ex.qnn.n_params), C, cfg.seed)
    ex.mlp_full = MLP(Ftr.shape[1], 32, C, cfg.seed)
    opts = {"quantum": Adam(ex.qnn.params(), cfg.lr),
            "mlp_matched": Adam(ex.mlp_matched.params(), cfg.lr * 0.5),
            "mlp_full": Adam(ex.mlp_full.params(), cfg.lr * 0.25)}
    rng = np.random.default_rng(cfg.seed + 1)
    t_used = {k: 0.0 for k in MODEL_KEYS}

    def evaluate(k, X, y):
        p = ex.model(k).predict_proba(X)
        return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()), float((p.argmax(1) == y).mean())

    for ep in range(1, cfg.epochs + 1):
        if stop is not None and stop.is_set():
            break
        perm = rng.permutation(len(tr_y))
        row = {}
        for k in MODEL_KEYS:
            t0 = time.perf_counter()
            for s in range(0, len(perm), cfg.batch):
                b = perm[s:s + cfg.batch]
                _, g = ex.model(k).loss_grad(Xtr[k][b], tr_y[b])
                opts[k].step(g)
            t_used[k] += time.perf_counter() - t0
            l, a = evaluate(k, Xtr[k], tr_y)
            vl, va = evaluate(k, Xte[k], te_y)
            h = ex.history[k]
            h["loss"].append(l); h["acc"].append(a); h["val_loss"].append(vl); h["val_acc"].append(va)
            row[k] = dict(loss=l, acc=a, val_loss=vl, val_acc=va)
        if on_epoch:
            on_epoch(ep, cfg.epochs, row, {k: t_used[k] for k in MODEL_KEYS})
        say(f"Epoch {ep}/{cfg.epochs}")

    for k in MODEL_KEYS:
        m = ex.model(k)
        r = metrics(te_y, m.predict_proba(Xte[k]).argmax(1), C)
        r.update(params=int(m.n_params), time=t_used[k], train_acc=ex.history[k]["acc"][-1] if ex.history[k]["acc"] else 0.0)
        ex.results[k] = r
    return ex
