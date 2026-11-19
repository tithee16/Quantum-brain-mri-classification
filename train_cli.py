"""Headless training:  python train_cli.py --classes 4 --qubits 4 --layers 3 --epochs 40 --out models/default.npz"""
import argparse
from qmri.trainer import Config, train, MODEL_KEYS, MODEL_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="synthetic", choices=["synthetic", "folder"])
ap.add_argument("--folder", default="")
ap.add_argument("--classes", type=int, default=2, choices=[2, 4])
ap.add_argument("--per-class", type=int, default=120)
ap.add_argument("--qubits", type=int, default=4)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--fmap", default="zz", choices=["zz", "angle"])
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=0.08)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="models/last.npz")
a = ap.parse_args()
cfg = Config(a.source, a.folder, a.classes, a.per_class, a.qubits, a.layers, a.fmap, a.reps, a.epochs, a.lr, a.batch, a.seed)
ex = train(cfg, on_epoch=lambda ep, tot, row, t: print(f"epoch {ep}/{tot}  quantum val acc {row['quantum']['val_acc']:.3f}  mlp val acc {row['mlp_matched']['val_acc']:.3f}"))
for k in MODEL_KEYS:
    r = ex.results[k]
    print(f"{MODEL_NAMES[k]:34s} test acc {r['acc']:.3f}  F1 {r['f1']:.3f}  params {r['params']:5d}  time {r['time']:.1f}s")
ex.save(a.out); print("saved", a.out)
