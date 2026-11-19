"""Tkinter front-end: drag & drop classify, animated circuit, training dashboard, comparison."""
from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
import traceback
from dataclasses import asdict
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from . import data as D
from .trainer import Config, Experiment, MODEL_KEYS, MODEL_NAMES, train

try:                                    # optional native-feeling drag & drop
    from tkinterdnd2 import DND_FILES, TkinterDnD
    BaseTk, HAS_DND = TkinterDnD.Tk, True
except Exception:                       # pragma: no cover
    BaseTk, HAS_DND, DND_FILES = tk.Tk, False, None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")

# ------------------------------------------------------------------ theme
BG, PANEL, PANEL2, TRACK = "#0e1120", "#161a2e", "#1d2240", "#2a3050"
TEXT, MUTED = "#e8eaf6", "#8b92b0"
CYAN, PURPLE, ORANGE, GREEN, RED, AMBER = "#4cc9f0", "#9b5de5", "#f9a03f", "#2ec4b6", "#ef476f", "#ffd166"
MODEL_COLOR = {"quantum": CYAN, "mlp_matched": ORANGE, "mlp_full": GREEN}
SHORT = {"quantum": "Quantum (VQC)", "mlp_matched": "MLP matched", "mlp_full": "MLP full"}
FONT = "Helvetica"
TH = 98   # pipeline thumbnail size


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def conf_color(c):
    return GREEN if c >= 0.85 else AMBER if c >= 0.6 else RED


def arr_to_photo(a, size, heat=False):
    a = np.clip(a, 0, 1)
    if heat:
        rgb = np.stack([a, a * 0.55, a * 0.1], -1)
        a8 = (rgb * 255).astype(np.uint8)
        im = Image.fromarray(a8, "RGB")
    else:
        im = Image.fromarray((a * 255).astype(np.uint8), "L").convert("RGB")
    return ImageTk.PhotoImage(im.resize((size, size), Image.BICUBIC))


def style_mpl():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": "#10132a", "axes.edgecolor": TRACK,
        "axes.labelcolor": MUTED, "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": TRACK, "grid.alpha": 0.6, "axes.titlecolor": TEXT, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.facecolor": PANEL2, "legend.edgecolor": TRACK,
        "legend.labelcolor": TEXT, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# ------------------------------------------------------------------ drop zone
class DropZone(tk.Canvas):
    S = 300

    def __init__(self, parent, on_path):
        super().__init__(parent, width=self.S, height=self.S, bg=PANEL, highlightthickness=0, cursor="hand2")
        self.on_path, self.photo, self.hover = on_path, None, False
        self.bind("<Button-1>", lambda e: self.browse())
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        if HAS_DND:
            try:
                self.drop_target_register(DND_FILES)
                self.dnd_bind("<<Drop>>", self._drop)
                self.dnd_bind("<<DropEnter>>", lambda e: self._set_hover(True))
                self.dnd_bind("<<DropLeave>>", lambda e: self._set_hover(False))
            except Exception:
                pass
        self.draw()

    def _set_hover(self, v):
        self.hover = v
        self.draw()

    def _drop(self, e):
        try:
            paths = self.tk.splitlist(e.data)
            if paths:
                self.on_path(paths[0])
        finally:
            self._set_hover(False)
        return e.action if hasattr(e, "action") else None

    def browse(self):
        p = filedialog.askopenfilename(title="Select a brain MRI image",
                                       filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
        if p:
            self.on_path(p)

    def draw(self):
        self.delete("all")
        S, col = self.S, (CYAN if self.hover else TRACK)
        self.create_rectangle(2, 2, S - 2, S - 2, outline=col, width=2, dash=(7, 5), fill=PANEL2 if self.hover else PANEL)
        if self.photo is not None:
            self.create_image(S // 2, S // 2, image=self.photo)
            self.create_rectangle(2, 2, S - 2, S - 2, outline=col, width=2, dash=(7, 5))
            return
        cx, cy = S // 2, S // 2 - 22
        self.create_polygon(cx - 26, cy + 12, cx, cy - 26, cx + 26, cy + 12, cx + 10, cy + 12, cx + 10, cy + 34,
                            cx - 10, cy + 34, cx - 10, cy + 12, fill=col, outline="")
        self.create_text(cx, cy + 70, text="Drop a brain MRI here" if HAS_DND else "Click to choose a brain MRI",
                         fill=TEXT, font=(FONT, 13, "bold"))
        self.create_text(cx, cy + 94, text="or click to browse  ·  PNG / JPG / BMP / TIFF" if HAS_DND else
                         "(pip install tkinterdnd2 enables drag & drop)", fill=MUTED, font=(FONT, 9))

    def show(self, pil):
        im = pil.convert("L")
        w, h = im.size
        sc = (self.S - 16) / max(w, h)
        im = im.resize((max(1, int(w * sc)), max(1, int(h * sc))), Image.BICUBIC).convert("RGB")
        self.photo = ImageTk.PhotoImage(im)
        self.draw()


# ------------------------------------------------------------------ animated circuit
class CircuitView(ttk.Frame):
    LEFT, RIGHT, TOP = 70, 190, 36

    def __init__(self, parent, on_hover=None):
        super().__init__(parent)
        self.cv = tk.Canvas(self, bg=PANEL, highlightthickness=0, height=290)
        self.sb = ttk.Scrollbar(self, orient="horizontal", command=self.cv.xview)
        self.cv.configure(xscrollcommand=self.sb.set)
        self.cv.pack(fill="both", expand=True)
        self.sb.pack(fill="x")
        self.on_hover = on_hover or (lambda s: None)
        self.qnn, self.angles, self.trace = None, None, None
        self.cur, self.z, self.job, self.playing = -2, None, None, False
        self.cv.bind("<Configure>", lambda e: self.redraw())

    # -- public
    def set_model(self, qnn):
        self.stop()
        self.qnn, self.angles, self.trace, self.cur = qnn, None, None, -2
        self.z = np.ones(qnn.n)
        self.redraw()

    def set_sample(self, angles, trace):
        self.stop()
        self.angles, self.trace, self.cur = angles, trace, -2
        self.z = np.ones(self.qnn.n)
        self.redraw()

    def stop(self):
        if self.job:
            self.after_cancel(self.job)
        self.job, self.playing = None, False

    def play(self, on_done=None):
        if not self.qnn or self.trace is None:
            return
        self.stop()
        self.cur, self.playing, self.on_done = -1, True, on_done
        self.z = np.ones(self.qnn.n)
        self.delay = int(min(200, max(60, 3200 / max(1, len(self.qnn.cols)))))
        self._tick()

    def _tick(self):
        self.cur += 1
        nc = len(self.qnn.cols)
        if self.cur >= nc:
            self.playing, self.job = False, None
            self.cur = nc - 1 + 1          # everything executed, no active column
            self.z = self.trace[-1]
            self.redraw()
            if self.on_done:
                self.on_done()
            return
        self.z = self.trace[self.cur + 1]
        self.redraw(follow=True)
        self.job = self.after(self.delay, self._tick)

    # -- drawing
    def redraw(self, follow=False):
        c = self.cv
        c.delete("all")
        q = self.qnn
        if q is None:
            return
        n, cols = q.n, q.cols
        vw = max(c.winfo_width(), 600)
        nc = len(cols)
        colw = max(34, min(58, (vw - self.LEFT - self.RIGHT) / nc))
        H = max(c.winfo_height(), 260)
        rowh = max(28, min(54, (H - self.TOP - 14) / n))
        gh, gw = min(30, rowh - 10), colw - 8
        end_x = self.LEFT + nc * colw
        total_w = end_x + self.RIGHT
        c.configure(scrollregion=(0, 0, total_w, H))
        ry = lambda i: self.TOP + rowh * (i + 0.5)
        cx = lambda col: self.LEFT + (col + 0.5) * colw

        # block headers
        blocks, seen = {}, []
        for ci, col in enumerate(cols):
            for idx in col:
                b = q.ops[idx].block
                if b not in blocks:
                    blocks[b] = [ci, ci]; seen.append(b)
                blocks[b][0], blocks[b][1] = min(blocks[b][0], ci), max(blocks[b][1], ci)
        for b in seen:
            a, z = blocks[b]
            x0, x1 = self.LEFT + a * colw + 2, self.LEFT + (z + 1) * colw - 2
            tint = CYAN if b == "Feature map" else PURPLE
            c.create_rectangle(x0, 6, x1, 24, fill=mix(PANEL, tint, 0.18), outline=mix(PANEL, tint, 0.5))
            c.create_text((x0 + x1) / 2, 15, text=b if x1 - x0 > 90 else b.replace("Variational layer", "L"),
                          fill=mix(tint, "#ffffff", 0.3), font=(FONT, 8, "bold"))
        # active column band
        if 0 <= self.cur < nc:
            c.create_rectangle(self.LEFT + self.cur * colw, self.TOP - 4, self.LEFT + (self.cur + 1) * colw,
                               self.TOP + n * rowh + 2, fill="#262d52", outline=CYAN, width=1)
            if follow:
                x = cx(self.cur)
                vis = c.winfo_width()
                if total_w > vis:
                    c.xview_moveto(max(0.0, min(1.0, (x - vis * 0.45) / total_w)))
        # wires & labels
        for i in range(n):
            c.create_line(self.LEFT - 18, ry(i), end_x + 6, ry(i), fill=TRACK, width=2)
            c.create_text(12, ry(i), text=f"q{i}", anchor="w", fill=TEXT, font=(FONT, 10, "bold"))
            c.create_text(self.LEFT - 22, ry(i), text="|0>", anchor="e", fill=MUTED, font=(FONT, 9))
        # gates
        ang = self.angles
        for ci, col in enumerate(cols):
            done, active = ci < self.cur or (self.cur >= nc), ci == self.cur
            for idx in col:
                op = q.ops[idx]
                base = {"encode": CYAN, "var": PURPLE, "ent": ORANGE}[op.group]
                if op.kind == "H":
                    base = GREEN
                fill = base if (done or active) else mix(PANEL, base, 0.38)
                outline = "#ffffff" if active else mix(fill, "#000000", 0.25)
                x = cx(ci)
                tag = f"g{idx}"
                if op.kind == "CNOT":
                    k, t = op.qubits
                    yc, yt = ry(k), ry(t)
                    c.create_line(x, yc, x, yt, fill=fill, width=2, tags=tag)
                    c.create_oval(x - 4, yc - 4, x + 4, yc + 4, fill=fill, outline=outline, tags=tag)
                    c.create_oval(x - 10, yt - 10, x + 10, yt + 10, fill=PANEL, outline=fill, width=2, tags=tag)
                    c.create_line(x - 10, yt, x + 10, yt, fill=fill, width=2, tags=tag)
                    c.create_line(x, yt - 10, x, yt + 10, fill=fill, width=2, tags=tag)
                elif op.kind == "RZZ":
                    a_, b_ = op.qubits
                    ya, yb = ry(a_), ry(b_)
                    c.create_line(x, ya, x, yb, fill=fill, width=2, tags=tag)
                    for yy in (ya, yb):
                        c.create_oval(x - 5, yy - 5, x + 5, yy + 5, fill=fill, outline=outline, tags=tag)
                    ym = (ya + yb) / 2
                    c.create_rectangle(x - gw / 2 + 2, ym - 9, x + gw / 2 - 2, ym + 9, fill=fill, outline=outline, tags=tag)
                    c.create_text(x, ym, text="ZZ", fill="#0b0e1c", font=(FONT, 8, "bold"), tags=tag)
                else:
                    y = ry(op.qubits[0])
                    c.create_rectangle(x - gw / 2, y - gh / 2, x + gw / 2, y + gh / 2, fill=fill, outline=outline,
                                       width=2 if active else 1, tags=tag)
                    c.create_text(x, y, text=op.label, fill="#0b0e1c" if (done or active) else mix(PANEL, "#ffffff", 0.5),
                                  font=(FONT, 9, "bold"), tags=tag)
                txt = self._describe(op, ang[idx] if ang else None)
                c.tag_bind(tag, "<Enter>", lambda e, t=txt: self.on_hover(t))
                c.tag_bind(tag, "<Leave>", lambda e: self.on_hover(""))
        # <Z> read-outs
        x0 = end_x + 28
        c.create_text(x0 + 70, 15, text="measure  <Z>", fill=TEXT, font=(FONT, 9, "bold"))
        for i in range(n):
            y = ry(i)
            v = float(self.z[i]) if self.z is not None else 1.0
            c.create_rectangle(x0, y - 7, x0 + 120, y + 7, fill=TRACK, outline="")
            c.create_line(x0 + 60, y - 9, x0 + 60, y + 9, fill=MUTED)
            col = CYAN if v >= 0 else ORANGE
            c.create_rectangle(x0 + 60, y - 7, x0 + 60 + 60 * v, y + 7, fill=col, outline="")
            c.create_text(x0 + 128, y, text=f"{v:+.2f}", anchor="w", fill=TEXT, font=(FONT, 9))

    @staticmethod
    def _describe(op, a):
        qs = ",".join(f"q{i}" for i in op.qubits)
        kind = {"encode": "feature-map gate", "var": "trainable gate", "ent": "entangling gate"}[op.group]
        s = f"{op.kind} on {qs}  ({kind}"
        if a is not None:
            s += f", angle = {a:+.3f} rad"
        if op.src and op.src[0] in ("x", "xx"):
            s += f", encodes image feature{'s' if op.src[0] == 'xx' else ''}"
        return s + ")"


# ------------------------------------------------------------------ result panel
class ResultPanel(tk.Canvas):
    def __init__(self, parent):
        super().__init__(parent, bg=PANEL, highlightthickness=0, width=440, height=420)
        self.job, self.data = None, None
        self.clear()

    def clear(self, msg="Load an MRI image to see the prediction"):
        if self.job:
            self.after_cancel(self.job)
        self.delete("all")
        self.create_text(220, 200, text=msg, fill=MUTED, font=(FONT, 11), width=360, justify="center")

    def show(self, names, qp, cp, truth=None):
        if self.job:
            self.after_cancel(self.job)
        self.data = (names, qp, cp, truth)
        self.config(height=210 + 2 * (len(names) * 21 + 30))
        self.t0 = time.perf_counter()
        self._frame()

    def _frame(self):
        t = min(1.0, (time.perf_counter() - self.t0) / 0.9)
        self._draw(1 - (1 - t) ** 3)
        self.job = self.after(25, self._frame) if t < 1 else None

    def _draw(self, t):
        names, qp, cp, truth = self.data
        self.delete("all")
        qi, ci = int(np.argmax(qp)), int(np.argmax(cp))
        conf = float(qp[qi])
        col = conf_color(conf)
        cx, cy, r = 100, 105, 70
        self.create_oval(cx - r, cy - r, cx + r, cy + r, outline=TRACK, width=14)
        self.create_arc(cx - r, cy - r, cx + r, cy + r, start=90, extent=-359.9 * conf * t, style="arc",
                        outline=col, width=14)
        self.create_text(cx, cy - 6, text=f"{conf * t * 100:.1f}%", fill=TEXT, font=(FONT, 22, "bold"))
        self.create_text(cx, cy + 22, text="confidence", fill=MUTED, font=(FONT, 9))
        self.create_text(200, 52, text="HYBRID QUANTUM PREDICTION", anchor="w", fill=MUTED, font=(FONT, 8, "bold"))
        self.create_text(200, 82, text=names[qi], anchor="w", fill=col, font=(FONT, 20, "bold"))
        level = "High confidence" if conf >= 0.85 else "Moderate confidence" if conf >= 0.6 else "Low confidence – uncertain"
        self.create_text(200, 112, text=level, anchor="w", fill=TEXT, font=(FONT, 10))
        agree = qi == ci
        self.create_text(200, 134, anchor="w", fill=GREEN if agree else AMBER, font=(FONT, 9),
                         text=("Classical MLP agrees" if agree else f"Classical MLP says {names[ci]}"))
        if truth:
            ok = truth == names[qi]
            self.create_text(200, 154, anchor="w", fill=GREEN if ok else RED, font=(FONT, 9),
                             text=f"Ground truth (synthetic): {truth}  {'(correct)' if ok else '(wrong)'}")
        y = 205
        for title, p, color in (("Hybrid quantum model", qp, CYAN), ("Classical MLP (matched)", cp, ORANGE)):
            self.create_text(14, y, text=title, anchor="w", fill=color, font=(FONT, 10, "bold"))
            y += 20
            for i, nm in enumerate(names):
                self.create_text(14, y + 8, text=nm, anchor="w", fill=TEXT, font=(FONT, 9))
                self.create_rectangle(130, y + 1, 380, y + 15, fill=TRACK, outline="")
                self.create_rectangle(130, y + 1, 130 + 250 * float(p[i]) * t, y + 15,
                                      fill=color if i == int(np.argmax(p)) else mix(PANEL, color, 0.45), outline="")
                self.create_text(388, y + 8, text=f"{p[i] * t * 100:5.1f}%", anchor="w", fill=TEXT, font=(FONT, 9))
                y += 21
            y += 10


# ------------------------------------------------------------------ scrolling helper
class ScrollFrame(ttk.Frame):
    """Vertically scrollable container (scrollbar + mouse wheel). Put children in `.inner`."""

    def __init__(self, parent):
        super().__init__(parent)
        self.cv = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.sb = ttk.Scrollbar(self, orient="vertical", command=self.cv.yview)
        self.cv.configure(yscrollcommand=self.sb.set)
        self.sb.pack(side="right", fill="y")
        self.cv.pack(side="left", fill="both", expand=True)
        self.inner = ttk.Frame(self.cv)
        self.win = self.cv.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda e: self.cv.configure(scrollregion=self.cv.bbox("all")))
        self.cv.bind("<Configure>", lambda e: self.cv.itemconfigure(self.win, width=e.width))
        for w in (self.cv, self.inner):
            w.bind("<Enter>", self._bind_wheel); w.bind("<Leave>", self._unbind_wheel)

    def _bind_wheel(self, _e):
        self.cv.bind_all("<MouseWheel>", self._wheel)
        self.cv.bind_all("<Button-4>", lambda e: self.cv.yview_scroll(-2, "units"))
        self.cv.bind_all("<Button-5>", lambda e: self.cv.yview_scroll(2, "units"))

    def _unbind_wheel(self, _e):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.cv.unbind_all(ev)

    def _wheel(self, e):
        step = -1 * (e.delta // 120) if abs(e.delta) >= 120 else -e.delta   # Windows / macOS
        self.cv.yview_scroll(step * 2, "units")

    def scroll_to_bottom(self):
        self.update_idletasks(); self.cv.yview_moveto(1.0)


class CircuitTab(ttk.Frame):
    """Full-size copy of the animated circuit on its own page."""

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        head = ttk.Frame(self); head.pack(fill="x", padx=12, pady=(10, 2))
        ttk.Label(head, text="PARAMETERISED QUANTUM CIRCUIT  (feature map, variational layers, measurement)",
                  style="Sec.TLabel").pack(side="left")
        ttk.Button(head, text="Replay", command=lambda: app.classify.replay()).pack(side="right")
        self.hover = tk.StringVar(value="Hover a gate for details")
        ttk.Label(head, textvariable=self.hover, style="Cap.TLabel").pack(side="right", padx=14)
        legend = ttk.Frame(self); legend.pack(fill="x", padx=12, pady=(2, 3))
        for nm, c in (("feature-map (data)", CYAN), ("Hadamard", GREEN), ("trainable", PURPLE), ("entangling CNOT", ORANGE)):
            tk.Label(legend, text="■", fg=c, bg=BG, font=(FONT, 10)).pack(side="left")
            tk.Label(legend, text=nm + "   ", fg=MUTED, bg=BG, font=(FONT, 8)).pack(side="left")
        self.info = tk.StringVar(value="Classify an image to run the circuit on its features.")
        ttk.Label(self, textvariable=self.info, style="Cap.TLabel").pack(anchor="w", padx=12)
        self.circuit = CircuitView(self, on_hover=lambda s: self.hover.set(s or "Hover a gate for details"))
        self.circuit.pack(fill="both", expand=True, padx=12, pady=8)


# ------------------------------------------------------------------ tabs
class ClassifyTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app, self.res, self.truth, self.pil = app, None, None, None
        self.thumbs = []
        self.rng = np.random.default_rng()
        self.circuit2 = None                      # mirror on the Circuit tab (set by App)
        self.scroll = ScrollFrame(self); self.scroll.pack(fill="both", expand=True)
        body = self.scroll.inner
        body.columnconfigure(1, weight=1)

        left = ttk.Frame(body); left.grid(row=0, column=0, sticky="n", padx=(10, 6), pady=8)
        ttk.Label(left, text="1 · UPLOAD", style="Sec.TLabel").pack(anchor="w", pady=(0, 4))
        self.drop = DropZone(left, self.load_path); self.drop.pack()
        row = ttk.Frame(left); row.pack(fill="x", pady=(8, 0))
        ttk.Button(row, text="Browse…", command=self.drop.browse).pack(side="left")
        self.kind = tk.StringVar(value="Glioma")
        ttk.Button(row, text="Synthetic sample", command=self.sample).pack(side="right")
        ttk.Combobox(row, textvariable=self.kind, values=D.SYN_CLASSES, width=11, state="readonly").pack(side="right", padx=6)

        mid = ttk.Frame(body); mid.grid(row=0, column=1, sticky="nsew", padx=6, pady=8)
        ttk.Label(mid, text="2 · PREPROCESS & EXTRACT FEATURES", style="Sec.TLabel").pack(anchor="w", pady=(0, 4))
        self.pipe = ttk.Frame(mid, style="Card.TFrame"); self.pipe.pack(fill="x")
        self.pimgs, self.pcaps = [], []
        for i in range(5):
            f = ttk.Frame(self.pipe, style="Card.TFrame"); f.grid(row=0, column=i, padx=5, pady=8)
            l = tk.Label(f, bg=PANEL, width=TH, height=TH, image=self._blank(TH)); l.pack()
            self.pimgs.append(l)
            cap = ttk.Label(f, text="", style="Cap.TLabel"); cap.pack(); self.pcaps.append(cap)
        ttk.Label(mid, text="3 · QUANTUM-ENCODED FEATURES  (PCA -> rotation angle in [0, pi])", style="Sec.TLabel").pack(anchor="w", pady=(12, 4))
        self.angcv = tk.Canvas(mid, bg=PANEL, height=150, highlightthickness=0); self.angcv.pack(fill="x")
        self.angcv.bind("<Configure>", lambda e: self.draw_angles())
        self.note = ttk.Label(mid, text="", style="Cap.TLabel", wraplength=620); self.note.pack(anchor="w", pady=(6, 0))

        right = ttk.Frame(body); right.grid(row=0, column=2, sticky="n", padx=(6, 10), pady=8)
        ttk.Label(right, text="5 · CLASSIFY  (classical output layer)", style="Sec.TLabel").pack(anchor="w", pady=(0, 4))
        self.result = ResultPanel(right); self.result.pack()

        box = ttk.Frame(body); box.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=10, pady=(0, 6))
        head = ttk.Frame(box); head.pack(fill="x")
        ttk.Label(head, text="4 · PARAMETERISED QUANTUM CIRCUIT  (feature map, variational layers, measurement)",
                  style="Sec.TLabel").pack(side="left")
        ttk.Button(head, text="Replay", command=self.replay).pack(side="right")
        ttk.Button(head, text="Open on its own page", command=lambda: self.app.nb.select(self.app.circuit_tab)).pack(side="right", padx=8)
        self.hover = tk.StringVar(value="Hover a gate for details")
        ttk.Label(head, textvariable=self.hover, style="Cap.TLabel").pack(side="right", padx=14)
        legend = ttk.Frame(box); legend.pack(fill="x", pady=(2, 3))
        for nm, c in (("feature-map (data)", CYAN), ("Hadamard", GREEN), ("trainable", PURPLE), ("entangling CNOT", ORANGE)):
            tk.Label(legend, text="■", fg=c, bg=BG, font=(FONT, 10)).pack(side="left")
            tk.Label(legend, text=nm + "   ", fg=MUTED, bg=BG, font=(FONT, 8)).pack(side="left")
        self.circuit = CircuitView(box, on_hover=lambda s: self.hover.set(s or "Hover a gate for details"))
        self.circuit.pack(fill="both", expand=True)

    def circuits(self):
        return [c for c in (self.circuit, self.circuit2) if c is not None]

    @staticmethod
    def _blank(s):
        return tk.PhotoImage(width=1, height=1)

    # -- model
    def set_model(self, exp):
        self.res = None
        for c in self.circuits():
            c.set_model(exp.qnn)
        self.result.clear()
        self.draw_angles()
        for l, c in zip(self.pimgs, self.pcaps):
            l.configure(image=""); c.configure(text="")

    # -- actions
    def sample(self):
        if not self.app.exp:
            return messagebox.showinfo("No model", "Train or load a model first (Training tab).")
        k = D.SYN_CLASSES.index(self.kind.get())
        im = Image.fromarray(D.make_mri(k, np.random.default_rng(int(time.time() * 1000) % 2 ** 32)))
        names = self.app.exp.class_names
        truth = D.SYN_CLASSES[k] if len(names) == 4 else ("No Tumor" if k == 0 else "Tumor")
        self.classify(im, truth if truth in names else None)

    def load_path(self, path):
        try:
            im = Image.open(path); im.load()
        except Exception as e:
            return messagebox.showerror("Cannot open image", f"{path}\n\n{e}")
        self.classify(im, None)

    def classify(self, pil, truth):
        if not self.app.exp:
            return messagebox.showinfo("No model", "Train or load a model first (Training tab).")
        self.pil, self.truth = pil, truth
        self.drop.show(pil)
        exp = self.app.exp
        res = exp.predict_image(pil)
        self.res = res
        # pipeline thumbnails
        self.thumbs = []
        items = [("Original", np.asarray(pil.convert("L").resize((64, 64), Image.BICUBIC), dtype=float) / 255, False)]
        for nm, a in res["stages"].items():
            items.append((nm, a, nm.startswith("Hot")))
        for l, c, (nm, a, heat) in zip(self.pimgs, self.pcaps, items):
            ph = arr_to_photo(a, TH, heat)
            self.thumbs.append(ph)
            l.configure(image=ph, width=TH, height=TH)
            c.configure(text=nm)
        self.draw_angles()
        q = exp.qnn
        angs, tr = q.op_angles(res["angles"]), q.trace(res["angles"])
        for c in self.circuits():
            c.set_sample(angs, tr)
        self.app.circuit_tab.info.set("Image features -> rotation angles: " + ", ".join(f"x{i}={a:.2f}" for i, a in enumerate(res["angles"])))
        self.result.clear("Running quantum circuit …")
        self.circuit.play(on_done=self._show_result)
        if self.circuit2 is not None:
            self.circuit2.play()
        self.note.configure(text="Features: intensity statistics, L/R asymmetry, hot-spot geometry & 4x4 grids → "
                                 f"standardised -> PCA({q.n}) -> min-max -> angles.")

    def _show_result(self):
        r, names = self.res, self.app.exp.class_names
        self.result.show(names, r["quantum"], r["mlp_matched"], self.truth)

    def replay(self):
        if self.res is not None:
            self.result.clear("Running quantum circuit …")
            self.circuit.play(on_done=self._show_result)
            if self.circuit2 is not None:
                self.circuit2.play()

    def draw_angles(self):
        c = self.angcv
        c.delete("all")
        exp = self.app.exp
        if not exp:
            return
        n = exp.qnn.n
        W = max(c.winfo_width(), 400)
        a = self.res["angles"] if self.res else None
        slot = min(110, (W - 40) / n)
        for i in range(n):
            x = 30 + i * slot
            c.create_rectangle(x, 14, x + slot * 0.55, 116, fill=TRACK, outline="")
            if a is not None:
                h = 102 * float(a[i]) / np.pi
                c.create_rectangle(x, 116 - h, x + slot * 0.55, 116, fill=CYAN, outline="")
                c.create_text(x + slot * 0.275, 116 - h - 9, text=f"{a[i]:.2f}", fill=TEXT, font=(FONT, 9, "bold"))
            c.create_text(x + slot * 0.275, 132, text=f"x{i} -> q{i}", fill=MUTED, font=(FONT, 9))
        c.create_text(W - 8, 14, text="pi", anchor="e", fill=MUTED, font=(FONT, 9))
        c.create_text(W - 8, 116, text="0", anchor="e", fill=MUTED, font=(FONT, 9))


class TrainTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.q, self.stop_evt, self.thread = queue.Queue(), threading.Event(), None
        self.hist = {k: dict(loss=[], acc=[], val_loss=[], val_acc=[]) for k in MODEL_KEYS}
        self.t_start = 0.0
        self.columnconfigure(1, weight=1); self.rowconfigure(0, weight=1)

        ctl = ttk.Frame(self, width=300); ctl.grid(row=0, column=0, sticky="ns", padx=(10, 6), pady=8)
        v = self.v = dict(source=tk.StringVar(value="synthetic"), folder=tk.StringVar(), ncls=tk.StringVar(value="2 classes: Tumor / No tumor"),
                          per=tk.IntVar(value=120), nq=tk.IntVar(value=4), layers=tk.IntVar(value=3),
                          fmap=tk.StringVar(value="ZZ feature map (entangling)"), reps=tk.IntVar(value=1),
                          epochs=tk.IntVar(value=30), lr=tk.DoubleVar(value=0.08), batch=tk.IntVar(value=32), seed=tk.IntVar(value=7))

        def group(title):
            f = ttk.LabelFrame(ctl, text=title, padding=8); f.pack(fill="x", pady=(0, 8)); f.columnconfigure(1, weight=1); return f

        def row(f, r, label, w):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2, padx=(0, 8)); w.grid(row=r, column=1, sticky="ew", pady=2)

        g = group("Dataset")
        ttk.Radiobutton(g, text="Synthetic MRI-like demo data", variable=v["source"], value="synthetic").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(g, text="Image folder (class sub-folders)", variable=v["source"], value="folder").grid(row=1, column=0, columnspan=2, sticky="w")
        fr = ttk.Frame(g); fr.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2); fr.columnconfigure(0, weight=1)
        ttk.Entry(fr, textvariable=v["folder"]).grid(row=0, column=0, sticky="ew")
        ttk.Button(fr, text="…", width=3, command=self.pick_folder).grid(row=0, column=1, padx=(4, 0))
        row(g, 3, "Classes (demo)", ttk.Combobox(g, textvariable=v["ncls"], state="readonly", width=26,
                                                   values=["2 classes: Tumor / No tumor", "4 classes: Glio/Men/Pit/None"]))
        row(g, 4, "Images / class", ttk.Spinbox(g, from_=20, to=2000, increment=20, textvariable=v["per"], width=8))
        g = group("Quantum model")
        row(g, 0, "Qubits (= features)", ttk.Spinbox(g, from_=2, to=8, textvariable=v["nq"], width=8))
        row(g, 1, "Variational layers", ttk.Spinbox(g, from_=1, to=6, textvariable=v["layers"], width=8))
        row(g, 2, "Feature map", ttk.Combobox(g, textvariable=v["fmap"], state="readonly", width=26,
                                               values=["ZZ feature map (entangling)", "Angle encoding (RY)"]))
        row(g, 3, "Feature-map reps", ttk.Spinbox(g, from_=1, to=3, textvariable=v["reps"], width=8))
        g = group("Optimisation (Adam)")
        row(g, 0, "Epochs", ttk.Spinbox(g, from_=1, to=300, textvariable=v["epochs"], width=8))
        row(g, 1, "Learning rate", ttk.Spinbox(g, from_=0.001, to=0.5, increment=0.01, textvariable=v["lr"], width=8))
        row(g, 2, "Batch size", ttk.Spinbox(g, from_=4, to=256, increment=4, textvariable=v["batch"], width=8))
        row(g, 3, "Seed", ttk.Spinbox(g, from_=0, to=9999, textvariable=v["seed"], width=8))
        bt = ttk.Frame(ctl); bt.pack(fill="x", pady=4)
        self.btn = ttk.Button(bt, text="Train all models", style="Accent.TButton", command=self.start); self.btn.pack(side="left", fill="x", expand=True)
        self.stopb = ttk.Button(bt, text="Stop", command=self.stop_evt.set, state="disabled"); self.stopb.pack(side="left", padx=(6, 0))
        ttk.Label(ctl, text="The quantum model is simulated exactly on a state-vector\nsimulator and trained with the parameter-shift rule;\ncost grows ~2^n in qubits.",
                  style="Cap.TLabel", justify="left").pack(anchor="w", pady=(6, 0))

        dash = ttk.Frame(self); dash.grid(row=0, column=1, sticky="nsew", padx=(6, 10), pady=8)
        dash.columnconfigure(0, weight=1); dash.rowconfigure(2, weight=1)
        cards = ttk.Frame(dash); cards.grid(row=0, column=0, sticky="ew")
        self.card = {}
        for i, (k, t) in enumerate((("epoch", "EPOCH"), ("qacc", "QUANTUM VAL ACC"), ("cacc", "CLASSICAL VAL ACC"), ("time", "ELAPSED"))):
            cards.columnconfigure(i, weight=1)
            f = ttk.Frame(cards, style="Card.TFrame", padding=(12, 8)); f.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))
            ttk.Label(f, text=t, style="Cap.TLabel", background=PANEL).pack(anchor="w")
            lab = ttk.Label(f, text="–", style="Big.TLabel"); lab.pack(anchor="w"); self.card[k] = lab
        self.prog = ttk.Progressbar(dash, mode="determinate", maximum=100); self.prog.grid(row=1, column=0, sticky="ew", pady=8)
        style_mpl()
        self.fig = Figure(figsize=(8, 4.4), dpi=100)
        self.ax1, self.ax2 = self.fig.subplots(1, 2)
        self.cv = FigureCanvasTkAgg(self.fig, master=dash); self.cv.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        self.log = tk.Text(dash, height=5, bg=PANEL, fg=MUTED, relief="flat", font=("Courier", 9), state="disabled", highlightthickness=0)
        self.log.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.plot()

    def pick_folder(self):
        p = filedialog.askdirectory(title="Dataset folder (one sub-folder per class)")
        if p:
            self.v["folder"].set(p); self.v["source"].set("folder")

    def say(self, s):
        self.log.configure(state="normal"); self.log.insert("end", s + "\n"); self.log.see("end"); self.log.configure(state="disabled")

    def make_config(self):
        v = self.v
        if v["source"].get() == "folder" and not os.path.isdir(v["folder"].get()):
            raise ValueError("Choose a valid dataset folder.")
        return Config(source=v["source"].get(), folder=v["folder"].get(), n_classes=2 if v["ncls"].get().startswith("2") else 4,
                      n_per_class=int(v["per"].get()), n_qubits=int(v["nq"].get()), layers=int(v["layers"].get()),
                      fmap="zz" if v["fmap"].get().startswith("ZZ") else "angle", reps=int(v["reps"].get()),
                      epochs=int(v["epochs"].get()), lr=float(v["lr"].get()), batch=int(v["batch"].get()), seed=int(v["seed"].get()))

    def start(self):
        try:
            cfg = self.make_config()
        except Exception as e:
            return messagebox.showerror("Invalid settings", str(e))
        if cfg.n_qubits > 6 and not messagebox.askyesno("Slow", "More than 6 qubits trains noticeably slower on a CPU simulator. Continue?"):
            return
        self.hist = {k: dict(loss=[], acc=[], val_loss=[], val_acc=[]) for k in MODEL_KEYS}
        self.stop_evt.clear(); self.btn.configure(state="disabled"); self.stopb.configure(state="normal")
        self.prog["value"] = 0; self.t_start = time.perf_counter()
        self.say(f"— new run: {cfg.n_qubits} qubits, {cfg.layers} layers, {cfg.fmap}, {cfg.epochs} epochs")
        self.plot()

        def work():
            try:
                ex = train(cfg, on_epoch=lambda *a: self.q.put(("epoch",) + a), on_status=lambda s: self.q.put(("status", s)), stop=self.stop_evt)
                self.q.put(("done", ex))
            except Exception:
                self.q.put(("error", traceback.format_exc()))
        self.thread = threading.Thread(target=work, daemon=True); self.thread.start()

    def poll(self):
        dirty = False
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "status":
                    self.app.status.set(m[1])
                elif m[0] == "epoch":
                    _, ep, tot, row, times = m
                    for k in MODEL_KEYS:
                        for key, val in row[k].items():
                            self.hist[k][key].append(val)
                    self.prog["value"] = 100 * ep / tot
                    self.card["epoch"].configure(text=f"{ep}/{tot}")
                    self.card["qacc"].configure(text=f"{row['quantum']['val_acc'] * 100:.1f}%", foreground=CYAN)
                    self.card["cacc"].configure(text=f"{row['mlp_matched']['val_acc'] * 100:.1f}%", foreground=ORANGE)
                    self.say(f"epoch {ep:3d}  Q loss {row['quantum']['loss']:.3f} val {row['quantum']['val_acc']:.3f} | "
                             f"MLP loss {row['mlp_matched']['loss']:.3f} val {row['mlp_matched']['val_acc']:.3f}")
                    dirty = True
                elif m[0] == "done":
                    self.finish(m[1])
                elif m[0] == "error":
                    self.btn.configure(state="normal"); self.stopb.configure(state="disabled")
                    self.say(m[1]); messagebox.showerror("Training failed", m[1].strip().splitlines()[-1])
        except queue.Empty:
            pass
        if self.thread and self.thread.is_alive():
            self.card["time"].configure(text=f"{time.perf_counter() - self.t_start:.0f} s")
        if dirty:
            self.plot()

    def finish(self, ex):
        self.btn.configure(state="normal"); self.stopb.configure(state="disabled")
        self.prog["value"] = 100
        try:
            os.makedirs(MODEL_DIR, exist_ok=True); ex.save(os.path.join(MODEL_DIR, "last.npz"))
        except Exception as e:
            self.say(f"(could not auto-save: {e})")
        self.say("Training finished – results in the Compare tab.")
        self.app.set_experiment(ex, switch="Compare")

    def show_history(self, exp):
        self.hist = exp.history
        self.plot()
        eps = len(exp.history["quantum"]["acc"])
        if eps:
            self.card["epoch"].configure(text=f"{eps}/{eps}")
            self.card["qacc"].configure(text=f"{exp.history['quantum']['val_acc'][-1] * 100:.1f}%", foreground=CYAN)
            self.card["cacc"].configure(text=f"{exp.history['mlp_matched']['val_acc'][-1] * 100:.1f}%", foreground=ORANGE)
            self.card["time"].configure(text=f"{sum(r['time'] for r in exp.results.values()):.0f} s")
            self.prog["value"] = 100

    def plot(self):
        for ax, (key, title) in zip((self.ax1, self.ax2), (("loss", "Cross-entropy loss"), ("acc", "Accuracy"))):
            ax.clear(); ax.grid(True)
            for k in MODEL_KEYS:
                h = self.hist[k]
                x = np.arange(1, len(h[key]) + 1)
                ax.plot(x, h[key], color=MODEL_COLOR[k], lw=2, label=SHORT[k] if key == "loss" else None)
                ax.plot(x, h["val_" + key], color=MODEL_COLOR[k], lw=1.4, ls="--")
            ax.set_title(f"{title}  (solid = train, dashed = validation)", fontsize=9); ax.set_xlabel("epoch")
            if key == "acc":
                ax.set_ylim(0, 1.02)
        self.ax1.legend(loc="upper right", fontsize=8)
        self.fig.tight_layout(); self.cv.draw_idle()


class CompareTab(ttk.Frame):
    COLS = [("model", "Model", 230), ("params", "Params", 70), ("train", "Train acc", 80), ("acc", "Test acc", 80),
            ("prec", "Precision", 80), ("rec", "Recall", 80), ("f1", "F1", 80), ("time", "Train time", 90)]

    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.columnconfigure(0, weight=1); self.rowconfigure(2, weight=1)
        top = ttk.Frame(self); top.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        top.columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(top, columns=[c[0] for c in self.COLS], show="headings", height=3)
        for cid, name, w in self.COLS:
            self.tree.heading(cid, text=name); self.tree.column(cid, width=w, anchor="w" if cid == "model" else "center")
        self.tree.grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Save charts…", command=self.save).grid(row=0, column=1, padx=(10, 0), sticky="n")
        self.verdict = ttk.Label(self, text="Train or load a model to compare quantum and classical performance.",
                                 style="Cap.TLabel", wraplength=1100, justify="left")
        self.verdict.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 4))
        style_mpl()
        self.fig = Figure(figsize=(11, 6), dpi=100)
        self.cv = FigureCanvasTkAgg(self.fig, master=self); self.cv.get_tk_widget().grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 8))

    def save(self):
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")], initialfile="comparison.png")
        if p:
            self.fig.savefig(p, dpi=160, facecolor=self.fig.get_facecolor())

    def update_view(self, exp):
        for i in self.tree.get_children():
            self.tree.delete(i)
        R = exp.results
        best = max(R, key=lambda k: R[k]["acc"])
        for k in MODEL_KEYS:
            r = R[k]
            self.tree.insert("", "end", values=(MODEL_NAMES[k], r["params"], f"{r['train_acc'] * 100:.1f}%", f"{r['acc'] * 100:.1f}%",
                                                f"{r['precision'] * 100:.1f}%", f"{r['recall'] * 100:.1f}%", f"{r['f1'] * 100:.1f}%", f"{r['time']:.1f} s"))
        q, m, f = R["quantum"], R["mlp_matched"], R["mlp_full"]
        d = (q["acc"] - m["acc"]) * 100
        rel = "on par with" if abs(d) < 1 else ("ahead of" if d > 0 else "behind")
        self.verdict.configure(text=(f"Quantum test accuracy {q['acc'] * 100:.1f}% with {q['params']} parameters is {rel} the parameter-matched classical MLP "
                                     f"({m['acc'] * 100:.1f}%, {m['params']} params); the full-feature MLP reaches {f['acc'] * 100:.1f}% with {f['params']} params. "
                                     f"Quantum training took {q['time']:.1f} s vs {m['time']:.2f} s – the simulator is the cost, not the model. "
                                     f"Best test accuracy: {MODEL_NAMES[best]}."))
        fig = self.fig; fig.clear()
        gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.5, wspace=0.35, left=0.06, right=0.98, top=0.94, bottom=0.08)
        ax = fig.add_subplot(gs[0, :2]); ax.grid(True, axis="y")
        labs, keys = ["Accuracy", "Precision", "Recall", "F1"], ["acc", "precision", "recall", "f1"]
        w = 0.26
        for i, k in enumerate(MODEL_KEYS):
            vals = [R[k][kk] for kk in keys]
            bars = ax.bar(np.arange(4) + (i - 1) * w, vals, w, color=MODEL_COLOR[k], label=SHORT[k])
            for b, vv in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, vv + 0.01, f"{vv * 100:.0f}", ha="center", fontsize=7, color=TEXT)
        ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylim(0, 1.3); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_title("Test-set metrics (%)"); ax.legend(ncol=3, fontsize=8, loc="upper center")
        ax = fig.add_subplot(gs[0, 2]); ax.grid(True, axis="y")
        ax.bar([SHORT[k].replace(" ", "\n", 1) for k in MODEL_KEYS], [R[k]["params"] for k in MODEL_KEYS], color=[MODEL_COLOR[k] for k in MODEL_KEYS])
        ax.set_yscale("log"); ax.set_title("Trainable parameters (log)")
        for i, k in enumerate(MODEL_KEYS):
            ax.text(i, R[k]["params"] * 1.1, str(R[k]["params"]), ha="center", fontsize=8, color=TEXT)
        ax.tick_params(axis="x", labelsize=7)
        names = exp.class_names
        for i, k in enumerate(MODEL_KEYS):
            ax = fig.add_subplot(gs[1, i]); cm = R[k]["cm"]
            ax.imshow(cm, cmap="Blues" if k == "quantum" else "Oranges" if k == "mlp_matched" else "Greens")
            ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
            ab = [n[:4] for n in names]
            ax.set_xticklabels(ab, fontsize=7); ax.set_yticklabels(ab, fontsize=7)
            ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{SHORT[k]} – confusion", fontsize=9)
            ax.spines[:].set_visible(False)
            for a_ in range(cm.shape[0]):
                for b_ in range(cm.shape[1]):
                    ax.text(b_, a_, cm[a_, b_], ha="center", va="center", fontsize=9, color="#ffffff" if cm[a_, b_] > cm.max() * 0.5 else "#0b0e1c")
        self.cv.draw_idle()


# ------------------------------------------------------------------ app
class App(BaseTk):
    def __init__(self):
        super().__init__()
        self.title("QuantumMRI – Hybrid Quantum-Classical Brain MRI Classifier")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1400, sw - 40)}x{min(900, sh - 70)}+10+10")
        self.minsize(1100, 680)
        self.configure(bg=BG)
        self.exp: Experiment | None = None
        self.status = tk.StringVar(value="Ready")
        self._style()
        self._menu()

        head = tk.Frame(self, bg=BG); head.pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(head, text="QuantumMRI", bg=BG, fg=CYAN, font=(FONT, 18, "bold")).pack(side="left")
        tk.Label(head, text="  Hybrid quantum-classical brain-MRI classification  ·  variational quantum circuit + classical output layer",
                 bg=BG, fg=MUTED, font=(FONT, 10)).pack(side="left", pady=(6, 0))
        self.chip = tk.Label(head, text="no model", bg=PANEL2, fg=TEXT, font=(FONT, 9), padx=10, pady=4); self.chip.pack(side="right")

        self.nb = ttk.Notebook(self); self.nb.pack(fill="both", expand=True, padx=6, pady=4)
        self.classify = ClassifyTab(self.nb, self); self.train_tab = TrainTab(self.nb, self); self.compare = CompareTab(self.nb, self)
        self.circuit_tab = CircuitTab(self.nb, self); self.classify.circuit2 = self.circuit_tab.circuit
        self.nb.add(self.classify, text="  Classify  "); self.nb.add(self.circuit_tab, text="  Quantum Circuit  "); self.nb.add(self.train_tab, text="  Training  "); self.nb.add(self.compare, text="  Compare  ")
        bar = tk.Frame(self, bg=PANEL); bar.pack(fill="x", side="bottom")
        tk.Label(bar, textvariable=self.status, bg=PANEL, fg=MUTED, font=(FONT, 9), anchor="w").pack(side="left", padx=10, pady=3)
        tk.Label(bar, text="Research / education demo – NOT a medical device. Default model uses synthetic images.", bg=PANEL, fg=AMBER,
                 font=(FONT, 9)).pack(side="right", padx=10)
        self.after(100, self._poll)
        self.after(200, self._autoload)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL2, bordercolor=TRACK, lightcolor=BG, darkcolor=BG, font=(FONT, 10))
        s.configure("TFrame", background=BG); s.configure("Card.TFrame", background=PANEL)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("Sec.TLabel", foreground=CYAN, font=(FONT, 9, "bold"))
        s.configure("Cap.TLabel", foreground=MUTED, font=(FONT, 9)); s.configure("Big.TLabel", background=PANEL, foreground=TEXT, font=(FONT, 20, "bold"))
        s.configure("TLabelframe", background=BG, bordercolor=TRACK); s.configure("TLabelframe.Label", background=BG, foreground=PURPLE, font=(FONT, 9, "bold"))
        s.configure("TButton", background=PANEL2, foreground=TEXT, padding=(10, 5), borderwidth=0)
        s.map("TButton", background=[("active", TRACK), ("disabled", PANEL)], foreground=[("disabled", MUTED)])
        s.configure("Accent.TButton", background=CYAN, foreground="#06101c", font=(FONT, 10, "bold"))
        s.map("Accent.TButton", background=[("active", "#7ddcf7"), ("disabled", TRACK)], foreground=[("disabled", MUTED)])
        s.configure("TNotebook", background=BG, borderwidth=0); s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(14, 7), borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", PANEL2)], foreground=[("selected", CYAN)])
        for w in ("TCombobox", "TSpinbox", "TEntry"):
            s.configure(w, fieldbackground=PANEL2, foreground=TEXT, arrowcolor=TEXT, insertcolor=TEXT, selectbackground=TRACK)
        s.map("TCombobox", fieldbackground=[("readonly", PANEL2)], foreground=[("readonly", TEXT)], selectbackground=[("readonly", PANEL2)], selectforeground=[("readonly", TEXT)])
        s.configure("TRadiobutton", background=BG, foreground=TEXT); s.map("TRadiobutton", background=[("active", BG)])
        s.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=26, borderwidth=0)
        s.configure("Treeview.Heading", background=PANEL2, foreground=CYAN, font=(FONT, 9, "bold"), borderwidth=0)
        s.map("Treeview", background=[("selected", TRACK)])
        s.configure("Horizontal.TProgressbar", background=CYAN, troughcolor=TRACK, borderwidth=0)
        s.configure("Horizontal.TScrollbar", background=PANEL2, troughcolor=PANEL, borderwidth=0, arrowcolor=MUTED)
        self.option_add("*TCombobox*Listbox.background", PANEL2); self.option_add("*TCombobox*Listbox.foreground", TEXT)

    def _menu(self):
        m = tk.Menu(self); self.config(menu=m)
        f = tk.Menu(m, tearoff=0); m.add_cascade(label="File", menu=f)
        f.add_command(label="Open MRI image…", command=lambda: self.classify.drop.browse())
        f.add_separator()
        f.add_command(label="Save model…", command=self.save_model); f.add_command(label="Load model…", command=self.load_model)
        f.add_separator(); f.add_command(label="Exit", command=self.destroy)
        h = tk.Menu(m, tearoff=0); m.add_cascade(label="Help", menu=h)
        h.add_command(label="About", command=lambda: messagebox.showinfo(
            "About", "QuantumMRI\n\nHybrid quantum-classical classifier: quantum feature map + variational circuit "
                    "(exact NumPy state-vector simulator, parameter-shift gradients) + classical softmax layer.\n\n"
                    "For research and education only – not for diagnosis."))

    # -- model management
    def set_experiment(self, exp, switch=None):
        self.exp = exp
        c = exp.cfg
        acc = exp.results["quantum"]["acc"] * 100 if exp.results else float("nan")
        self.chip.configure(text=f"{len(exp.class_names)} classes · {c.n_qubits} qubits · {c.layers} layers · {'ZZ' if c.fmap == 'zz' else 'RY'} map · "
                                 f"quantum test acc {acc:.1f}%")
        self.classify.set_model(exp)
        self.train_tab.show_history(exp)
        if exp.results:
            self.compare.update_view(exp)
        self.status.set("Model ready.")
        if switch:
            self.nb.select({"Compare": self.compare, "Classify": self.classify}[switch])

    def _autoload(self):
        for nm in ("last.npz", "default.npz"):
            p = os.path.join(MODEL_DIR, nm)
            if os.path.exists(p):
                try:
                    self.set_experiment(Experiment.load(p)); self.status.set(f"Loaded {nm}"); return
                except Exception as e:
                    self.status.set(f"Could not load {nm}: {e}")
        self.status.set("No model yet – open the Training tab and press “Train all models”.")
        self.nb.select(self.train_tab)

    def save_model(self):
        if not self.exp:
            return messagebox.showinfo("Nothing to save", "Train a model first.")
        p = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("Model", "*.npz")], initialdir=MODEL_DIR)
        if p:
            self.exp.save(p); self.status.set(f"Saved {p}")

    def load_model(self):
        p = filedialog.askopenfilename(filetypes=[("Model", "*.npz")], initialdir=MODEL_DIR)
        if p:
            try:
                self.set_experiment(Experiment.load(p), switch="Classify")
            except Exception as e:
                messagebox.showerror("Cannot load model", str(e))

    def _poll(self):
        self.train_tab.poll()
        self.after(100, self._poll)


def main():
    App().mainloop()
