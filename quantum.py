"""
A small, fast, batched state-vector simulator + hybrid quantum-classical model.

Pipeline of the hybrid model
    features x (n_qubits angles in [0, pi])
      -> quantum FEATURE MAP  (angle / ZZ encoding)      [data-dependent gates]
      -> VARIATIONAL ansatz   (RY, RZ, CNOT ring) x L    [trainable gates]
      -> measure <Z_i> on each qubit                      [n_qubits numbers in [-1, 1]]
      -> classical dense layer + softmax                  [class probabilities]

Gradients of the circuit use the exact PARAMETER-SHIFT rule, so the simulator behaves
exactly like quantum hardware would (no back-propagation through the state vector).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

PI = np.pi
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


# ----------------------------------------------------------------------------- circuit description
@dataclass
class Op:
    kind: str                       # 'H', 'RY', 'RZ', 'RZZ', 'CNOT'
    qubits: Tuple[int, ...]
    src: Optional[tuple] = None     # ('x', i, scale) | ('xx', i, j) | ('t', layer, qubit, k)
    group: str = "encode"           # 'encode' | 'var' | 'ent'
    block: str = ""                 # human-readable block name for drawing

    @property
    def label(self) -> str:
        return {"H": "H", "RY": "RY", "RZ": "RZ", "RZZ": "ZZ", "CNOT": "X"}[self.kind]


def build_ops(n: int, layers: int, fmap: str = "zz", reps: int = 1) -> List[Op]:
    """Feature map followed by the variational ansatz."""
    ops: List[Op] = []
    fm = "Feature map"
    if fmap == "angle":
        for i in range(n):
            ops.append(Op("RY", (i,), ("x", i, 1.0), "encode", fm))
    elif fmap == "zz":                                   # Qiskit-style ZZFeatureMap
        for _ in range(reps):
            for i in range(n):
                ops.append(Op("H", (i,), None, "encode", fm))
            for i in range(n):
                ops.append(Op("RZ", (i,), ("x", i, 1.0), "encode", fm))
            for i in range(n - 1):
                ops.append(Op("RZZ", (i, i + 1), ("xx", i, i + 1), "encode", fm))
    else:
        raise ValueError(f"unknown feature map {fmap}")
    for l in range(layers):                              # hardware-efficient ansatz
        blk = f"Variational layer {l + 1}"
        for q in range(n):
            ops.append(Op("RY", (q,), ("t", l, q, 0), "var", blk))
        for q in range(n):
            ops.append(Op("RZ", (q,), ("t", l, q, 1), "var", blk))
        if n > 1:
            for q in range(n if n > 2 else 1):
                ops.append(Op("CNOT", (q, (q + 1) % n), None, "ent", blk))
    return ops


def schedule(ops: List[Op], n: int) -> List[List[int]]:
    """Greedy left-aligned layering. Returns, per column, the indices of ops in it."""
    free = [0] * n
    cols: List[List[int]] = []
    prev_block = None
    for idx, op in enumerate(ops):
        if prev_block is not None and op.block != prev_block:      # barrier between blocks
            free = [max(free)] * n
        prev_block = op.block
        lo, hi = min(op.qubits), max(op.qubits)
        c = max(free[lo:hi + 1])
        for q in range(lo, hi + 1):
            free[q] = c + 1
        while len(cols) <= c:
            cols.append([])
        cols[c].append(idx)
    return cols


# ----------------------------------------------------------------------------- state-vector kernels
def _zero_state(N: int, n: int) -> np.ndarray:
    s = np.zeros((N,) + (2,) * n, dtype=np.complex128)
    s[(slice(None),) + (0,) * n] = 1.0
    return s


def _apply1(state, U, q):
    ax = q + 1
    s = np.moveaxis(state, ax, 1)
    shp = s.shape
    s = s.reshape(shp[0], 2, -1)
    out = np.einsum("ij,njr->nir", U, s) if U.ndim == 2 else np.einsum("nij,njr->nir", U, s)
    return np.moveaxis(out.reshape(shp), 1, ax)


def _bshape(N, n, axes_vals):
    shp = [N] + [1] * n
    for ax in axes_vals:
        shp[ax + 1] = 2
    return shp


def _apply_op(state, op: Op, ang, n):
    N = state.shape[0]
    k = op.kind
    if k == "H":
        return _apply1(state, _H, op.qubits[0])
    if k == "RY":
        c, s = np.cos(ang / 2), np.sin(ang / 2)
        U = np.empty((N, 2, 2), dtype=np.complex128)
        U[:, 0, 0] = c; U[:, 0, 1] = -s; U[:, 1, 0] = s; U[:, 1, 1] = c
        return _apply1(state, U, op.qubits[0])
    if k == "RZ":
        ph = np.stack([np.exp(-0.5j * ang), np.exp(0.5j * ang)], axis=-1)
        return state * ph.reshape(_bshape(N, n, [op.qubits[0]]))
    if k == "RZZ":
        i, j = op.qubits
        zi = np.array([1.0, -1.0]).reshape(_bshape(1, n, [i])[1:])
        zj = np.array([1.0, -1.0]).reshape(_bshape(1, n, [j])[1:])
        return state * np.exp(-0.5j * ang.reshape((N,) + (1,) * n) * (zi * zj)[None])
    if k == "CNOT":
        c, t = op.qubits
        idx = [slice(None)] * (n + 1)
        idx[c + 1] = 1
        sub = state[tuple(idx)]
        ta = t + 1 if t < c else t
        out = state.copy()
        out[tuple(idx)] = np.flip(sub, axis=ta)
        return out
    raise ValueError(k)


def _z_table(n):
    k = np.arange(2 ** n)
    return np.stack([1.0 - 2.0 * ((k >> (n - 1 - q)) & 1) for q in range(n)], axis=1)  # (2^n, n)


def _expect_z(state, n):
    p = (state.real ** 2 + state.imag ** 2).reshape(state.shape[0], -1)
    return p @ _z_table(n)


# ----------------------------------------------------------------------------- hybrid model
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Adam:
    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


class HybridQNN:
    """Quantum feature map + variational circuit + classical softmax output layer."""

    def __init__(self, n_qubits=4, layers=3, fmap="zz", reps=1, n_classes=2, seed=0):
        self.n, self.L, self.fmap, self.reps, self.C = n_qubits, layers, fmap, reps, n_classes
        rng = np.random.default_rng(seed)
        self.theta = rng.uniform(-PI, PI, size=(layers, n_qubits, 2)) * 0.5
        self.W = rng.normal(0, 0.5, size=(n_classes, n_qubits))
        self.b = np.zeros(n_classes)
        self.ops = build_ops(n_qubits, layers, fmap, reps)
        self.enc_ops = [o for o in self.ops if o.group == "encode"]
        self.var_ops = [o for o in self.ops if o.group in ("var", "ent")]
        self.cols = schedule(self.ops, n_qubits)

    # -- bookkeeping
    @property
    def n_circuit_params(self):
        return self.theta.size

    @property
    def n_params(self):
        return self.theta.size + self.W.size + self.b.size

    def params(self):
        return [self.theta, self.W, self.b]

    # -- circuit evaluation
    def _data_angle(self, op, x):
        s = op.src
        if s is None:
            return None
        if s[0] == "x":
            return s[2] * x[:, s[1]]
        if s[0] == "xx":
            return (PI - x[:, s[1]]) * (PI - x[:, s[2]]) / PI   # rescaled ZZ-map interaction
        raise ValueError

    def encode(self, x):
        st = _zero_state(x.shape[0], self.n)
        for op in self.enc_ops:
            st = _apply_op(st, op, self._data_angle(op, x), self.n)
        return st

    def _run_var(self, state, thetas):
        """state (B,...), thetas (S,L,n,2) -> expectation values (S,B,n)."""
        S, B = thetas.shape[0], state.shape[0]
        st = np.broadcast_to(state[None], (S,) + state.shape).reshape((S * B,) + state.shape[1:])
        for op in self.var_ops:
            ang = None
            if op.src is not None:
                _, l, q, k = op.src
                ang = np.repeat(thetas[:, l, q, k], B)
            st = _apply_op(st, op, ang, self.n)
        return _expect_z(st, self.n).reshape(S, B, self.n)

    def expectations(self, x, state=None):
        st = self.encode(x) if state is None else state
        return self._run_var(st, self.theta[None])[0]

    def predict_proba(self, x):
        z = self.expectations(x)
        return softmax(z @ self.W.T + self.b)

    # -- training
    def loss_grad(self, x, y):
        B = x.shape[0]
        st = self.encode(x)
        z = self._run_var(st, self.theta[None])[0]
        p = softmax(z @ self.W.T + self.b)
        loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
        dl = p.copy(); dl[np.arange(B), y] -= 1; dl /= B
        dW, db, dz = dl.T @ z, dl.sum(0), dl @ self.W
        P = self.theta.size
        flat = np.tile(self.theta.reshape(1, -1), (2 * P, 1))
        for k in range(P):                               # parameter-shift rule
            flat[k, k] += PI / 2
            flat[P + k, k] -= PI / 2
        Z = self._run_var(st, flat.reshape((2 * P,) + self.theta.shape))
        J = 0.5 * (Z[:P] - Z[P:])                        # (P,B,n)  d<Z>/d theta_k
        dtheta = np.einsum("pbi,bi->p", J, dz).reshape(self.theta.shape)
        return loss, [dtheta, dW, db]

    # -- animation / inspection helpers
    def trace(self, x1):
        """Run ONE sample column-by-column. Returns list of <Z> vectors after each column."""
        x = np.asarray(x1, dtype=float).reshape(1, -1)
        st = _zero_state(1, self.n)
        out = [np.ones(self.n)]
        for col in self.cols:
            for idx in col:
                op = self.ops[idx]
                if op.src is None:
                    ang = None
                elif op.src[0] == "t":
                    ang = np.array([self.theta[op.src[1], op.src[2], op.src[3]]])
                else:
                    ang = self._data_angle(op, x)
                st = _apply_op(st, op, ang, self.n)
            out.append(_expect_z(st, self.n)[0])
        return out

    def op_angles(self, x1):
        """Concrete angle for every op (None for fixed gates) – used to label the drawing."""
        x = np.asarray(x1, dtype=float).reshape(1, -1)
        res = []
        for op in self.ops:
            if op.src is None:
                res.append(None)
            elif op.src[0] == "t":
                res.append(float(self.theta[op.src[1], op.src[2], op.src[3]]))
            else:
                res.append(float(self._data_angle(op, x)[0]))
        return res
