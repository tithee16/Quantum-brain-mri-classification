"""Classical baselines: small NumPy MLPs trained with the same optimiser."""
import numpy as np
from .quantum import softmax


class MLP:
    def __init__(self, n_in, hidden, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.n_in, self.hidden, self.C = n_in, hidden, n_classes

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    @property
    def n_params(self):
        return sum(p.size for p in self.params())

    def predict_proba(self, x):
        return softmax(np.tanh(x @ self.W1 + self.b1) @ self.W2 + self.b2)

    def loss_grad(self, x, y):
        B = x.shape[0]
        h = np.tanh(x @ self.W1 + self.b1)
        p = softmax(h @ self.W2 + self.b2)
        loss = -np.log(p[np.arange(B), y] + 1e-12).mean()
        d = p.copy(); d[np.arange(B), y] -= 1; d /= B
        dW2, db2 = h.T @ d, d.sum(0)
        dh = (d @ self.W2.T) * (1 - h ** 2)
        return loss, [x.T @ dh, dh.sum(0), dW2, db2]


def matched_hidden(n_in, n_classes, target_params):
    """Hidden width so the MLP has ~ the same number of parameters as the hybrid model."""
    return max(2, int(round((target_params - n_classes) / (n_in + 1 + n_classes))))
