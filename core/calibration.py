"""Calibration metrics and a small learned calibrator (logistic regression, numpy only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURES = ["consistency", "verification", "verbalized"]


# ----------------------------------------------------------------- metrics
def ece(probs, labels, n_bins: int = 5) -> float:
    """Expected Calibration Error: weighted gap between confidence and accuracy per bin."""
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def brier(probs, labels) -> float:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    return float(np.mean((p - y) ** 2))


def auroc(scores, labels) -> float | None:
    """Probability that a random correct answer gets a higher score than a random wrong one."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def reliability_bins(probs, labels, n_bins: int = 5) -> list[dict]:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        if mask.any():
            out.append({"confidence": float(p[mask].mean()), "accuracy": float(y[mask].mean()),
                        "count": int(mask.sum())})
    return out


# -------------------------------------------------------------- calibrator
class LogisticCalibrator:
    """P(correct | signals) = sigmoid(w · [consistency, verification, verbalized] + b).

    Platt-style calibration generalised to several uncertainty signals, trained with
    L2-regularised gradient descent.
    """

    def __init__(self, weights=None, bias: float = 0.0, meta: dict | None = None):
        self.w = np.zeros(len(FEATURES)) if weights is None else np.asarray(weights, float)
        self.b = float(bias)
        self.meta = meta or {}

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y, l2: float = 0.002, lr: float = 0.5, iters: int = 8000):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n = len(y)
        self.w = np.zeros(X.shape[1])
        prior = np.clip(y.mean(), 1e-3, 1 - 1e-3)
        self.b = float(np.log(prior / (1 - prior)))
        for _ in range(iters):
            p = self._sigmoid(X @ self.w + self.b)
            grad_w = X.T @ (p - y) / n + l2 * self.w
            grad_b = float(np.mean(p - y))
            self.w -= lr * grad_w
            self.b -= lr * grad_b
        return self

    def predict(self, X) -> np.ndarray:
        return self._sigmoid(np.asarray(X, float) @ self.w + self.b)

    def predict_one(self, features: dict) -> float:
        return float(self.predict([[features[f] for f in FEATURES]])[0])

    def to_dict(self) -> dict:
        return {"features": FEATURES, "weights": self.w.tolist(), "bias": self.b, "meta": self.meta}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "LogisticCalibrator | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(data["weights"], data["bias"], data.get("meta"))


def cross_val_predict(X, y, k: int = 5, seed: int = 7) -> np.ndarray:
    """Out-of-fold predictions so the reported metrics are not measured on training data."""
    X, y = np.asarray(X, float), np.asarray(y, int)
    n = len(y)
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    # Stratified folds: spread correct and wrong answers evenly across folds.
    k = max(2, min(k, n))
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = rng.permutation(np.where(y == cls)[0])
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    folds = [np.array(f, int) for f in folds if f]
    preds = np.zeros(n)
    for fold in folds:
        train = np.setdiff1d(order, fold)
        if len(set(y[train])) < 2:
            preds[fold] = y[train].mean() if len(train) else 0.5
            continue
        preds[fold] = LogisticCalibrator().fit(X[train], y[train]).predict(X[fold])
    return preds
