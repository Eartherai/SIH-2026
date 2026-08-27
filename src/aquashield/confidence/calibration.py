"""Confidence calibration.

A detector's raw output is NOT a probability. On a survey dominated by natural
seabed, a raw score of 0.8 does not mean "80% of such detections are real".
AQUA-SHIELD therefore fits an explicit calibration map on a HELD-OUT survey and
reports calibrated numbers separately from raw scores, so an operator is never
shown a fabricated probability.

If no calibration has been fitted, `IdentityCalibrator` is used and every report
is stamped `calibrated: false`. We never silently pass a raw score off as a
probability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ReliabilityCurve:
    bin_edges: list[float]
    bin_confidence: list[float]   # mean predicted score in bin
    bin_accuracy: list[float]     # observed fraction of true positives in bin
    bin_count: list[int]
    ece: float                    # expected calibration error
    mce: float                    # maximum calibration error

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def reliability(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> ReliabilityCurve:
    """Measure calibration error. labels are 1 for true positive, 0 for false positive."""
    scores = np.asarray(scores, np.float64).ravel()
    labels = np.asarray(labels, np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    ece = 0.0
    mce = 0.0
    n = max(len(scores), 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (scores > lo) & (scores <= hi) if i > 0 else (scores >= lo) & (scores <= hi)
        c = int(m.sum())
        cnt.append(c)
        if c == 0:
            conf.append(float((lo + hi) / 2))
            acc.append(float("nan"))
            continue
        cm, am = float(scores[m].mean()), float(labels[m].mean())
        conf.append(cm)
        acc.append(am)
        gap = abs(am - cm)
        ece += (c / n) * gap
        mce = max(mce, gap)
    return ReliabilityCurve([float(e) for e in edges], conf, acc, cnt,
                            float(ece), float(mce))


class IdentityCalibrator:
    """Passes scores through unchanged and declares itself uncalibrated."""
    kind = "identity"
    fitted = False

    def transform(self, s: np.ndarray) -> np.ndarray:
        return np.asarray(s, np.float64)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "fitted": False,
                "note": "No calibration fitted. Scores are RAW detector outputs, "
                        "not probabilities."}


class PlattCalibrator:
    """Logistic (Platt) scaling: p = sigmoid(a * logit(s) + b).

    Fitted with plain gradient descent on the held-out calibration split, so the
    only dependency is numpy and the fit is fully inspectable.
    """
    kind = "platt"

    def __init__(self, a: float = 1.0, b: float = 0.0, fitted: bool = False,
                 meta: dict | None = None):
        self.a, self.b, self.fitted = float(a), float(b), bool(fitted)
        self.meta = meta or {}

    @staticmethod
    def _logit(s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, np.float64), 1e-6, 1 - 1e-6)
        return np.log(s / (1 - s))

    def fit(self, scores: np.ndarray, labels: np.ndarray, iters: int = 4000,
            lr: float = 0.05) -> "PlattCalibrator":
        x = self._logit(scores)
        y = np.asarray(labels, np.float64).ravel()
        if len(y) < 10 or len(np.unique(y)) < 2:
            # Refuse to "fit" a calibration that has no information in it.
            self.fitted = False
            self.meta = {"error": "insufficient or single-class calibration data",
                         "n": int(len(y))}
            return self
        a, b = 1.0, 0.0
        n = len(y)
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(a * x + b)))
            g = p - y
            a -= lr * float((g * x).sum()) / n
            b -= lr * float(g.sum()) / n
        self.a, self.b, self.fitted = float(a), float(b), True
        self.meta = {"n": int(n), "positives": int(y.sum())}
        return self

    def transform(self, s: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.asarray(s, np.float64)
        x = self._logit(s)
        return 1.0 / (1.0 + np.exp(-(self.a * x + self.b)))

    def as_dict(self) -> dict:
        return {"kind": self.kind, "fitted": self.fitted, "a": self.a, "b": self.b,
                **self.meta}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path):
        p = Path(path)
        if not p.exists():
            return IdentityCalibrator()
        d = json.loads(p.read_text())
        if not d.get("fitted"):
            return IdentityCalibrator()
        return cls(d["a"], d["b"], True,
                   {k: v for k, v in d.items() if k not in ("a", "b", "fitted", "kind")})


def band(conf_pct: float) -> str:
    """Map a 0-100 confidence onto the operator-facing band.

    Thresholds are an AQUA-SHIELD product convention, not a marine standard.
    """
    if conf_pct >= 85:
        return "CRITICAL"
    if conf_pct >= 65:
        return "HIGH"
    if conf_pct >= 40:
        return "MEDIUM"
    return "LOW"
