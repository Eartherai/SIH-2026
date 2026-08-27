"""False-positive engine.

PS 26057 asks specifically for a module that "minimizes false positives caused by
natural acoustic shadows or rock clusters". This is that module.

Method
------
We do NOT hand-write threshold rules and assert they work. Instead:

  1. `features.extract` computes 10 physically-motivated descriptors from the
     pixels around each candidate (shadow coherence, target/background contrast,
     highlight compactness, relative texture roughness, ...). These are
     independent of the detector's own opinion.
  2. A small logistic model is FITTED on a held-out calibration survey, where
     each candidate is labelled true/false by IoU against ground truth.
  3. The learned weights are inspectable and are reported in
     docs/BENCHMARKS.md alongside the measured effect on the test surveys.

If no fitted model is present we fall back to `RuleBasedFilter`, which is
explicitly labelled as a heuristic and which reports its reasons. It is a
fallback, not a claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES

# The detector's own score is appended as an 11th input so the filter can learn
# how much to trust it relative to the physical evidence.
INPUT_NAMES = FEATURE_NAMES + ["raw_score"]


@dataclass
class FilterVerdict:
    accepted: bool
    score: float                 # P(true target) from the filter, 0..1
    reason: str
    detail: dict


class RuleBasedFilter:
    """Transparent fallback used only when no learned filter has been fitted.

    Every rejection carries the rule that fired, so an operator can audit it.
    These thresholds are ENGINEERING DEFAULTS, not validated constants - the
    learned filter should be used whenever calibration data exists.
    """
    kind = "rule_based"
    fitted = False

    def __init__(self, min_shadow: float = 0.03, min_snr: float = 1.5,
                 max_aspect: float = 12.0, min_raw: float = 0.15):
        self.min_shadow, self.min_snr = min_shadow, min_snr
        self.max_aspect, self.min_raw = max_aspect, min_raw

    def predict(self, feats: np.ndarray, raw: np.ndarray) -> list[FilterVerdict]:
        feats = np.atleast_2d(feats)
        raw = np.atleast_1d(raw)
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        out: list[FilterVerdict] = []
        for f, r in zip(feats, raw):
            reasons = []
            if r < self.min_raw:
                reasons.append(f"raw_score {r:.2f} < {self.min_raw}")
            if f[idx["local_snr"]] < self.min_snr:
                reasons.append(f"local_snr {f[idx['local_snr']]:.2f} < {self.min_snr}")
            if f[idx["shadow_ratio"]] < self.min_shadow:
                reasons.append(f"no coherent acoustic shadow "
                               f"({f[idx['shadow_ratio']]:.3f} < {self.min_shadow})")
            if f[idx["aspect_ratio"]] > self.max_aspect:
                reasons.append(f"extreme aspect ratio {f[idx['aspect_ratio']]:.1f} "
                               f"(likely ripple/seam artefact)")
            ok = not reasons
            out.append(FilterVerdict(
                accepted=ok,
                score=float(r if ok else r * 0.5),
                reason="passed heuristic checks" if ok else "; ".join(reasons),
                detail={"filter": self.kind, "rules_fired": reasons},
            ))
        return out

    def as_dict(self) -> dict:
        return {"kind": self.kind, "fitted": False,
                "thresholds": {"min_shadow": self.min_shadow, "min_snr": self.min_snr,
                               "max_aspect": self.max_aspect, "min_raw": self.min_raw},
                "note": "Heuristic fallback. Not validated on this survey."}


class LearnedFPFilter:
    """L2-regularised logistic regression over standardised physical features."""
    kind = "learned_logistic"

    def __init__(self, w: np.ndarray | None = None, b: float = 0.0,
                 mu: np.ndarray | None = None, sd: np.ndarray | None = None,
                 threshold: float = 0.5, fitted: bool = False, meta: dict | None = None):
        n = len(INPUT_NAMES)
        self.w = np.zeros(n) if w is None else np.asarray(w, np.float64)
        self.b = float(b)
        self.mu = np.zeros(n) if mu is None else np.asarray(mu, np.float64)
        self.sd = np.ones(n) if sd is None else np.asarray(sd, np.float64)
        self.threshold = float(threshold)
        self.fitted = bool(fitted)
        self.meta = meta or {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, iters: int = 6000, lr: float = 0.1,
            l2: float = 1e-3) -> "LearnedFPFilter":
        X = np.atleast_2d(np.asarray(X, np.float64))
        y = np.asarray(y, np.float64).ravel()
        if len(y) < 30 or len(np.unique(y)) < 2:
            self.fitted = False
            self.meta = {"error": "insufficient calibration data for a learned filter",
                         "n": int(len(y)), "positives": int(y.sum())}
            return self

        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-6] = 1.0
        Z = (X - self.mu) / self.sd

        n, d = Z.shape
        w, b = np.zeros(d), 0.0
        # class weighting: false positives usually vastly outnumber true targets
        pos_w = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
        sw = np.where(y == 1, pos_w, 1.0)
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            g = (p - y) * sw
            w -= lr * ((Z.T @ g) / n + l2 * w)
            b -= lr * (g.sum() / n)
        self.w, self.b, self.fitted = w, float(b), True
        self.meta = {"n": int(n), "positives": int(y.sum()),
                     "negatives": int((y == 0).sum()), "pos_weight": round(pos_w, 3),
                     "l2": l2, "iters": iters}
        return self

    def proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, np.float64))
        if not self.fitted:
            return np.full(len(X), np.nan)
        Z = (X - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))

    def predict(self, feats: np.ndarray, raw: np.ndarray) -> list[FilterVerdict]:
        feats = np.atleast_2d(feats)
        raw = np.atleast_1d(raw).reshape(-1, 1)
        X = np.hstack([feats, raw])
        p = self.proba(X)
        out = []
        for pi, row in zip(p, X):
            top = self.top_contributions(row, k=3)
            desc = ", ".join(f"{n}{'+' if c > 0 else '-'}{abs(c):.2f}" for n, c in top)
            out.append(FilterVerdict(
                accepted=bool(pi >= self.threshold),
                score=float(pi),
                reason=(f"learned filter p={pi:.3f} "
                        f"{'>=' if pi >= self.threshold else '<'} {self.threshold:.2f}; "
                        f"drivers: {desc}"),
                detail={"filter": self.kind, "p_true": float(pi),
                        "top_contributions": [{"feature": n, "contribution": round(float(c), 4)}
                                              for n, c in top]},
            ))
        return out

    def top_contributions(self, x_row: np.ndarray, k: int = 3):
        """Per-detection contribution of each input to the logit -- explainability."""
        z = (np.asarray(x_row, np.float64) - self.mu) / self.sd
        contrib = z * self.w
        order = np.argsort(-np.abs(contrib))[:k]
        return [(INPUT_NAMES[i], float(contrib[i])) for i in order]

    # ------------------------------------------------------------- persistence
    def as_dict(self) -> dict:
        return {"kind": self.kind, "fitted": self.fitted, "threshold": self.threshold,
                "input_names": INPUT_NAMES,
                "weights": {n: round(float(v), 10) for n, v in zip(INPUT_NAMES, self.w)},
                "bias": round(self.b, 10),
                "standardisation": {"mu": [round(float(v), 10) for v in self.mu],
                                    "sd": [round(float(v), 10) for v in self.sd]},
                **self.meta}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path):
        p = Path(path)
        if not p.exists() or not json.loads(p.read_text()).get("fitted"):
            return RuleBasedFilter()
        d = json.loads(p.read_text())
        return cls(w=np.array([d["weights"][n] for n in d["input_names"]]),
                   b=d["bias"],
                   mu=np.array(d["standardisation"]["mu"]),
                   sd=np.array(d["standardisation"]["sd"]),
                   threshold=d.get("threshold", 0.5), fitted=True,
                   meta={k: v for k, v in d.items()
                         if k not in ("weights", "bias", "standardisation",
                                      "input_names", "kind", "fitted", "threshold")})
