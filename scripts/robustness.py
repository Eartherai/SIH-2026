#!/usr/bin/env python3
"""Controlled degradation study.

PS 26057 names the conditions that break sonar analysis: speckle noise, varying
resolution, acoustic shadow, and dropouts from vehicle motion. This script
applies each as a CONTROLLED perturbation to the held-out test frames and
measures what actually happens to detection performance.

Perturbations are synthetic, and that is stated. They isolate one variable at a
time, which real data cannot do. They do not replace validation on real degraded
surveys.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aquashield.confidence.calibration import PlattCalibrator  # noqa: E402
from aquashield.confidence.fp_filter import LearnedFPFilter  # noqa: E402
from aquashield.detection.boxes import xywhn_to_xyxy  # noqa: E402
from aquashield.detection.detector import Detector  # noqa: E402
from aquashield.detection.taxonomy import Taxonomy  # noqa: E402
from aquashield.evaluation.matching import aggregate, match  # noqa: E402
from aquashield.pipeline import AquaShieldPipeline, PipelineConfig  # noqa: E402

RNG = np.random.default_rng(0)


# --------------------------------------------------------------- perturbations
def identity(img):
    return img


def speckle(img, sigma):
    """Multiplicative Rayleigh-like speckle -- the correct noise model for
    coherent sonar, unlike additive Gaussian."""
    g = RNG.gamma(shape=1.0 / max(sigma, 1e-3), scale=max(sigma, 1e-3), size=img.shape)
    return np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)


def low_contrast(img, factor):
    m = float(img.mean())
    return np.clip((img.astype(np.float32) - m) * factor + m, 0, 255).astype(np.uint8)


def blur(img, k):
    k = int(k) | 1
    return cv2.GaussianBlur(img, (k, k), 0)


def downscale(img, factor):
    """Resolution degradation: shrink then restore size, as a lower-resolution
    sonar or a faster tow speed would produce."""
    h, w = img.shape
    small = cv2.resize(img, (max(8, int(w * factor)), max(8, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def dropout(img, frac):
    """Dead ping rows in bursts, as heave/pitch/roll or telemetry loss produces."""
    out = img.copy()
    h = img.shape[0]
    n = int(h * frac)
    i = 0
    while i < n:
        burst = int(RNG.integers(2, 9))
        r = int(RNG.integers(0, max(h - burst, 1)))
        out[r:r + burst, :] = 0
        i += burst
    return out


def gain_shift(img, factor):
    """Whole-frame gain change: a different survey's amplifier settings."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


CONDITIONS = [
    ("baseline", identity, [None]),
    ("speckle", speckle, [0.25, 0.5, 1.0]),
    ("low_contrast", low_contrast, [0.7, 0.5, 0.3]),
    ("blur", blur, [3, 5, 9]),
    ("resolution_loss", downscale, [0.75, 0.5, 0.25]),
    ("ping_dropout", dropout, [0.05, 0.15, 0.30]),
    ("gain_shift", gain_shift, [0.6, 1.5]),
]


def load_gt(p: Path, w: int, h: int):
    if not p.exists():
        return np.zeros((0, 4), np.float32)
    rows = [r.split() for r in p.read_text().splitlines() if r.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.array([[float(v) for v in r] for r in rows], np.float32)
    return xywhn_to_xyxy(a[:, 1:5], w, h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="data/processed/milco_nombo_yolo")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--out", default="experiments/robustness.json")
    args = ap.parse_args()

    split = Path(args.data_root) / "test"
    ips = sorted((split / "images").glob("*.jpg"))
    # keep a mix: all target-bearing frames first, then empties, so both recall
    # and false-alarm rate stay measurable under perturbation
    withgt = [p for p in ips
              if (split / "labels" / p.name).with_suffix(".txt").read_text().strip()]
    empty = [p for p in ips if p not in set(withgt)]
    chosen = (withgt[: args.limit // 2] + empty[: args.limit // 2])
    frames = []
    for p in chosen:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            frames.append((p.stem, g,
                           load_gt((split / "labels" / p.name).with_suffix(".txt"),
                                   g.shape[1], g.shape[0])))
    print(f"{len(frames)} frames "
          f"({sum(1 for _,_,g in frames if len(g))} with targets, "
          f"{sum(1 for _,_,g in frames if not len(g))} empty)\n")

    det = Detector(args.weights, conf=args.conf)
    pipe = AquaShieldPipeline(
        det, PipelineConfig(preprocess_profile="standard"),
        fp_filter=LearnedFPFilter.load("models/fp_filter_milco_nombo.json"),
        calibrator=PlattCalibrator.load("models/calibration_milco_nombo.json"),
        taxonomy=Taxonomy("milco_nombo"))

    rows = []
    for name, fn, levels in CONDITIONS:
        for lvl in levels:
            per_frame = []
            t0 = time.perf_counter()
            for fid, img, gt in frames:
                pert = img if lvl is None else fn(img, lvl)
                fr = pipe.process_frame(pert, fid, make_preview=False)
                db = (np.array([d["box_xyxy"] for d in fr.accepted], np.float32)
                      if fr.accepted else np.zeros((0, 4), np.float32))
                ds = (np.array([d["confidence_pct"] for d in fr.accepted], np.float32)
                      if fr.accepted else np.zeros(0, np.float32))
                m = match(db, ds, gt, iou_thr=args.iou_thr)
                per_frame.append({"n_gt": len(gt), "n_det": len(db),
                                  "tp": m.tp, "fp": m.fp, "fn": m.fn})
            agg = aggregate(per_frame, args.iou_thr).as_dict()
            agg.update(condition=name, level=lvl,
                       seconds=round(time.perf_counter() - t0, 1))
            rows.append(agg)
            lab = f"{name}" + (f"={lvl}" if lvl is not None else "")
            print(f"  {lab:26s} P={agg['precision']:.4f} R={agg['recall']:.4f} "
                  f"F1={agg['f1']:.4f}  FA-frames={agg['false_alarm_frames']}/"
                  f"{agg['n_frames_empty']}")

    base = rows[0]
    for r in rows:
        r["recall_retained_vs_baseline"] = (
            round(r["recall"] / base["recall"], 3) if base["recall"] else None)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weights": args.weights, "device": det.device,
        "match_iou_threshold": args.iou_thr,
        "note": "Perturbations are SYNTHETIC and isolate one variable at a time. "
                "They do not replace validation on real degraded surveys.",
        "frames": len(frames), "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
