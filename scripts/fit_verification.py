#!/usr/bin/env python3
"""Fit the verification stage (FP filter + confidence calibration).

Fitted STRICTLY on the validation survey, evaluated STRICTLY on the held-out
test surveys. The detector never saw the validation survey during training, and
neither the detector nor the filter ever sees the test surveys before
scripts/evaluate.py is run. That separation is what makes the reported numbers
mean anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aquashield.confidence.calibration import PlattCalibrator, reliability  # noqa: E402
from aquashield.confidence.features import extract  # noqa: E402
from aquashield.confidence.fp_filter import LearnedFPFilter  # noqa: E402
from aquashield.detection.boxes import xywhn_to_xyxy  # noqa: E402
from aquashield.detection.detector import Detector  # noqa: E402
from aquashield.evaluation.matching import match  # noqa: E402
from aquashield.sonar.preprocess import PROFILES, preprocess  # noqa: E402
from aquashield.sonar.qc import assess  # noqa: E402


def load_gt(label_path: Path, w: int, h: int):
    if not label_path.exists():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    rows = [r.split() for r in label_path.read_text().splitlines() if r.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    arr = np.array([[float(v) for v in r] for r in rows], np.float32)
    return xywhn_to_xyxy(arr[:, 1:5], w, h), arr[:, 0].astype(np.int64)


def collect(det: Detector, split_dir: Path, profile: str, iou_thr: float,
            tile: int, overlap: int):
    """Run the detector over a split and label every candidate TP/FP."""
    cfg = PROFILES[profile]
    X, y, raws, meta = [], [], [], []
    imgs = sorted((split_dir / "images").glob("*.jpg"))
    for n, ip in enumerate(imgs):
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        h, w = g.shape
        q = assess(g)
        proc = preprocess(g, cfg).image
        res = det.detect(proc, tile, overlap)
        gt_b, gt_c = load_gt((split_dir / "labels" / ip.name).with_suffix(".txt"), w, h)

        if not res.detections:
            continue
        db = np.array([d.box_xyxy for d in res.detections], np.float32)
        ds = np.array([d.raw_score for d in res.detections], np.float32)
        m = match(db, ds, gt_b, iou_thr=iou_thr)
        is_tp = np.zeros(len(db), np.float64)
        for di, _, _ in m.matched_pairs:
            is_tp[di] = 1.0

        gray01 = proc.astype(np.float32) / 255.0
        nadir = (float(sum(q.water_column_bounds)) / 2.0
                 if q.water_column_detected and q.water_column_bounds else None)
        for k, d in enumerate(res.detections):
            f = extract(gray01, d.box_xyxy, nadir)
            X.append(np.concatenate([f.vector(), [d.raw_score]]))
            y.append(is_tp[k])
            raws.append(d.raw_score)
            meta.append({"frame": ip.stem, "class_id": d.class_id})
        if (n + 1) % 25 == 0:
            print(f"  ...{n+1}/{len(imgs)} frames, {len(X)} candidates")
    return (np.array(X, np.float64), np.array(y, np.float64),
            np.array(raws, np.float64), meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="data/processed/milco_nombo_yolo")
    ap.add_argument("--split", default="val")
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--conf", type=float, default=0.02,
                    help="deliberately very low -- the filter can only learn from "
                         "candidates the detector actually emits")
    ap.add_argument("--min-recall", type=float, default=0.80,
                    help="recall floor for threshold selection. Without it, a thin "
                         "fit split has a degenerate optimum: reject everything, "
                         "score perfect precision, and detect nothing.")
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--tag", default="milco_nombo")
    args = ap.parse_args()

    det = Detector(args.weights, conf=args.conf)
    split_dir = Path(args.data_root) / args.split
    print(f"collecting candidates from {split_dir} (conf>={args.conf}) ...")
    X, y, raws, meta = collect(det, split_dir, args.profile, args.iou_thr,
                               args.tile, args.overlap)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"\ncandidates={len(y)}  true={n_pos}  false={n_neg}")
    if len(y) < 30 or n_pos == 0:
        print("REFUSING to fit: not enough labelled candidates on this split.\n"
              "The pipeline will fall back to the rule-based filter and report "
              "'calibrated: false'. This is the honest outcome, not a failure.")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- FP filter ----
    # Regularise harder when the fit split is small: 11 inputs fitted on a few
    # dozen candidates will otherwise memorise this survey and reject everything
    # it has not seen. Scale L2 with the inverse sample count.
    l2 = 1e-3 if len(y) >= 400 else (1e-2 if len(y) >= 150 else 5e-2)
    filt = LearnedFPFilter().fit(X, y, l2=l2)
    p = filt.proba(X)

    cands = [(t, *_prf(p >= t, y)) for t in np.linspace(0.05, 0.95, 91)]
    # Only consider thresholds that retain enough true targets. Maximising F1
    # alone on a thin split happily picks "accept almost nothing".
    viable = [c for c in cands if c[2] >= args.min_recall]
    if viable:
        best = max(viable, key=lambda z: z[3])
    else:
        best = max(cands, key=lambda z: z[2])       # fall back to best recall
        print(f"WARNING: no threshold reaches recall >= {args.min_recall}; "
              f"selected the highest-recall threshold instead.")
    filt.threshold = float(best[0])
    filt.meta["selected_threshold_f1"] = round(float(best[3]), 4)
    filt.meta["selected_threshold_recall"] = round(float(best[2]), 4)
    filt.meta["min_recall_constraint"] = args.min_recall
    filt.meta["l2"] = l2
    filt.meta["fit_split"] = args.split
    filt.meta["iou_threshold"] = args.iou_thr
    filt.save(out / f"fp_filter_{args.tag}.json")

    print(f"\nFP filter fitted. threshold={filt.threshold:.2f} "
          f"(P={best[1]:.3f} R={best[2]:.3f} F1={best[3]:.3f} on the FIT split)")
    print("top weights:")
    for k, v in sorted(filt.as_dict()["weights"].items(), key=lambda kv: -abs(kv[1]))[:6]:
        print(f"   {k:24s} {v:+.4f}")

    # ---- calibration ----
    cal = PlattCalibrator().fit(raws, y)
    before = reliability(raws, y)
    after = reliability(cal.transform(raws), y)
    cal.meta["ece_before"] = round(before.ece, 4)
    cal.meta["ece_after"] = round(after.ece, 4)
    cal.meta["fit_split"] = args.split
    cal.save(out / f"calibration_{args.tag}.json")
    print(f"\nCalibration fitted (Platt). ECE {before.ece:.4f} -> {after.ece:.4f} "
          f"(on the FIT split)")

    (out / f"verification_fit_{args.tag}.json").write_text(json.dumps({
        "weights": args.weights, "split": args.split, "profile": args.profile,
        "detector_conf": args.conf, "iou_threshold": args.iou_thr,
        "candidates": len(y), "true_positives": n_pos, "false_positives": n_neg,
        "fp_filter": filt.as_dict(), "calibration": cal.as_dict(),
        "reliability_before": before.as_dict(), "reliability_after": after.as_dict(),
    }, indent=2))
    print(f"\nwrote {out}/fp_filter_{args.tag}.json, {out}/calibration_{args.tag}.json")


def _prf(pred, y):
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, (2 * p * r / (p + r) if (p + r) else 0.0)


if __name__ == "__main__":
    main()
