#!/usr/bin/env python3
"""Regenerate the documentation figures from real inference.

Nothing here is drawn by hand or mocked. Every box comes from running the
detector and the verification stage on held-out test frames.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aquashield.confidence.calibration import PlattCalibrator  # noqa: E402
from aquashield.confidence.fp_filter import LearnedFPFilter  # noqa: E402
from aquashield.detection.boxes import xywhn_to_xyxy  # noqa: E402
from aquashield.detection.detector import Detector  # noqa: E402
from aquashield.detection.model_meta import preprocess_profile_for_model  # noqa: E402
from aquashield.detection.taxonomy import Taxonomy  # noqa: E402
from aquashield.pipeline import AquaShieldPipeline, PipelineConfig  # noqa: E402

# Chosen because they exercise both directions of the filter: clutter removed on
# empty seabed, and a true positive retained on a target frame.
FRAMES = [("0504_2018.jpg", "NATURAL SEABED - no target present"),
          ("0162_2018.jpg", "NATURAL SEABED - no target present"),
          ("0444_2018.jpg", "TARGET PRESENT")]


def panel(img, title, sub, items, gt):
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in gt.astype(int):
        cv2.rectangle(bgr, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (255, 255, 255), 2)
        cv2.putText(bgr, "GT", (x0 - 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    for b, c, l in items:
        x0, y0, x1, y1 = (int(v) for v in b)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), c, 2)
        if l:
            cv2.putText(bgr, l, (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, 1, cv2.LINE_AA)
    bgr = cv2.resize(bgr, (440, 440), interpolation=cv2.INTER_AREA)
    bar = np.full((44, 440, 3), 30, np.uint8)
    cv2.putText(bar, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 190, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, bgr])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/aquashield_primary.pt")
    ap.add_argument("--data-root", default="data/processed/milco_nombo_yolo")
    ap.add_argument("--conf", type=float, default=0.03)
    ap.add_argument("--out", default="docs/images/verification_effect.png")
    args = ap.parse_args()

    prof = preprocess_profile_for_model(args.weights)
    det = Detector(args.weights, conf=args.conf)
    tax = Taxonomy("milco_nombo")
    cal = PlattCalibrator.load("models/calibration_milco_nombo.json")
    off = AquaShieldPipeline(det, PipelineConfig(preprocess_profile=prof, use_fp_filter=False),
                             calibrator=cal, taxonomy=tax)
    on = AquaShieldPipeline(det, PipelineConfig(preprocess_profile=prof, use_fp_filter=True),
                            fp_filter=LearnedFPFilter.load("models/fp_filter_milco_nombo.json"),
                            calibrator=cal, taxonomy=tax)

    S = Path(args.data_root) / "test"
    rows = []
    for name, tag in FRAMES:
        g = cv2.imread(str(S / "images" / name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"skipping {name}: not found")
            continue
        lp = (S / "labels" / name).with_suffix(".txt")
        rws = [r.split() for r in lp.read_text().splitlines() if r.strip()] if lp.exists() else []
        gt = (xywhn_to_xyxy(np.array([[float(v) for v in r] for r in rws], np.float32)[:, 1:5],
                            g.shape[1], g.shape[0]) if rws else np.zeros((0, 4), np.float32))
        a = off.process_frame(g, "x", make_preview=False)
        b = on.process_frame(g, "x", make_preview=False)
        print(f"  {name}: detector-only={len(a.accepted)} -> kept={len(b.accepted)} "
              f"rejected={len(b.rejected)}")
        L = panel(g, f"{tag}  |  DETECTOR ONLY", f"{len(a.accepted)} alarms raised",
                  [(d["box_xyxy"], (70, 90, 240), f"{d['confidence_pct']:.0f}%")
                   for d in a.accepted], gt)
        R = panel(g, f"{tag}  |  AFTER VERIFICATION",
                  f"{len(b.accepted)} kept, {len(b.rejected)} rejected as clutter",
                  [(d["box_xyxy"], (90, 90, 120), "") for d in b.rejected]
                  + [(d["box_xyxy"], (0, 200, 255), f"{d['confidence_pct']:.0f}%")
                     for d in b.accepted], gt)
        rows.append(np.hstack([L, np.full((L.shape[0], 8, 3), 35, np.uint8), R]))

    if not rows:
        print("no frames rendered")
        return
    sep = np.full((8, rows[0].shape[1], 3), 35, np.uint8)
    stacked = [x for r in rows for x in (r, sep)][:-1]
    out = np.vstack(stacked)
    cap = np.full((64, out.shape[1], 3), 18, np.uint8)
    cv2.putText(cap, "AQUA-SHIELD  -  the false-positive engine on held-out surveys "
                     "the model has never seen", (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(cap, "white = ground truth      blue = detector alarm      "
                     "orange = kept after verification      grey = rejected as clutter",
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (155, 155, 155), 1, cv2.LINE_AA)
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.vstack([out, cap]))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
