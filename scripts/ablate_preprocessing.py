#!/usr/bin/env python3
"""Does sonar preprocessing actually help? A matched 2x2 experiment.

PS 26057 section on preprocessing says: do not apply an operation because it
sounds appropriate -- measure its effect. Our first attempt measured it WRONGLY:
we applied preprocessing at inference to a detector trained on raw frames, saw
precision collapse, and nearly concluded "preprocessing hurts". The real cause
was a train/inference distribution mismatch.

This script runs the full 2x2 so the confound is visible rather than hidden:

                        inference on RAW      inference on PREPROCESSED
    trained on RAW         matched                  MISMATCHED
    trained on PREPROC     MISMATCHED               matched

Only the two matched cells answer "does preprocessing help?". The two mismatched
cells quantify the cost of getting it wrong.
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
from aquashield.detection.boxes import xywhn_to_xyxy  # noqa: E402
from aquashield.detection.detector import Detector  # noqa: E402
from aquashield.evaluation.matching import aggregate, match  # noqa: E402


def load_gt(p: Path, w: int, h: int):
    if not p.exists() or not p.read_text().strip():
        return np.zeros((0, 4), np.float32)
    a = np.array([[float(v) for v in r.split()]
                  for r in p.read_text().splitlines() if r.strip()], np.float32)
    return xywhn_to_xyxy(a[:, 1:5], w, h)


def load_split(root: Path, limit: int = 0):
    d = root / "test"
    ips = sorted((d / "images").glob("*.jpg"))
    if limit:
        ips = ips[:limit]
    frames = []
    for ip in ips:
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        frames.append((ip.name, g,
                       load_gt((d / "labels" / ip.name).with_suffix(".txt"),
                               g.shape[1], g.shape[0])))
    return frames


def run(det: Detector, frames, iou_thr: float, tile: int = 640, overlap: int = 128):
    per_frame, lat = [], []
    for _, img, gt in frames:
        t0 = time.perf_counter()
        r = det.detect(img, tile, overlap)
        lat.append((time.perf_counter() - t0) * 1000)
        db = (np.array([d.box_xyxy for d in r.detections], np.float32)
              if r.detections else np.zeros((0, 4), np.float32))
        ds = (np.array([d.raw_score for d in r.detections], np.float32)
              if r.detections else np.zeros(0, np.float32))
        m = match(db, ds, gt, iou_thr=iou_thr)
        per_frame.append({"n_gt": len(gt), "n_det": len(db),
                          "tp": m.tp, "fp": m.fp, "fn": m.fn})
    a = aggregate(per_frame, iou_thr).as_dict()
    a["latency_ms_mean"] = round(float(np.mean(lat)), 1)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-weights", required=True, help="detector trained on RAW frames")
    ap.add_argument("--pp-weights", required=True, help="detector trained on PREPROCESSED frames")
    ap.add_argument("--raw-data", default="data/processed/milco_nombo_yolo")
    ap.add_argument("--pp-data", default="data/processed/milco_nombo_yolo_pp")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="experiments/preprocessing_ablation.json")
    args = ap.parse_args()

    raw_frames = load_split(Path(args.raw_data), args.limit)
    pp_frames = load_split(Path(args.pp_data), args.limit)
    assert len(raw_frames) == len(pp_frames), "raw and preprocessed splits differ in size"
    print(f"{len(raw_frames)} test frames "
          f"({sum(1 for _,_,g in raw_frames if len(g))} with targets, "
          f"{sum(1 for _,_,g in raw_frames if not len(g))} empty)\n")

    det_raw = Detector(args.raw_weights, conf=args.conf)
    det_pp = Detector(args.pp_weights, conf=args.conf)

    cells = [
        ("raw", "raw", det_raw, raw_frames, True),
        ("raw", "preprocessed", det_raw, pp_frames, False),
        ("preprocessed", "preprocessed", det_pp, pp_frames, True),
        ("preprocessed", "raw", det_pp, raw_frames, False),
    ]
    rows = []
    for trained, infer, det, frames, matched in cells:
        r = run(det, frames, args.iou_thr)
        r.update(trained_on=trained, inference_on=infer, matched=matched)
        rows.append(r)
        tag = "matched   " if matched else "MISMATCHED"
        print(f"  {tag} train={trained:12s} infer={infer:12s} "
              f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
              f"FP={r['fp']:4d} FA={r['false_alarm_frames']}/{r['n_frames_empty']}")

    m_raw = next(r for r in rows if r["trained_on"] == "raw" and r["matched"])
    m_pp = next(r for r in rows if r["trained_on"] == "preprocessed" and r["matched"])
    mm = next(r for r in rows if r["trained_on"] == "raw" and not r["matched"])

    better = "preprocessing" if m_pp["f1"] > m_raw["f1"] else "raw imagery"
    conclusion = (
        f"Comparing only the MATCHED cells, training and inferring on "
        f"{better} gives the better F1 "
        f"(preprocessed {m_pp['f1']:.4f} vs raw {m_raw['f1']:.4f}); "
        f"false-alarm frames {m_pp['false_alarm_frames']} vs "
        f"{m_raw['false_alarm_frames']} of {m_raw['n_frames_empty']} empty frames. "
        f"The MISMATCHED cell (raw-trained detector, preprocessed input) scores "
        f"F1 {mm['f1']:.4f} — the cost of applying a preprocessing chain the "
        f"detector was never trained on."
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_weights": args.raw_weights, "pp_weights": args.pp_weights,
        "detector_conf": args.conf, "match_iou_threshold": args.iou_thr,
        "device": det_raw.device,
        "preprocess_profile": json.loads(
            (Path(args.pp_data) / "preprocess_config.json").read_text())["config"],
        "rows": rows, "conclusion": conclusion,
        "caveat": "The two detectors are separately trained, so this comparison also "
                  "carries ordinary run-to-run training variance. With 191 test "
                  "objects, small differences are not significant.",
    }, indent=2))
    print(f"\n{conclusion}\n\nwrote {out}")


if __name__ == "__main__":
    main()
