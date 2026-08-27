#!/usr/bin/env python3
"""Measure AQUA-SHIELD on the held-out test surveys and run the ablation.

Nothing in this file invents a number. Every metric is computed from real
detections on real frames the model has never seen. Where a configuration
cannot be evaluated, the row says so instead of being filled in.
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
from aquashield.confidence.calibration import IdentityCalibrator, PlattCalibrator  # noqa: E402
from aquashield.confidence.fp_filter import LearnedFPFilter, RuleBasedFilter  # noqa: E402
from aquashield.detection.boxes import xywhn_to_xyxy  # noqa: E402
from aquashield.detection.detector import Detector  # noqa: E402
from aquashield.detection.taxonomy import Taxonomy  # noqa: E402
from aquashield.evaluation.matching import aggregate, match  # noqa: E402
from aquashield.pipeline import AquaShieldPipeline, PipelineConfig  # noqa: E402
from aquashield.sonar.preprocess import PROFILES, PreprocessConfig  # noqa: E402


def load_gt(p: Path, w: int, h: int):
    if not p.exists():
        return np.zeros((0, 4), np.float32)
    rows = [r.split() for r in p.read_text().splitlines() if r.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    arr = np.array([[float(v) for v in r] for r in rows], np.float32)
    return xywhn_to_xyxy(arr[:, 1:5], w, h)


def run_variant(name: str, det: Detector, frames, gts, *, pp_cfg: PreprocessConfig,
                fp_filter, calibrator, iou_thr: float, tax: Taxonomy,
                use_fp: bool, use_tiling: bool) -> dict:
    cfg = PipelineConfig(preprocess_config=pp_cfg, use_fp_filter=use_fp,
                         use_calibration=calibrator is not None,
                         tile_size=(640 if use_tiling else 100_000),
                         tile_overlap=(128 if use_tiling else 0))
    pipe = AquaShieldPipeline(det, cfg, fp_filter=fp_filter,
                              calibrator=calibrator or IdentityCalibrator(), taxonomy=tax)

    per_frame, lat = [], []
    for (fid, img), gt in zip(frames, gts):
        t0 = time.perf_counter()
        fr = pipe.process_frame(img, fid, make_preview=False)
        lat.append((time.perf_counter() - t0) * 1000)
        dets = fr.accepted
        db = (np.array([d["box_xyxy"] for d in dets], np.float32)
              if dets else np.zeros((0, 4), np.float32))
        ds = (np.array([d["confidence_pct"] for d in dets], np.float32)
              if dets else np.zeros(0, np.float32))
        m = match(db, ds, gt, iou_thr=iou_thr)
        per_frame.append({"n_gt": len(gt), "n_det": len(db),
                          "tp": m.tp, "fp": m.fp, "fn": m.fn})

    agg = aggregate(per_frame, iou_thr).as_dict()
    agg["variant"] = name
    agg["latency_ms_mean"] = round(float(np.mean(lat)), 1)
    agg["latency_ms_p50"] = round(float(np.percentile(lat, 50)), 1)
    agg["latency_ms_p95"] = round(float(np.percentile(lat, 95)), 1)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="data/processed/milco_nombo_yolo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou-thr", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--fp-filter", default="models/fp_filter_milco_nombo.json")
    ap.add_argument("--calibration", default="models/calibration_milco_nombo.json")
    ap.add_argument("--out", default="experiments/ablation.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    split_dir = Path(args.data_root) / args.split
    ips = sorted((split_dir / "images").glob("*.jpg"))
    if args.limit:
        ips = ips[: args.limit]
    frames, gts = [], []
    for ip in ips:
        g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        frames.append((ip.stem, g))
        gts.append(load_gt((split_dir / "labels" / ip.name).with_suffix(".txt"),
                           g.shape[1], g.shape[0]))
    print(f"evaluating {len(frames)} frames from {split_dir}")
    print(f"  frames with targets : {sum(1 for g in gts if len(g))}")
    print(f"  empty frames        : {sum(1 for g in gts if not len(g))}")
    print(f"  ground-truth objects: {sum(len(g) for g in gts)}\n")

    det = Detector(args.weights, conf=args.conf)
    tax = Taxonomy("milco_nombo")
    learned = LearnedFPFilter.load(args.fp_filter)
    cal = PlattCalibrator.load(args.calibration)
    has_learned = getattr(learned, "fitted", False)
    has_cal = getattr(cal, "fitted", False)
    print(f"learned FP filter: {'LOADED' if has_learned else 'NOT AVAILABLE (rule-based fallback)'}")
    print(f"calibration      : {'LOADED' if has_cal else 'NOT AVAILABLE (raw scores)'}\n")

    none_pp = PROFILES["none"]
    std_pp = PROFILES["standard"]
    wc_pp = PreprocessConfig(water_column_removal=True, water_column_mode="inpaint")

    variants = [
        ("A_detector_only_no_preprocess", dict(pp_cfg=none_pp, fp_filter=None,
                                               calibrator=None, use_fp=False, use_tiling=True)),
        ("B_plus_preprocessing", dict(pp_cfg=std_pp, fp_filter=None, calibrator=None,
                                      use_fp=False, use_tiling=True)),
        ("C_plus_water_column_removal", dict(pp_cfg=wc_pp, fp_filter=None, calibrator=None,
                                             use_fp=False, use_tiling=True)),
        ("D_plus_rule_based_fp_filter", dict(pp_cfg=wc_pp, fp_filter=RuleBasedFilter(),
                                             calibrator=None, use_fp=True, use_tiling=True)),
    ]
    if has_learned:
        variants.append(("E_plus_learned_fp_filter",
                         dict(pp_cfg=wc_pp, fp_filter=learned, calibrator=None,
                              use_fp=True, use_tiling=True)))
    if has_learned and has_cal:
        variants.append(("F_full_pipeline_calibrated",
                         dict(pp_cfg=wc_pp, fp_filter=learned, calibrator=cal,
                              use_fp=True, use_tiling=True)))

    rows = []
    for name, kw in variants:
        print(f"-> {name}")
        r = run_variant(name, det, frames, gts, iou_thr=args.iou_thr, tax=tax, **kw)
        rows.append(r)
        print(f"   P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
              f"| FP={r['fp']:5d} | false-alarm frames={r['false_alarm_frames']}/"
              f"{r['n_frames_empty']} ({r['false_alarm_frame_rate']:.1%}) "
              f"| {r['latency_ms_mean']:.0f} ms/frame")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag, "weights": args.weights, "split": args.split,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detector_conf": args.conf, "match_iou_threshold": args.iou_thr,
        "device": det.device,
        "learned_fp_filter_available": has_learned,
        "calibration_available": has_cal,
        "dataset": {"frames": len(frames),
                    "frames_with_targets": sum(1 for g in gts if len(g)),
                    "empty_frames": sum(1 for g in gts if not len(g)),
                    "gt_objects": sum(len(g) for g in gts)},
        "variants": rows,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")

    print("\n" + "=" * 108)
    print(f"{'variant':34s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>5s} {'FP':>6s} "
          f"{'FN':>5s} {'FA-frames':>10s} {'ms/frame':>9s}")
    print("-" * 108)
    for r in rows:
        print(f"{r['variant']:34s} {r['precision']:7.4f} {r['recall']:7.4f} {r['f1']:7.4f} "
              f"{r['tp']:5d} {r['fp']:6d} {r['fn']:5d} "
              f"{r['false_alarm_frames']:4d}/{r['n_frames_empty']:<5d} "
              f"{r['latency_ms_mean']:9.1f}")
    print("=" * 108)


if __name__ == "__main__":
    main()
