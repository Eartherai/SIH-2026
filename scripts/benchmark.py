#!/usr/bin/env python3
"""Measure real latency, throughput and memory on this machine.

Every number this prints is measured here and now. Nothing is copied from a
datasheet or a paper. Results are appended to experiments/benchmarks.jsonl.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aquashield.detection.detector import Detector          # noqa: E402
from aquashield.detection.model_meta import preprocess_profile_for_model  # noqa: E402
from aquashield.detection.taxonomy import Taxonomy          # noqa: E402
from aquashield.device import select_device                 # noqa: E402
from aquashield.pipeline import AquaShieldPipeline, PipelineConfig  # noqa: E402
from aquashield.sonar.preprocess import PROFILES, preprocess  # noqa: E402
from aquashield.sonar.qc import assess                      # noqa: E402


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def timeit(fn, n: int, warmup: int = 3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    a = np.array(ts)
    return {"mean_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "min_ms": round(float(a.min()), 2), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default="data/processed/milco_nombo_yolo/test/images")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--devices", default="mps,cpu")
    ap.add_argument("--out", default="experiments/benchmarks.jsonl")
    args = ap.parse_args()

    paths = sorted(Path(args.images).glob("*.jpg"))[: args.n]
    if not paths:
        print(f"no images under {args.images}")
        return
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    imgs = [i for i in imgs if i is not None]
    shapes = {}
    for i in imgs:
        shapes[i.shape] = shapes.get(i.shape, 0) + 1

    info = select_device("auto")
    model_bytes = Path(args.weights).stat().st_size
    result = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "python": platform.python_version()},
        "device_probe": info.as_dict(),
        "weights": args.weights,
        "model_size_mb": round(model_bytes / 1e6, 2),
        "n_images": len(imgs),
        "preprocess_profile": preprocess_profile_for_model(args.weights),
        "image_shapes": {str(k): v for k, v in shapes.items()},
        "stages": {}, "devices": {},
    }

    print(f"AQUA-SHIELD benchmark  —  {platform.platform()} / {platform.machine()}")
    print(f"model: {args.weights} ({result['model_size_mb']} MB), {len(imgs)} frames\n")

    # ---- CPU-side stages (device independent) ----
    ref = imgs[0]
    print("Per-frame CPU stages")
    for name, fn in [("quality_control", lambda: assess(ref)),
                     ("preprocess_standard", lambda: preprocess(ref, PROFILES["standard"])),
                     ("preprocess_aggressive", lambda: preprocess(ref, PROFILES["aggressive"]))]:
        r = timeit(fn, 20)
        result["stages"][name] = r
        print(f"  {name:24s} {r['mean_ms']:7.2f} ms  (p95 {r['p95_ms']:.2f})")

    # ---- per-device inference and end-to-end ----
    for dev in [d.strip() for d in args.devices.split(",") if d.strip()]:
        probe = select_device(dev)
        if probe.device != dev:
            print(f"\n[{dev}] unavailable — {probe.reason}; skipping")
            result["devices"][dev] = {"available": False, "reason": probe.reason}
            continue
        print(f"\n[{dev}]")
        det = Detector(args.weights, device=dev, conf=0.10)
        prof = preprocess_profile_for_model(args.weights)
        pipe = AquaShieldPipeline(det, PipelineConfig(preprocess_profile=prof,
                                                      preprocess_config=PROFILES.get(prof)),
                                  taxonomy=Taxonomy("milco_nombo"))

        pure = timeit(lambda: det.detect(ref, 640, 128), 20)
        e2e = timeit(lambda: pipe.process_frame(ref, "bench", make_preview=False), 15)

        t0 = time.perf_counter()
        pipe.process_survey([(f"f{i}", im) for i, im in enumerate(imgs)],
                            survey_id="BENCH", make_previews=False)
        survey_s = time.perf_counter() - t0

        result["devices"][dev] = {
            "available": True,
            "inference_only": pure,
            "end_to_end_per_frame": e2e,
            "survey": {"frames": len(imgs), "seconds": round(survey_s, 2),
                       "frames_per_second": round(len(imgs) / survey_s, 2),
                       "ms_per_frame": round(1000 * survey_s / len(imgs), 1)},
            "peak_rss_mb": round(peak_rss_mb(), 1),
        }
        d = result["devices"][dev]
        print(f"  inference only      {pure['mean_ms']:7.2f} ms  (p95 {pure['p95_ms']:.2f})")
        print(f"  full frame pipeline {e2e['mean_ms']:7.2f} ms  (p95 {e2e['p95_ms']:.2f})")
        print(f"  survey throughput   {d['survey']['frames_per_second']:7.2f} frames/s "
              f"({d['survey']['ms_per_frame']:.0f} ms/frame)")
        print(f"  peak RSS            {d['peak_rss_mb']:7.1f} MB")

    if "mps" in result["devices"] and "cpu" in result["devices"] \
            and result["devices"]["mps"].get("available") and result["devices"]["cpu"].get("available"):
        sp = (result["devices"]["cpu"]["inference_only"]["mean_ms"]
              / result["devices"]["mps"]["inference_only"]["mean_ms"])
        result["mps_speedup_vs_cpu"] = round(sp, 2)
        print(f"\nMPS speedup over CPU (inference only): {sp:.2f}x")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
