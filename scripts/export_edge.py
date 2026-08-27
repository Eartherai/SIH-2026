#!/usr/bin/env python3
"""Export the detector for edge deployment and MEASURE the result.

Produces ONNX (portable, runs under ONNX Runtime on ARM, x86 and Jetson) and
optionally CoreML (Apple Neural Engine). Every claim printed here is measured on
this machine -- we do not assert Jetson performance we have not tested.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def bench_onnx(path: str, imgsz: int, n: int = 30) -> dict | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed; pip install onnxruntime"}
    providers = ort.get_available_providers()
    sess = ort.InferenceSession(path, providers=providers)
    name = sess.get_inputs()[0].name
    shape = sess.get_inputs()[0].shape
    h = imgsz if not isinstance(shape[2], int) else shape[2]
    w = imgsz if not isinstance(shape[3], int) else shape[3]
    x = np.random.rand(1, 3, h, w).astype(np.float32)
    for _ in range(5):
        sess.run(None, {name: x})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t) * 1000)
    a = np.array(ts)
    return {"providers": providers, "input_shape": [1, 3, h, w],
            "mean_ms": round(float(a.mean()), 2),
            "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--formats", default="onnx")
    ap.add_argument("--out", default="experiments/edge_export.json")
    args = ap.parse_args()

    from ultralytics import YOLO
    src = Path(args.weights)
    result = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "host": {"platform": platform.platform(), "machine": platform.machine()},
              "source_weights": str(src),
              "source_size_mb": round(src.stat().st_size / 1e6, 2),
              "imgsz": args.imgsz, "exports": {}}
    print(f"source: {src} ({result['source_size_mb']} MB)\n")

    for fmt in [f.strip() for f in args.formats.split(",") if f.strip()]:
        print(f"exporting {fmt} ...")
        try:
            m = YOLO(str(src))
            t0 = time.perf_counter()
            out = m.export(format=fmt, imgsz=args.imgsz, simplify=True, device="cpu")
            secs = time.perf_counter() - t0
            p = Path(out)
            size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    if p.is_dir() else p.stat().st_size)
            entry = {"path": str(p), "size_mb": round(size / 1e6, 2),
                     "export_seconds": round(secs, 1)}
            if fmt == "onnx":
                entry["benchmark"] = bench_onnx(str(p), args.imgsz)
            result["exports"][fmt] = entry
            print(f"  -> {p.name}  {entry['size_mb']} MB  ({secs:.1f}s)")
            if entry.get("benchmark", {}).get("mean_ms"):
                b = entry["benchmark"]
                print(f"     ONNX Runtime: {b['mean_ms']} ms mean "
                      f"(p95 {b['p95_ms']}) providers={b['providers']}")
            elif entry.get("benchmark", {}).get("error"):
                print(f"     benchmark skipped: {entry['benchmark']['error']}")
        except Exception as e:                                     # noqa: BLE001
            result["exports"][fmt] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  FAILED: {type(e).__name__}: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    print("\nNOTE: these are measurements on THIS machine only. No Jetson, AUV or "
          "other embedded target has been tested; see docs/LIMITATIONS.md.")


if __name__ == "__main__":
    main()
