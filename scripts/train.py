#!/usr/bin/env python3
"""Train an AQUA-SHIELD detector and record the run in the experiment registry.

Every run appends one row to experiments/registry.jsonl. Nothing is written to
that registry unless the run actually completed and metrics were measured.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aquashield.detection.model_meta import write_meta  # noqa: E402
from aquashield.device import select_device  # noqa: E402

REGISTRY = Path("experiments/registry.jsonl")


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "uncommitted"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True, help="stable experiment identifier, e.g. E01-baseline")
    ap.add_argument("--data", default="data/processed/milco_nombo_yolo/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--notes", default="")
    # augmentation knobs kept explicit so ablations are auditable
    ap.add_argument("--degrees", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--scale", type=float, default=0.5)
    # Numerical stability controls. Mixed precision on the MPS backend was
    # observed to diverge on this dataset (val cls-loss -> 1e6), so AMP is OFF
    # by default on Apple Silicon. See docs/BENCHMARKS.md "Training stability".
    ap.add_argument("--amp", action="store_true", help="enable mixed precision (unstable on MPS)")
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--cos-lr", action="store_true")
    # Domain-aware augmentation policy for side-scan sonar:
    #   * NO rotation. The across-track axis is range and the along-track axis is
    #     time; rotating a waterfall produces an image that no sonar can make,
    #     and destroys the range-dependent shadow geometry the model must learn.
    #   * BOTH flips are valid: fliplr swaps port/starboard, flipud reverses the
    #     survey heading. Both are physically realisable.
    #   * Hue/saturation are meaningless on single-channel acoustic data; only
    #     value (brightness) jitter is physical, standing in for gain changes.
    ap.add_argument("--hsv-h", type=float, default=0.0)
    ap.add_argument("--hsv-s", type=float, default=0.0)
    ap.add_argument("--hsv-v", type=float, default=0.4)
    args = ap.parse_args()

    dev = select_device(args.device)
    from ultralytics import YOLO

    print(f"[{args.exp_id}] device={dev.device} ({dev.reason})")
    model = YOLO(args.model)

    t0 = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=dev.device,
        seed=args.seed,
        project="runs/train",
        name=args.exp_id,
        exist_ok=True,
        deterministic=True,
        degrees=args.degrees,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        scale=args.scale,
        plots=True,
        val=True,
        amp=args.amp,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
    )
    train_seconds = time.time() - t0
    # Ultralytics may prepend its configured runs_dir, so record the ACTUAL path.
    save_dir = Path(getattr(model.trainer, "save_dir", f"runs/train/{args.exp_id}"))

    # Evaluate on the HELD-OUT TEST surveys, not on val.
    metrics = model.val(data=args.data, split="test", imgsz=args.imgsz,
                        device=dev.device, project="runs/val", name=args.exp_id,
                        exist_ok=True, plots=True)

    box = metrics.box
    row = {
        "experiment_id": args.exp_id,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_rev(),
        "dataset": args.data,
        "split_eval": "test (held-out surveys 2018+2021)",
        "model": args.model,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "amp": args.amp,
        "lr0": args.lr0,
        "cos_lr": args.cos_lr,
        "augment": {"degrees": args.degrees, "fliplr": args.fliplr,
                    "flipud": args.flipud, "mosaic": args.mosaic, "scale": args.scale,
                    "hsv_h": args.hsv_h, "hsv_s": args.hsv_s, "hsv_v": args.hsv_v},
        "hardware": {"device": dev.device, "machine": platform.machine(),
                     "platform": platform.platform()},
        "train_seconds": round(train_seconds, 1),
        "metrics_test": {
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "per_class_mAP50": {n: round(float(v), 4)
                                for n, v in zip(metrics.names.values(), box.ap50)},
        },
        "weights": str(save_dir / "weights" / "best.pt"),
        "run_dir": str(save_dir),
        "notes": args.notes,
    }
    # Record which preprocessing the detector actually saw, next to the weights,
    # so inference can never silently mismatch it.
    pp_profile = "none"
    pp_cfg_file = Path(args.data).parent / "preprocess_config.json"
    if pp_cfg_file.exists():
        pp_profile = json.loads(pp_cfg_file.read_text()).get("profile", "standard")
        row["trained_on_preprocessed"] = json.loads(pp_cfg_file.read_text())
    for w in (save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"):
        if w.exists():
            write_meta(w, preprocess_profile=pp_profile, experiment_id=args.exp_id,
                       data=args.data, imgsz=args.imgsz)
    row["preprocess_profile"] = pp_profile

    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print("\n=== MEASURED ON HELD-OUT TEST SURVEYS ===")
    print(json.dumps(row["metrics_test"], indent=2))
    print(f"registry <- {REGISTRY}")


if __name__ == "__main__":
    main()
