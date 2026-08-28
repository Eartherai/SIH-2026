#!/usr/bin/env python3
"""Train the seabed autoencoder on NORMAL seabed only, then measure it.

Training data: patches from the TRAIN split's target-free frames (natural seabed).
The model never sees a target during training, so a target region is
out-of-distribution and reconstructs poorly.

Evaluation is real and held-out:
  - frame-level: do TEST frames that contain a target score higher than empty
    TEST frames? (ROC-AUC)
  - patch-level: do patches centred on a GT target score higher than random
    empty-seabed patches? (ROC-AUC)

Nothing here is reported unless measured; AUROC is printed and saved.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aquashield.anomaly.autoencoder import SeabedAutoencoder, AnomalyScorer  # noqa: E402
from aquashield.detection.boxes import xywhn_to_xyxy  # noqa: E402
from aquashield.device import select_device  # noqa: E402


def empty_frames(split_dir: Path):
    out = []
    for ip in sorted((split_dir / "images").glob("*.jpg")):
        lp = (split_dir / "labels" / ip.name).with_suffix(".txt")
        if not lp.exists() or not lp.read_text().strip():
            g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                out.append(g)
    return out


def target_frames(split_dir: Path):
    out = []
    for ip in sorted((split_dir / "images").glob("*.jpg")):
        lp = (split_dir / "labels" / ip.name).with_suffix(".txt")
        rows = [l.split() for l in lp.read_text().splitlines() if l.strip()] if lp.exists() else []
        if rows:
            g = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                arr = np.array([[float(v) for v in r] for r in rows], np.float32)
                out.append((g, xywhn_to_xyxy(arr[:, 1:5], g.shape[1], g.shape[0])))
    return out


def patches_from(frames, patch, stride, rng, max_patches=20000):
    P = []
    for g in frames:
        g01 = g.astype(np.float32) / 255.0
        h, w = g01.shape
        for y in range(0, max(1, h - patch + 1), stride):
            for x in range(0, max(1, w - patch + 1), stride):
                p = g01[y:y + patch, x:x + patch]
                if p.shape == (patch, patch):
                    P.append(p)
    P = np.stack(P)
    if len(P) > max_patches:
        P = P[rng.choice(len(P), max_patches, replace=False)]
    return P


def roc_auc(scores_pos, scores_neg) -> float:
    s = np.concatenate([scores_pos, scores_neg])
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed/milco_nombo_yolo")
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="models/anomaly/anomaly_ae.pt")
    args = ap.parse_args()

    dev = select_device("auto").device
    rng = np.random.default_rng(0)
    root = Path(args.data_root)

    train_empty = empty_frames(root / "train")
    print(f"normal-seabed TRAIN frames: {len(train_empty)}")
    X = patches_from(train_empty, args.patch, args.stride, rng)
    print(f"training patches: {len(X)}")
    Xt = torch.from_numpy(X[:, None].astype(np.float32))

    torch.manual_seed(0)
    model = SeabedAutoencoder().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    n = len(Xt)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, args.batch):
            xb = Xt[perm[i:i + args.batch]].to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward(); opt.step()
            tot += float(loss.item()) * len(xb)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1:2d}/{args.epochs}  mse={tot/n:.5f}")
    print(f"trained in {time.time()-t0:.1f}s on {dev}")

    # calibrate normal error distribution on a held-out chunk of train-empty patches
    model.eval()
    with torch.inference_mode():
        errs = []
        for i in range(0, n, args.batch):
            xb = Xt[i:i + args.batch].to(dev)
            out = model(xb)
            errs.append(((out - xb) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
    errs = np.concatenate(errs)
    norm_mean, norm_std = float(errs.mean()), float(errs.std())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "patch": args.patch, "stride": args.stride,
                "norm_mean": norm_mean, "norm_std": norm_std}, args.out)

    # ---- evaluation on held-out TEST ----
    scorer = AnomalyScorer(model, dev, args.patch, args.stride, norm_mean, norm_std)
    test_empty = empty_frames(root / "test")
    test_tgt = target_frames(root / "test")

    # frame-level
    s_tgt = np.array([scorer.frame_score(g.astype(np.float32) / 255.0) for g, _ in test_tgt])
    s_emp = np.array([scorer.frame_score(g.astype(np.float32) / 255.0) for g in test_empty])
    frame_auc = roc_auc(s_tgt, s_emp)

    # patch-level: GT-centred patches vs random empty patches
    pos, neg = [], []
    for g, gt in test_tgt:
        g01 = g.astype(np.float32) / 255.0
        for b in gt:
            pos.append(scorer.region_score(g01, b))
    for g in test_empty:
        g01 = g.astype(np.float32) / 255.0
        h, w = g01.shape
        for _ in range(2):
            x = int(rng.integers(0, max(1, w - args.patch)))
            y = int(rng.integers(0, max(1, h - args.patch)))
            neg.append(scorer.region_score(g01, [x, y, x + args.patch, y + args.patch]))
    patch_auc = roc_auc(np.array(pos), np.array(neg))

    print(f"\nHELD-OUT TEST evaluation:")
    print(f"  frame-level ROC-AUC (target vs empty frame): {frame_auc:.3f}")
    print(f"  patch-level ROC-AUC (GT region vs empty patch): {patch_auc:.3f}")
    print(f"  normal-seabed error: mean={norm_mean:.5f} std={norm_std:.5f}")

    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "device": dev, "patch": args.patch, "stride": args.stride,
               "epochs": args.epochs, "train_normal_frames": len(train_empty),
               "train_patches": int(len(X)),
               "frame_auc": round(frame_auc, 4), "patch_auc": round(patch_auc, 4),
               "norm_mean": norm_mean, "norm_std": norm_std,
               "n_test_target_frames": len(test_tgt), "n_test_empty_frames": len(test_empty),
               "note": "Autoencoder trained ONLY on normal-seabed patches from the TRAIN "
                       "split. AUROC measured on held-out TEST. This is an UNKNOWN-anomaly "
                       "score, independent of the supervised detector's class list."},
              open("experiments/anomaly_ae.json", "w"), indent=2)
    print("wrote experiments/anomaly_ae.json")


if __name__ == "__main__":
    main()
