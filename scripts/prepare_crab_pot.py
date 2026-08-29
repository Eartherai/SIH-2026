#!/usr/bin/env python3
"""Download + prepare the ghost-gear (crab-pot) SSS dataset for training.

PINGEcosystem/sss-crab-pot-detection-ds is ACCESS-GATED on HuggingFace (gate
type "auto"): a token authenticates you, but you must additionally click
"Agree and access repository" once on the dataset page before any file will
resolve. Until that is done, every image/metadata request returns HTTP 403
with x-error-code: GatedRepo -- this script fails fast with that explanation
rather than retrying or guessing.

    1. Visit https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds
    2. Click "Agree and access repository" (auto-approved, no waiting)
    3. export HF_TOKEN=hf_...        (never hardcode it, never commit it)
    4. python scripts/prepare_crab_pot.py

Splits are by RECORDING ID (the "Rec09" in each filename), not by image --
consecutive frames from one recording are highly correlated, exactly like the
MILCO/NOMBO survey-year split. This is the leakage-free split the dataset's
own train/valid/test folders do NOT guarantee (Roboflow-style exports often
split by image), so we re-derive it from the filenames ourselves and verify
disjointness with a test, matching the discipline used everywhere else in
this repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aquashield.ingestion.jsonl_bbox import survey_key, to_yolo_labels  # noqa: E402

REPO = "PINGEcosystem/sss-crab-pot-detection-ds"
CLASS_NAMES = ["Crab-Pot", "Maybe-Crab-Pot"]


def download(dest: Path, token: str) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    try:
        snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=str(dest),
                          max_workers=8, token=token)
    except GatedRepoError as e:
        raise SystemExit(
            "\nGATED: this dataset needs a one-time manual approval step that no "
            "API token can perform.\n\n"
            f"  1. Open https://huggingface.co/datasets/{REPO}\n"
            "  2. Click 'Agree and access repository' (auto-approved instantly)\n"
            "  3. Re-run this script with the same HF_TOKEN\n\n"
            f"Underlying error: {e}") from None


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/sss_crab_pot")
    ap.add_argument("--out", default="data/processed/crab_pot_yolo")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of RECORDINGS (not images) held out for val")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="data/splits",
                    help="where the split manifest is written; override in tests so a "
                         "run against throwaway data cannot clobber the real manifest")
    args = ap.parse_args()

    raw = Path(args.raw)
    if not (raw / "train" / "metadata.jsonl").exists():
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN in your environment first (export HF_TOKEN=hf_...). "
                             "Never pass it on the command line or hardcode it in a file.")
        print(f"downloading {REPO} to {raw} ...")
        download(raw, token)
    else:
        print(f"found existing download at {raw}")

    # The upstream train/valid/test folders are pooled and re-split BY RECORDING,
    # not trusted as-is, so we get a leakage-free split we can verify.
    all_records = []
    for split in ("train", "valid", "test"):
        d = raw / split
        if not (d / "metadata.jsonl").exists():
            continue
        for rec in load_jsonl(d / "metadata.jsonl"):
            rec["_dir"] = d
            rec["_recording"] = survey_key(rec["file_name"])
            all_records.append(rec)
    if not all_records:
        raise SystemExit(f"no metadata found under {raw} -- did the download complete?")

    by_rec = defaultdict(list)
    for r in all_records:
        by_rec[r["_recording"]].append(r)
    recordings = sorted(by_rec)
    print(f"{len(all_records)} images across {len(recordings)} recordings")

    import random
    rng = random.Random(args.seed)
    rng.shuffle(recordings)
    n = len(recordings)
    n_test = max(1, int(n * args.test_frac))
    n_val = max(1, int(n * args.val_frac))
    test_recs = set(recordings[:n_test])
    val_recs = set(recordings[n_test:n_test + n_val])
    train_recs = set(recordings[n_test + n_val:])

    out = Path(args.out)
    if out.exists():
        import shutil
        shutil.rmtree(out)

    manifest = {"dataset": "sss_crab_pot", "source": REPO,
               "license": "CC BY-SA 4.0 (verified from repo card 2026-08-28)",
               "split_strategy": "by RECORDING ID (filename prefix), not by image",
               "class_names": CLASS_NAMES, "splits": {}}

    import cv2
    for split_name, recs_wanted in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        img_dir = out / split_name / "images"
        lbl_dir = out / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_img = n_pos = n_obj = 0
        for rec in all_records:
            if rec["_recording"] not in recs_wanted:
                continue
            src_img = rec["_dir"] / rec["file_name"]
            if not src_img.exists():
                continue
            img = cv2.imread(str(src_img))
            if img is None:
                continue
            h, w = img.shape[:2]
            objs = rec.get("objects") or {}
            boxes = objs.get("bbox", [])
            cats = objs.get("category", objs.get("categories", []))
            cat_ids = [CLASS_NAMES.index(c) if isinstance(c, str) and c in CLASS_NAMES
                      else (int(c) if str(c).isdigit() else 0) for c in cats]
            import numpy as np
            record = {"boxes_xyxy": np.array(
                [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes], np.float32)
                if boxes else np.zeros((0, 4), np.float32),
                "categories": np.array(cat_ids, np.int64)}
            label_txt = to_yolo_labels(record, w, h)
            name = Path(rec["file_name"]).stem + ".jpg"
            cv2.imwrite(str(img_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (lbl_dir / name).with_suffix(".txt").write_text(label_txt)
            n_img += 1
            n_obj += len(boxes)
            if boxes:
                n_pos += 1
        manifest["splits"][split_name] = {"recordings": len(recs_wanted), "images": n_img,
                                          "positive_images": n_pos, "objects": n_obj}
        print(f"  {split_name:5s} recordings={len(recs_wanted):3d} images={n_img:5d} "
              f"positive={n_pos:5d} objects={n_obj:5d}")

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    splits_dir = Path(args.splits)
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "crab_pot_recording_split.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out}/data.yaml")
    print(f"wrote {splits_dir}/crab_pot_recording_split.json")


if __name__ == "__main__":
    main()
