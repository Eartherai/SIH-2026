"""Adapter for HuggingFace-style JSONL bounding-box datasets.

Written for `PINGEcosystem/sss-crab-pot-detection-ds` (derelict crab pots, the
closest public match to the 'ghost nets' theme of PS 26057). That repository is
ACCESS-GATED, so AQUA-SHIELD has NOT been trained on it. This adapter exists so
that a user who obtains access can convert it without writing code -- it is
deliberately not presented as a capability we have exercised.

Expected layout:
    <root>/{train,valid,test}/metadata.jsonl  +  image files alongside

Each JSONL line:
    {"file_name": "...", "objects": {"bbox": [[x,y,w,h], ...],
                                     "categories": [0, 1, ...]}}
Boxes are ABSOLUTE [x, y, w, h] (COCO-style), not normalised.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "valid", "test")


def _iter_records(meta: Path):
    for line in meta.read_text().splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def read_split(root: str | Path, split: str) -> list[dict]:
    """Return [{image_path, boxes_xyxy, categories}] for one split."""
    d = Path(root) / split
    meta = d / "metadata.jsonl"
    if not meta.exists():
        raise FileNotFoundError(
            f"{meta} not found. This dataset is access-gated; request access at "
            "https://huggingface.co/datasets/PINGEcosystem/sss-crab-pot-detection-ds")
    out = []
    for rec in _iter_records(meta):
        img = d / rec["file_name"]
        objs = rec.get("objects") or {}
        bb = np.asarray(objs.get("bbox", []), np.float32).reshape(-1, 4)
        xyxy = (np.stack([bb[:, 0], bb[:, 1], bb[:, 0] + bb[:, 2], bb[:, 1] + bb[:, 3]],
                         axis=1) if len(bb) else np.zeros((0, 4), np.float32))
        out.append({"image_path": img, "boxes_xyxy": xyxy,
                    "categories": np.asarray(objs.get("categories", []), np.int64)})
    return out


def to_yolo_labels(record: dict, width: int, height: int) -> str:
    """Convert one record to YOLO normalised text, clipped to the frame."""
    lines = []
    for (x0, y0, x1, y1), c in zip(record["boxes_xyxy"], record["categories"]):
        x0, x1 = max(0.0, x0), min(float(width), x1)
        y0, y1 = max(0.0, y0), min(float(height), y1)
        if x1 <= x0 or y1 <= y0:
            continue                       # drop degenerate boxes, never repair them
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        w, h = (x1 - x0) / width, (y1 - y0) / height
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def survey_key(file_name: str) -> str:
    """Recording id used for leakage-free splitting.

    Filenames look like `Rec09_Sensor_Depth_wcp_ss_port_00001_jpg.rf.<hash>.jpg`,
    so the leading `RecNN` identifies the survey recording. Splitting on it keeps
    consecutive pings of one object out of two different splits.
    """
    stem = Path(file_name).name
    return stem.split("_")[0] if "_" in stem else stem
