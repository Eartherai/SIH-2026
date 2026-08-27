"""Bounding-box geometry, NMS, and cross-tile detection merging."""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def ios_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the SMALLER box.

    Needed at tile seams: when a target is clipped by a seam, one fragment is
    much smaller than the full detection from the neighbouring tile, so their
    IoU stays low even though they are plainly the same object. IoS catches it.
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    smaller = np.minimum(area_a[:, None], area_b[None, :])
    return (inter / np.maximum(smaller, 1e-9)).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    """Greedy NMS. Returns kept indices, highest score first."""
    if len(boxes) == 0:
        return []
    order = np.argsort(-np.asarray(scores))
    boxes = np.asarray(boxes, np.float32)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = iou_matrix(boxes[i : i + 1], boxes[rest])[0]
        order = rest[ious <= iou_thr]
    return keep


def merge_tiled_detections(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                           iou_thr: float = 0.45, ios_thr: float = 0.65,
                           class_agnostic: bool = False) -> list[int]:
    """De-duplicate detections produced by overlapping tiles.

    Two detections are merged when EITHER their IoU exceeds `iou_thr` (ordinary
    duplicate) OR their intersection-over-smaller exceeds `ios_thr` (seam-clipped
    fragment contained inside the full detection). Merging is per-class unless
    `class_agnostic`, so a MILCO and a NOMBO call on the same blob are both kept
    and left for the false-positive engine to arbitrate.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    scores = np.asarray(scores, np.float32).reshape(-1)
    classes = np.asarray(classes).reshape(-1)
    if len(boxes) == 0:
        return []

    keep: list[int] = []
    groups = [np.arange(len(boxes))] if class_agnostic else \
             [np.flatnonzero(classes == c) for c in np.unique(classes)]

    for idxs in groups:
        if idxs.size == 0:
            continue
        order = idxs[np.argsort(-scores[idxs])]
        while order.size:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            bi = boxes[i : i + 1]
            dup = (iou_matrix(bi, boxes[rest])[0] > iou_thr) | \
                  (ios_matrix(bi, boxes[rest])[0] > ios_thr)
            order = rest[~dup]
    return sorted(keep, key=lambda k: -scores[k])


def xywhn_to_xyxy(rows: np.ndarray, w: int, h: int) -> np.ndarray:
    """YOLO normalised cx,cy,w,h -> absolute xyxy."""
    r = np.asarray(rows, np.float32).reshape(-1, 4)
    cx, cy, bw, bh = r[:, 0] * w, r[:, 1] * h, r[:, 2] * w, r[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
