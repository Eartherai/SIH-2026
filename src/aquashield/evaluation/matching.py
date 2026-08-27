"""Greedy IoU matching of detections to ground truth, and the metrics we report.

Two metric families are computed, because they answer different questions:

  Object-level  precision / recall / F1 over annotated targets. The usual
                academic view.

  Frame-level   how many frames containing NO target still produced an alarm.
                On a real survey, ~3 frames in 4 are empty seabed, so this is
                the number that decides whether an operator trusts the system.
                PS 26057 is explicit about minimising exactly these.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from ..detection.boxes import iou_matrix


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    matched_pairs: list[tuple[int, int, float]]   # (det_idx, gt_idx, iou)
    unmatched_dets: list[int]
    unmatched_gts: list[int]

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def match(det_boxes: np.ndarray, det_scores: np.ndarray, gt_boxes: np.ndarray,
          iou_thr: float = 0.3, det_classes=None, gt_classes=None,
          class_aware: bool = False) -> MatchResult:
    """Greedy highest-score-first matching, one GT per detection.

    IoU threshold defaults to 0.3 rather than the COCO 0.5. Justification: these
    targets are tens of pixels across, so a 3-4 px annotation offset -- well
    inside inter-annotator agreement for sonar -- drops IoU below 0.5 even for a
    visually perfect detection. 0.3 is reported explicitly everywhere so the
    number is never mistaken for a COCO mAP50.
    """
    det_boxes = np.asarray(det_boxes, np.float32).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, np.float32).reshape(-1, 4)
    n_d, n_g = len(det_boxes), len(gt_boxes)

    if n_d == 0:
        return MatchResult(0, 0, n_g, [], [], list(range(n_g)))
    if n_g == 0:
        return MatchResult(0, n_d, 0, [], list(range(n_d)), [])

    ious = iou_matrix(det_boxes, gt_boxes)
    if class_aware and det_classes is not None and gt_classes is not None:
        dc = np.asarray(det_classes).reshape(-1, 1)
        gc = np.asarray(gt_classes).reshape(1, -1)
        ious = np.where(dc == gc, ious, 0.0)

    order = np.argsort(-np.asarray(det_scores).reshape(-1))
    gt_taken = np.zeros(n_g, bool)
    pairs: list[tuple[int, int, float]] = []
    unmatched_d: list[int] = []

    for di in order:
        row = ious[di].copy()
        row[gt_taken] = -1.0
        gi = int(np.argmax(row))
        if row[gi] >= iou_thr:
            gt_taken[gi] = True
            pairs.append((int(di), gi, float(row[gi])))
        else:
            unmatched_d.append(int(di))

    return MatchResult(len(pairs), len(unmatched_d), int((~gt_taken).sum()),
                       pairs, unmatched_d, [int(i) for i in np.flatnonzero(~gt_taken)])


@dataclass
class SurveyMetrics:
    """Aggregate metrics over a set of frames."""
    n_frames: int
    n_frames_with_targets: int
    n_frames_empty: int
    n_gt_objects: int
    n_detections: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    # frame-level operational metrics
    false_alarm_frames: int          # empty frames that produced >=1 detection
    false_alarm_frame_rate: float    # of the empty frames
    fp_per_empty_frame: float
    frames_fully_clean: int          # empty frames with zero detections
    iou_threshold: float

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("precision", "recall", "f1", "false_alarm_frame_rate", "fp_per_empty_frame"):
            d[k] = round(float(d[k]), 4)
        return d


def aggregate(per_frame: list[dict], iou_thr: float) -> SurveyMetrics:
    """per_frame entries: {n_gt, n_det, tp, fp, fn}"""
    n = len(per_frame)
    empty = [f for f in per_frame if f["n_gt"] == 0]
    tp = sum(f["tp"] for f in per_frame)
    fp = sum(f["fp"] for f in per_frame)
    fn = sum(f["fn"] for f in per_frame)
    alarms = sum(1 for f in empty if f["n_det"] > 0)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return SurveyMetrics(
        n_frames=n,
        n_frames_with_targets=n - len(empty),
        n_frames_empty=len(empty),
        n_gt_objects=sum(f["n_gt"] for f in per_frame),
        n_detections=sum(f["n_det"] for f in per_frame),
        tp=tp, fp=fp, fn=fn,
        precision=p, recall=r,
        f1=(2 * p * r / (p + r) if (p + r) else 0.0),
        false_alarm_frames=alarms,
        false_alarm_frame_rate=(alarms / len(empty) if empty else 0.0),
        fp_per_empty_frame=(sum(f["n_det"] for f in empty) / len(empty) if empty else 0.0),
        frames_fully_clean=len(empty) - alarms,
        iou_threshold=iou_thr,
    )
