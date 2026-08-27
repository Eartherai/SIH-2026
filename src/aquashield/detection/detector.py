"""Model-agnostic detector interface.

Two backends are provided:

  ultralytics  - YOLO family. Fast, strong on small targets, easy ONNX export.
                 LICENCE WARNING: AGPL-3.0. See LEGAL_AND_LICENSES.md.
  torchvision  - Permissive BSD-3 alternative (FCOS/RetinaNet) for deployments
                 that cannot accept AGPL.

The rest of AQUA-SHIELD talks only to `Detection` / `Detector`, so the backend
can be replaced without touching the pipeline, the reports, or the dashboard.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from ..device import select_device
from ..sonar.tiling import tile_image, to_global
from .boxes import merge_tiled_detections


@dataclass
class Detection:
    """One raw model candidate, before any AQUA-SHIELD verification."""
    box_xyxy: tuple[float, float, float, float]
    raw_score: float                 # detector output, NOT a calibrated probability
    class_id: int
    class_name: str
    tile_index: int = 0

    @property
    def width_px(self) -> float:
        return self.box_xyxy[2] - self.box_xyxy[0]

    @property
    def height_px(self) -> float:
        return self.box_xyxy[3] - self.box_xyxy[1]

    @property
    def area_px(self) -> float:
        return max(self.width_px, 0.0) * max(self.height_px, 0.0)

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box_xyxy
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["box_xyxy"] = [round(float(v), 2) for v in self.box_xyxy]
        d["raw_score"] = round(float(self.raw_score), 4)
        return d


@dataclass
class InferenceResult:
    detections: list[Detection] = field(default_factory=list)
    n_tiles: int = 1
    tiled: bool = False
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    device: str = "cpu"
    backend: str = ""
    model_path: str = ""

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


class Detector:
    def __init__(self, weights: str | Path, backend: str = "ultralytics",
                 device: str = "auto", conf: float = 0.10, iou: float = 0.45,
                 imgsz: int = 640):
        self.weights = str(weights)
        self.backend = backend
        self.conf = conf              # deliberately LOW: recall first, then verify
        self.iou = iou
        self.imgsz = imgsz
        self._dev = select_device(device)
        self.device = self._dev.device
        self._model = None
        self.names: dict[int, str] = {}
        self._load()

    # ---------------------------------------------------------------- loading
    def _load(self) -> None:
        if self.backend == "ultralytics":
            from ultralytics import YOLO
            if not Path(self.weights).exists():
                raise FileNotFoundError(
                    f"model weights not found: {self.weights}\n"
                    "AQUA-SHIELD will not run with a placeholder model. "
                    "Train one with scripts/train.py or point --model at a real checkpoint."
                )
            self._model = YOLO(self.weights)
            self.names = dict(self._model.names)
        elif self.backend == "torchvision":
            import torch
            ckpt = torch.load(self.weights, map_location="cpu", weights_only=False)
            self._model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self._model.eval().to(self.device)
            self.names = (ckpt.get("names", {}) if isinstance(ckpt, dict) else {}) or {}
        else:
            raise ValueError(f"unknown backend '{self.backend}'")

    # -------------------------------------------------------------- inference
    @staticmethod
    def _as_bgr(im: np.ndarray) -> np.ndarray:
        """Sonar frames are single-channel; COCO-pretrained backbones expect 3.

        We replicate the grey channel rather than colour-mapping it: a colour map
        would invent chromatic structure that the acoustic data does not contain.
        """
        if im.ndim == 2:
            return np.repeat(im[:, :, None], 3, axis=2)
        if im.ndim == 3 and im.shape[2] == 1:
            return np.repeat(im, 3, axis=2)
        return im

    def _infer_batch(self, images: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Returns one (boxes_xyxy, scores, class_ids) triple per input image."""
        images = [np.ascontiguousarray(self._as_bgr(im)) for im in images]
        if self.backend == "ultralytics":
            res = self._model.predict(images, imgsz=self.imgsz, conf=self.conf,
                                      iou=self.iou, device=self.device, verbose=False)
            out = []
            for r in res:
                b = r.boxes
                if b is None or len(b) == 0:
                    out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                                np.zeros(0, np.int64)))
                else:
                    out.append((b.xyxy.cpu().numpy().astype(np.float32),
                                b.conf.cpu().numpy().astype(np.float32),
                                b.cls.cpu().numpy().astype(np.int64)))
            return out

        # torchvision
        import torch
        out = []
        with torch.inference_mode():
            for im in images:
                t = torch.from_numpy(im).float() / 255.0
                if t.ndim == 2:
                    t = t.unsqueeze(0).repeat(3, 1, 1)
                elif t.shape[-1] == 3:
                    t = t.permute(2, 0, 1)
                pred = self._model([t.to(self.device)])[0]
                keep = pred["scores"] >= self.conf
                out.append((pred["boxes"][keep].cpu().numpy().astype(np.float32),
                            pred["scores"][keep].cpu().numpy().astype(np.float32),
                            pred["labels"][keep].cpu().numpy().astype(np.int64)))
        return out

    def detect(self, image: np.ndarray, tile_size: int = 640, overlap: int = 128,
               tile_batch: int = 8) -> InferenceResult:
        """Run tiled inference over one preprocessed sonar frame."""
        t0 = time.perf_counter()
        tiles, plan = tile_image(image, tile_size, overlap)
        t1 = time.perf_counter()

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_cls: list[np.ndarray] = []
        all_tidx: list[np.ndarray] = []

        for i in range(0, len(tiles), tile_batch):
            chunk = tiles[i : i + tile_batch]
            imgs = [t.image for t in chunk]
            for t, (b, s, c) in zip(chunk, self._infer_batch(imgs)):
                if len(b):
                    all_boxes.append(to_global(b, t.offset))
                    all_scores.append(s)
                    all_cls.append(c)
                    all_tidx.append(np.full(len(b), t.index, np.int64))
        t2 = time.perf_counter()

        if all_boxes:
            boxes = np.concatenate(all_boxes)
            scores = np.concatenate(all_scores)
            cls = np.concatenate(all_cls)
            tidx = np.concatenate(all_tidx)
            keep = merge_tiled_detections(boxes, scores, cls) if plan.tiled else \
                   list(np.argsort(-scores))
            dets = [Detection(tuple(map(float, boxes[k])), float(scores[k]), int(cls[k]),
                              self.names.get(int(cls[k]), f"class_{int(cls[k])}"), int(tidx[k]))
                    for k in keep]
        else:
            dets = []
        t3 = time.perf_counter()

        return InferenceResult(
            detections=dets,
            n_tiles=len(tiles),
            tiled=plan.tiled,
            preprocess_ms=(t1 - t0) * 1000,
            inference_ms=(t2 - t1) * 1000,
            postprocess_ms=(t3 - t2) * 1000,
            device=self.device,
            backend=self.backend,
            model_path=self.weights,
        )
