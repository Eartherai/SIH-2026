"""Plain raster ingestion (PNG / JPG / TIFF).

Sonar waterfalls arrive as ordinary images far more often than as vendor logs,
so this is the workhorse adapter. It normalises bit depth and channel count but
performs NO enhancement -- all conditioning belongs to sonar/preprocess.py so
that it stays switchable and measurable.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pgm"}


def can_read(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED


def read(path: str | Path) -> np.ndarray:
    """Return a single-channel uint8 frame."""
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode {p}")
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        # 16-bit sonar rasters: scale by the observed range, not by 65535, or a
        # frame that only uses the low bits collapses to black.
        lo, hi = np.percentile(img, [0.1, 99.9])
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
        img = (img * 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        f = img.astype(np.float32)
        lo, hi = float(f.min()), float(f.max())
        img = (((f - lo) / max(hi - lo, 1e-6)) * 255).astype(np.uint8)
    return img


def read_many(paths) -> list[tuple[str, np.ndarray]]:
    out = []
    for p in paths:
        try:
            out.append((Path(p).stem, read(p)))
        except ValueError:
            continue          # skip undecodable files rather than aborting a survey
    return out
